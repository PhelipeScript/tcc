"""
Descrição:
    Orquestra o experimento de ASR simulado do início ao fim:

    ```
    amostra estratificada do teste -> sintetiza áudio (TTS, cacheado)
    -> transcreve com n-best real (beam search) -> calcula WER
    -> classifica em condições (original, 1-best, n-best N=3, N=5, ...)
    -> agrega métricas por condição e por modelo
    ```

    Roda os modelos já treinados (`ModelSession`) sem retreinar nada; o único
    componente novo é o Whisper (carregado uma vez, reusado em toda a amostra).

Uso:
    from src.asr_eval.engine import run
    run(AsrEvalConfig())
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.asr_eval.classify import build_nbest_input, load_sessions
from src.asr_eval.config import AsrEvalConfig
from src.asr_eval.sampling import sample_test_set
from src.asr_eval.synth import VOICES, synthesize
from src.asr_eval.transcribe_nbest import transcribe_nbest
from src.asr_eval.wer import compute_wer
from src.data_prep import console
from src.data_prep.normalizer import normalize_text
from src.training import reporting
from src.training.config import LABEL2ID
from src.training.metrics import classification_metrics

#: Condições fixas avaliadas para todo N; as condições n-best (N>1) são
#: acrescentadas dinamicamente a partir de `config.n_values`.
BASE_CONDITIONS: tuple[str, ...] = ("upper_bound", "asr_1best")


@dataclass(slots=True)
class AsrEvalResult:
    """
    Descrição:
        Resultado agregado do experimento.

    Atributos:
        records_path: Caminho do JSONL com um registro por exemplo.
        metrics: Métricas por condição e por modelo
            (`{condição: {modelo: métricas}}`).
        mean_wer: WER médio da amostra (1-best).
    """

    records_path: Path
    metrics: dict[str, dict[str, dict[str, float]]]
    mean_wer: float


def conditions_for(config: AsrEvalConfig) -> tuple[str, ...]:
    """
    Descrição:
        Monta a lista de condições avaliadas: original + 1-best + uma por
        valor de N > 1 em `config.n_values`.

    Args:
        config: Configuração do experimento.

    Retorna:
        Nomes das condições, ex. `("upper_bound", "asr_1best", "asr_nbest_3", "asr_nbest_5")`.
    """
    return BASE_CONDITIONS + tuple(
        f"asr_nbest_{n}" for n in sorted(config.n_values) if n > 1
    )


def _input_text_for(condition: str, item, hypotheses: list[str], session) -> str:
    """
    Descrição:
        Monta o texto de entrada de uma condição para um modelo específico.

    Args:
        condition: Nome da condição (`"upper_bound"`, `"asr_1best"` ou
            `"asr_nbest_<N>"`).
        item: `Record` original do dataset.
        hypotheses: Hipóteses do Whisper, na ordem do beam search.
        session: `ModelSession` do modelo (dá o tokenizador certo para o
            separador do n-best).

    Retorna:
        Texto pronto para `session.predict`, ou string vazia se não houver
        conteúdo (ex.: hipóteses insuficientes).
    """
    if condition == "upper_bound":
        return item.text
    if condition == "asr_1best":
        return normalize_text(hypotheses[0]) if hypotheses else ""
    n = int(condition.rsplit("_", 1)[1])
    return build_nbest_input(hypotheses[:n], session)


class _Accumulator:
    """
    Descrição:
        Acumula rótulo verdadeiro, predição e score por condição e por
        modelo, mantendo as três listas sempre do mesmo tamanho por par
        (condição, modelo) — evita qualquer suposição de intercalação entre
        modelos.
    """

    def __init__(self, conditions: tuple[str, ...], model_keys: tuple[str, ...]) -> None:
        self.y_true: dict[str, dict[str, list[int]]] = {
            c: {k: [] for k in model_keys} for c in conditions
        }
        self.y_pred: dict[str, dict[str, list[int]]] = {
            c: {k: [] for k in model_keys} for c in conditions
        }
        self.y_score: dict[str, dict[str, list[float]]] = {
            c: {k: [] for k in model_keys} for c in conditions
        }

    def add(self, condition: str, model_key: str, label_id: int, pred: str, prob: float) -> None:
        """
        Descrição:
            Registra uma predição de um (condição, modelo) para um exemplo.

        Args:
            condition: Nome da condição.
            model_key: Chave do modelo.
            label_id: Rótulo verdadeiro (0/1).
            pred: Classe predita (`"DD"`/`"NDD"`).
            prob: Probabilidade da classe DD.
        """
        self.y_true[condition][model_key].append(label_id)
        self.y_pred[condition][model_key].append(LABEL2ID[pred])
        self.y_score[condition][model_key].append(prob)


def run(config: AsrEvalConfig) -> AsrEvalResult:
    """
    Descrição:
        Executa o experimento completo e grava os artefatos em
        `config.output_dir`.

    Args:
        config: Configuração do experimento.

    Retorna:
        `AsrEvalResult` com o caminho dos registros e as métricas agregadas.
    """
    from faster_whisper import WhisperModel

    console.header("ASR simulado: amostragem")
    sample = sample_test_set(config)
    console.info(f"{len(sample)} exemplos amostrados de {config.data_dir / 'test.jsonl'}")

    console.header("carregando modelos")
    console.info(f"Whisper {config.whisper_size} (beam_size={config.beam_size})")
    whisper = WhisperModel(
        config.whisper_size, device=config.device, compute_type=config.compute_type
    )
    sessions = load_sessions(config.model_keys)
    console.info(f"classificadores: {', '.join(config.model_keys)}")

    conditions = conditions_for(config)
    accumulator = _Accumulator(conditions, config.model_keys)
    wer_values: list[float] = []

    console.header(f"sintetizando, transcrevendo e classificando ({len(sample)} exemplos)")
    start = time.time()
    records_path = config.output_dir / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)

    with records_path.open("w", encoding="utf-8") as records_file:
        for i, item in enumerate(sample):
            voice = VOICES[i % len(VOICES)]
            audio_path = config.audio_dir / f"{i:05d}.mp3"
            try:
                synthesize(item.text_raw, audio_path, voice=voice)
                hypotheses = transcribe_nbest(
                    whisper,
                    audio_path,
                    n=config.max_n,
                    language=config.language,
                    beam_size=config.beam_size,
                )
            except RuntimeError as exc:
                console.warn(f"[{i}] pulado: {exc}")
                continue

            if not hypotheses:
                console.warn(f"[{i}] Whisper não transcreveu nada, pulado")
                continue

            wer_1best = compute_wer(item.text_raw, hypotheses[0])
            if not np.isnan(wer_1best):
                wer_values.append(wer_1best)

            label_id = LABEL2ID[item.label]
            record_predictions: dict[str, dict[str, tuple[str, float]]] = {}

            for condition in conditions:
                condition_predictions: dict[str, tuple[str, float]] = {}
                for model_key, session in sessions.items():
                    input_text = _input_text_for(condition, item, hypotheses, session)
                    if not input_text:
                        continue
                    pred, prob = session.predict(input_text)
                    condition_predictions[model_key] = (pred, prob)
                    accumulator.add(condition, model_key, label_id, pred, prob)
                record_predictions[condition] = condition_predictions

            records_file.write(
                _record_to_json(item, hypotheses, wer_1best, record_predictions) + "\n"
            )

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                console.info(f"{i + 1}/{len(sample)} ({elapsed:.0f}s)")

    console.header("agregando métricas")
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for condition in conditions:
        metrics[condition] = {}
        for model_key in config.model_keys:
            y_true = np.asarray(accumulator.y_true[condition][model_key])
            if len(y_true) == 0:
                continue
            y_pred = np.asarray(accumulator.y_pred[condition][model_key])
            y_score = np.asarray(accumulator.y_score[condition][model_key])
            metrics[condition][model_key] = classification_metrics(y_true, y_pred, y_score)

    mean_wer = float(np.mean(wer_values)) if wer_values else float("nan")
    reporting.save_json(metrics, config.output_dir / "metrics.json")
    _write_summary_markdown(config, conditions, metrics, mean_wer, len(sample))
    console.info(f"WER médio (1-best): {mean_wer:.4f}")
    console.info(f"artefatos em {config.output_dir}")

    return AsrEvalResult(records_path=records_path, metrics=metrics, mean_wer=mean_wer)


def _record_to_json(item, hypotheses: list[str], wer_1best: float, predictions: dict) -> str:
    """
    Descrição:
        Serializa um registro do experimento em uma linha JSON.

    Args:
        item: `Record` original do dataset.
        hypotheses: Hipóteses do Whisper.
        wer_1best: WER da hipótese 1-best.
        predictions: Predições por condição e por modelo.

    Retorna:
        String JSON de uma linha.
    """
    return json.dumps(
        {
            "text_raw": item.text_raw,
            "text": item.text,
            "label": item.label,
            "source": item.source,
            "hypotheses": hypotheses,
            "wer_1best": None if wer_1best != wer_1best else round(wer_1best, 4),
            "predictions": {
                condition: {key: [pred, round(prob, 6)] for key, (pred, prob) in preds.items()}
                for condition, preds in predictions.items()
            },
        },
        ensure_ascii=False,
    )


def _write_summary_markdown(
    config: AsrEvalConfig,
    conditions: tuple[str, ...],
    metrics: dict[str, dict[str, dict[str, float]]],
    mean_wer: float,
    n_sampled: int,
) -> None:
    """
    Descrição:
        Grava a tabela-resumo do experimento em Markdown.

    Args:
        config: Configuração do experimento.
        conditions: Condições avaliadas.
        metrics: Métricas agregadas por condição e modelo.
        mean_wer: WER médio da amostra.
        n_sampled: Tamanho da amostra.
    """
    lines = [
        "# ASR simulado (TTS -> Whisper) — degradação por condição",
        "",
        f"Amostra: {n_sampled} exemplos do conjunto de teste. "
        f"Whisper {config.whisper_size}, beam_size={config.beam_size}. "
        f"WER médio (1-best): {mean_wer:.4f}.",
        "",
        "| condição | modelo | accuracy | f1_macro | roc_auc | eer |",
        "|---|---|---|---|---|---|",
    ]
    for condition in conditions:
        for model_key, values in metrics.get(condition, {}).items():
            lines.append(
                f"| {condition} | {model_key} | {values['accuracy']:.4f} | "
                f"{values['f1_macro']:.4f} | {values['roc_auc']:.4f} | {values['eer']:.4f} |"
            )
    markdown = "\n".join(lines) + "\n"
    (config.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)

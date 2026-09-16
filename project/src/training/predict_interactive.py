"""
Descrição:
    Predição interativa contra o melhor checkpoint de cada modelo do TCC.

    O usuário informa qual modelo quer testar, digita uma frase e a classe que
    espera (DD ou NDD). O script normaliza a frase como o pipeline de dados,
    roda a predição e acumula, durante a sessão, as métricas de acerto e erro
    do modelo: acurácia, taxas de falso positivo/falso negativo, precisão,
    recall e F1 da classe DD.

    A frase é normalizada automaticamente (sem acento, minúscula, sem
    pontuação) porque foi assim que o modelo foi treinado — é a entrada que um
    ASR real produziria.

    Comandos no prompt de frase:
        (vazio) ou sair/s/q  -> encerra a sessão e mostra o resumo
        modelo/m             -> volta ao menu de seleção de modelo
        metricas/me          -> mostra as métricas acumuladas até agora
        ajuda/h              -> mostra esta lista de comandos

Uso:
    uv run python -m src.training.predict_interactive
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_prep import console
from src.data_prep.normalizer import normalize_text
from src.training.config import MODELS, DEFAULT_OUTPUT_DIR

#: Ordem de exibição dos modelos no menu.
MODEL_KEYS: tuple[str, ...] = ("debertinha", "bertimbau", "albertina")


class PredictorError(RuntimeError):
    """Erro de carregamento ou execução de predição."""


@dataclass
class SessionMetrics:
    """
    Métricas de acerto e erro acumuladas durante a sessão de teste de um modelo.

    Segundos a definição do TCC: DD é a classe positiva, portanto "ativar o
    assistente quando não deveria" conta como falso positivo (FP).
    """

    tp: int = 0  #: real DD, previsto DD
    fp: int = 0  #: real NDD, previsto DD (acordou à toa)
    tn: int = 0  #: real NDD, previsto NDD
    fn: int = 0  #: real DD, previsto NDD (falhou em acordar)
    previstos: dict[str, list[str]] = field(
        default_factory=lambda: {"DD": [], "NDD": []}
    )

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def acertos(self) -> int:
        return self.tp + self.tn

    @property
    def erros(self) -> int:
        return self.fp + self.fn

    def accuracy(self) -> float:
        return self.acertos / self.total if self.total else 0.0

    def precision_dd(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    def recall_dd(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    def f1_dd(self) -> float:
        p, r = self.precision_dd(), self.recall_dd()
        return 2 * p * r / (p + r) if p + r else 0.0

    def register(self, real: str, predito: str, frase: str) -> None:
        """Registra uma predição (DD/NDD) e guarda a frase para conferência."""
        self.previstos[predito].append(f"{frase} (real {real})")
        if real == "DD" and predito == "DD":
            self.tp += 1
        elif real == "NDD" and predito == "DD":
            self.fp += 1
        elif real == "NDD" and predito == "NDD":
            self.tn += 1
        else:
            self.fn += 1

    def describe(self) -> str:
        return (
            f"acertos={self.acertos}  erros={self.erros}  "
            f"acurácia={self.accuracy():.2%}   "
            f"FP={self.fp}  FN={self.fn}"
        )


@dataclass
class ModelSession:
    """Checkpoint carregado de um modelo, pronto para predição."""

    key: str
    name: str
    threshold: float
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification

    @torch.no_grad()
    def predict(self, frase_normalizada: str) -> tuple[str, float]:
        """
        Prediz a classe da frase normalizada.

        Args:
            frase_normalizada: Texto já passado por `normalize_text`.

        Retorna:
            Tupla `(classe_predita, probabilidade_da_classe_DD)`.
        """
        inputs = self.tokenizer(
            frase_normalizada,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        prob_dd = torch.softmax(logits, dim=-1)[0, 1].item()
        predito = "DD" if prob_dd >= self.threshold else "NDD"
        return predito, prob_dd

    @classmethod
    def load(cls, key: str) -> "ModelSession":
        """
        Carrega o melhor checkpoint de um modelo e seu limiar congelado.

        Args:
            key: Chave do modelo em `config.MODELS`.

        Retorna:
            Sessão pronta para predição.

        Levanta:
            PredictorError: Se o `run_manifest.json` do modelo não existir ou
                estiver sem o melhor checkpoint — nada foi treinado ainda.
        """
        manifest_path = DEFAULT_OUTPUT_DIR / key / "run_manifest.json"
        if not manifest_path.exists():
            raise PredictorError(
                f"não achei {manifest_path}. Treine o modelo "
                f"`{key}` antes de testá-lo."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = manifest.get("result", {})
        checkpoint = Path(result["best_checkpoint"])
        threshold = float(result["decision_threshold"])

        console.detail(
            f"carregando checkpoint {checkpoint.name} (limiar {threshold:.3f})"
        )
        if not checkpoint.exists():
            raise PredictorError(
                f"checkpoint {checkpoint} não existe mais. Reexecute o treino "
                f"`{key}`."
            )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return cls(key=key, name=MODELS[key].name, threshold=threshold, tokenizer=tokenizer, model=model)


def ask_model() -> str | None:
    """
    Menu de seleção do modelo a testar.

    Retorna:
        Chave do modelo escolhido, ou `None` para sair.
    """
    while True:
        console.header("Escolha o modelo a testar")
        for i, key in enumerate(MODEL_KEYS, start=1):
            model = MODELS[key]
            console.info(f"{i}) {model.name}  [{key}]")
        console.info("s) sair")
        escolha = input("> ").strip().lower()
        if escolha in ("s", "sair", "q", ""):
            return None
        if escolha.isdigit():
            idx = int(escolha) - 1
            if 0 <= idx < len(MODEL_KEYS):
                return MODEL_KEYS[idx]
        if escolha in MODELS:
            return escolha
        console.error(f"opção desconhecida: {escolha!r}")


def ask_expected_class() -> str:
    """
    Pergunta a classe esperada da frase (verdade de referência do usuário).

    Retorna:
        `"DD"` ou `"NDD"`.
    """
    while True:
        resposta = input("Classe esperada (DD/NDD): ").strip().upper()
        if resposta in ("DD", "NDD"):
            return resposta
        console.error("responda DD ou NDD")


def show_metrics(metrics: SessionMetrics) -> None:
    """Imprime a tabela de métricas de acerto e erro acumuladas."""
    console.header("Métricas acumuladas")
    if metrics.total == 0:
        console.detail("(nenhuma frase testada ainda)")
        return
    console.table(
        ["métrica", "valor"],
        [
            ["frases testadas", str(metrics.total)],
            ["acertos", str(metrics.acertos)],
            ["erros", str(metrics.erros)],
            ["acurácia", f"{metrics.accuracy():.4f}"],
            ["falsos positivos (acordou à toa)", str(metrics.fp)],
            ["falsos negativos (falhou em acordar)", str(metrics.fn)],
            ["precisão DD", f"{metrics.precision_dd():.4f}"],
            ["recall DD", f"{metrics.recall_dd():.4f}"],
            ["F1 DD", f"{metrics.f1_dd():.4f}"],
        ],
    )
    for predito, frases in metrics.previstos.items():
        if frases:
            console.detail(f"previstos {predito}:")
            for frase in frases:
                console.info(f"  - {frase}")


def run_session(key: str) -> None:
    """
    Sessão interativa de teste de um modelo.

    Args:
        key: Chave do modelo em `config.MODELS`.
    """
    try:
        session = ModelSession.load(key)
    except PredictorError as exc:
        console.error(str(exc))
        return

    metrics = SessionMetrics()
    console.header(f"Testando {session.name} [{session.key}]")
    console.info(f"limiar de decisão: {session.threshold:.3f}")
    console.info("digite a frase (ou `ajuda` para os comandos)")

    while True:
        try:
            frase = input("\nFrase: ").strip()
        except (EOFError, KeyboardInterrupt):
            frase = "sair"
        comando = frase.lower()
        if comando in ("", "sair", "s", "q"):
            break
        if comando in ("modelo", "m"):
            return
        if comando in ("metricas", "me"):
            show_metrics(metrics)
            continue
        if comando in ("ajuda", "h"):
            console.info(
                "comandos: (vazio)/sair = encerrar | modelo = trocar modelo | "
                "metricas = ver métricas | ajuda = esta mensagem"
            )
            continue

        normalizada = normalize_text(frase)
        if not normalizada:
            console.error("a frase ficou vazia após a normalização")
            continue

        esperada = ask_expected_class()
        predita, prob_dd = session.predict(normalizada)
        acertou = predita == esperada
        marcador = "OK  " if acertou else "ERRO"
        (console.ok if acertou else console.error)(
            f"{marcador} previsto={predita}  p(DD)={prob_dd:.4f}  "
            f"(limiar {session.threshold:.3f})  esperado={esperada}"
        )
        console.detail(f"normalizada: {normalizada}")

        metrics.register(esperada, predita, normalizada)
        console.info(metrics.describe())


def main() -> int:
    """Ponto de entrada: alterna entre a seleção de modelo e a sessão de teste."""
    console.header("Predição interativa — modelos DD vs NDD do TCC")
    console.detail(
        "as métricas são acumuladas por modelo durante a sessão; troque de "
        "modelo com `modelo` e encerre com `sair`."
    )
    while True:
        key = ask_model()
        if key is None:
            break
        run_session(key)
    console.info("encerrado.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
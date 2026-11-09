"""Classificadores DD/NDD treinados: carregamento dos artefatos e inferência de uma frase.

O mesmo modelo, com o mesmo limiar, vem de uma de duas origens:
- o modelo publicado no Hugging Face (`models/ddsd-ptbr-<modelo>/`, baixado por
  `ddsd download-models`): pesos, tokenizador e `config.json` com o limiar de decisão
  congelado (`decision_threshold`), conferido com o do `treinamento.json`;
- uma execução de treino (`models/<execução>/`): `run_manifest.json`, com o limiar
  congelado (escolhido na validação) e o melhor checkpoint, e `checkpoints/checkpoint-N/`,
  com os pesos e o tokenizador salvos pelo `Trainer`.

O caminho do checkpoint é resolvido pelo nome do diretório e rebaseado na
execução local. Isso funciona tanto para caminhos relativos (treino deste
projeto) quanto para caminhos absolutos do Windows.
`Path("C:\\\\...\\\\checkpoint-N").name` no macOS devolveria a string inteira, por
isso `PureWindowsPath`.

O texto que chega aqui já está normalizado (`ddsd.text.normalization`). A
transcrição RAW nunca é alterada.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import torch

from ddsd.common.io import read_json
from ddsd.observability.logger import get_logger

log = get_logger("classificador")

MAX_LENGTH = 128


class ArtifactError(RuntimeError):
    """Artefato de treino ausente ou inválido."""


def resolve_checkpoint(run_dir: Path, recorded: str | None) -> Path:
    """Diretório do melhor checkpoint dentro de `run_dir`, qualquer que seja o SO do treino."""
    candidates = []
    if recorded:
        name = PureWindowsPath(recorded).name if "\\" in recorded else Path(recorded).name
        candidates.append(run_dir / "checkpoints" / name)
        candidates.append(run_dir / recorded)  # caminho relativo gravado por este projeto
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    existing = sorted((run_dir / "checkpoints").glob("checkpoint-*"))
    raise ArtifactError(f"checkpoint {recorded!r} não encontrado em {run_dir} "
                        f"(existentes: {[p.name for p in existing]})")


def has_classifier(run_dir: Path) -> bool:
    """Se `run_dir` tem um classificador: modelo publicado no Hugging Face ou execução de treino."""
    return (run_dir / "run_manifest.json").exists() or (run_dir / "config.json").exists()


def locate(name: str, run_dir: Path) -> tuple[float, Path]:
    """Limiar de decisão congelado e diretório dos pesos, nas duas origens."""
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        result = read_json(manifest_path)["result"]
        return float(result["decision_threshold"]), resolve_checkpoint(run_dir, result.get("best_checkpoint"))
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise ArtifactError(f"{run_dir} não tem o classificador `{name}`; rode `uv run ddsd download-models`")
    threshold = read_json(config_path).get("decision_threshold")
    if threshold is None:
        raise ArtifactError(f"{config_path} sem `decision_threshold`")
    info_path = run_dir / "treinamento.json"
    if info_path.exists() and abs(float(read_json(info_path)["limiar_de_decisao"]) - float(threshold)) > 1e-12:
        raise ArtifactError(f"{run_dir}: o limiar de `config.json` difere do de `treinamento.json`")
    return float(threshold), run_dir


@dataclass(frozen=True, slots=True)
class Prediction:
    model: str
    p_dd: float
    threshold: float
    label: str
    latency_ms: float
    n_tokens: int
    truncated: bool


class DddClassifier:
    """Um classificador carregado (pesos + tokenizador + limiar), usado por uma única thread."""

    def __init__(self, name: str, run_dir: Path, device: str = "cpu") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.run_dir = run_dir
        self.threshold, self.checkpoint = locate(name, run_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        if not self.tokenizer.is_fast:
            raise ArtifactError(f"{name}: tokenizador lento carregado de {self.checkpoint}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, dtype=torch.float32)
        self.model.eval().to(device)
        self.device = device
        label2id = {k.upper(): int(v) for k, v in self.model.config.label2id.items()}
        if label2id.get("DD") != 1:
            raise ArtifactError(f"{name}: DD não é a classe 1 no checkpoint ({label2id})")
        log.info("%s carregado de %s (limiar %.3f, %s)", name, self.checkpoint.relative_to(run_dir.parent),
                 self.threshold, device)

    @torch.inference_mode()
    def predict(self, normalized_text: str) -> Prediction:
        """P(DD) e rótulo no limiar congelado. Entrada única: sem padding (equivale ao lote com máscara)."""
        started = time.perf_counter()
        encoded = self.tokenizer(normalized_text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        n_tokens = int(encoded["input_ids"].shape[1])
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        logits = self.model(**encoded).logits[0].float()
        p_dd = float(torch.softmax(logits, dim=-1)[1])
        return Prediction(model=self.name, p_dd=p_dd, threshold=self.threshold,
                          label="DD" if p_dd >= self.threshold else "NDD",
                          latency_ms=(time.perf_counter() - started) * 1000, n_tokens=n_tokens,
                          truncated=n_tokens >= MAX_LENGTH)

    @torch.inference_mode()
    def predict_batch(self, normalized_texts: list[str], batch_size: int = 64) -> list[Prediction]:
        """Inferência em lote (avaliação offline). Com máscara de atenção, equivale à inferência individual."""
        out: list[Prediction] = []
        for start in range(0, len(normalized_texts), batch_size):
            chunk = normalized_texts[start : start + batch_size]
            encoded = self.tokenizer(chunk, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            probs = torch.softmax(self.model(**encoded).logits.float(), dim=-1)[:, 1].cpu().tolist()
            out += [Prediction(self.name, p, self.threshold, "DD" if p >= self.threshold else "NDD", 0.0, int(n),
                               int(n) >= MAX_LENGTH) for p, n in zip(probs, lengths, strict=True)]
        return out

    def warm_up(self) -> None:
        """Algumas inferências de tamanhos variados: as primeiras chamadas do PyTorch alocam e otimizam."""
        for text in ("abre o navegador", "voce viu aquele video novo que a marina mandou no grupo ontem",
                     "pausa", "entao abre o navegador e procura aquela receita de bolo de chocolate por favor"):
            for _ in range(2):
                self.predict(text)

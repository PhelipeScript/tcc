"""Configuração do treino: identidade e hiperparâmetros de cada modelo e opções da execução.

Hiperparâmetros: lr 2e-5, lote efetivo 32, 4 ou 5 épocas, `max_length` 128, warmup
de 10%, dropout 0,1 (tabela em `reports/treino/treino.md`). São os mesmos nas duas
condições (com e sem correção), então a diferença de resultado vem do corpus, não
do regime de treino.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from ddsd.common.paths import DATA_DIR, MODELS_DIR

#: **DD é a classe positiva (1).** Com isso, "ativar o assistente quando não devia"
#: é falso positivo, e precisão, recall e EER são lidos na direção que importa.
LABEL2ID: Final[dict[str, int]] = {"NDD": 0, "DD": 1}
ID2LABEL: Final[dict[int, str]] = {index: label for label, index in LABEL2ID.items()}
LABEL_NAMES: Final[tuple[str, str]] = ("NDD", "DD")
POSITIVE_ID: Final[int] = LABEL2ID["DD"]

#: Partições do corpus congelado (`ddsd corpus freeze`).
DEFAULT_DATA_DIR: Final[Path] = DATA_DIR / "corpus" / "frozen" / "splits"

Precision = Literal["auto", "bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class ModelConfig:
    """Identidade e hiperparâmetros de um checkpoint a ajustar."""

    name: str
    checkpoint: str
    family: str
    #: BERT usa `classifier_dropout`; DeBERTa V1 e V2 usam `pooler_dropout`.
    #: Ajustar o atributo errado não gera erro nenhum, só deixa de ter efeito.
    dropout_attr: str
    params_millions: float
    batch_size: int = 32
    grad_accum_steps: int = 1
    eval_batch_size: int = 128
    learning_rate: float = 2e-5
    epochs: int = 4
    max_length: int = 128
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Precisão recomendada por backend quando o modelo não suporta o padrão (bf16 misto) nele:
    #: backend -> (precisão, motivo). Nenhum patch de código: só a precisão suportada.
    precision_by_backend: dict[str, tuple[str, str]] = field(default_factory=dict)
    notes: str = ""

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


@dataclass
class TrainingConfig:
    """Opções comuns a qualquer execução."""

    seed: int = 42
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = MODELS_DIR
    text_field: Literal["text", "text_raw"] = "text"
    precision: Precision = "auto"
    allow_cpu: bool = False
    early_stopping_patience: int = 2
    metric_for_best_model: str = "f1_macro"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    tune_threshold: bool = True
    resume: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_steps: int | None = None
    run_name: str | None = None
    epochs_override: int | None = None

    def run_dir(self, model: ModelConfig) -> Path:
        """`<output_dir>/<modelo>-s<semente>` (ou `run_name`, se definido)."""
        return self.output_dir / (self.run_name or f"{model.name}-s{self.seed}")


#: Ordem de treino: do menor para o maior, para que falhas apareçam cedo e barato.
TRAIN_ORDER: Final[tuple[str, ...]] = ("debertinha", "bertimbau", "albertina")

MODELS: Final[dict[str, ModelConfig]] = {
    "bertimbau": ModelConfig(
        name="bertimbau",
        checkpoint="neuralmind/bert-base-portuguese-cased",
        family="bert",
        dropout_attr="classifier_dropout",
        params_millions=108.0,
        epochs=4,
        notes="Modelo principal da metodologia. WordPiece cased (29.794 tokens) sobre texto normalizado.",
    ),
    "albertina": ModelConfig(
        name="albertina",
        checkpoint="PORTULAN/albertina-100m-portuguese-ptbr-encoder",
        family="deberta",
        dropout_attr="pooler_dropout",
        params_millions=139.0,
        eval_batch_size=96,
        epochs=4,
        # Checkpoint salvo em bfloat16; sem forçar float32 os pesos ficariam em bf16 puro,
        # sem cópia mestre em fp32, e a otimização degradaria em silêncio.
        model_kwargs={"dtype": "float32"},
        precision_by_backend={"mps": (
            "fp32",
            "DeBERTa V1 no MPS falha em bf16 misto: attention_scores.masked_fill(..., finfo.min) estoura "
            "('value cannot be converted to type c10::BFloat16 without overflow'); fp32 roda sem alteração.")},
        notes="DeBERTa V1 com BPE byte-level herdado do deberta-base inglês; checkpoint em bfloat16.",
    ),
    "debertinha": ModelConfig(
        name="debertinha",
        checkpoint="sagui-nlp/debertinha-ptbr-xsmall",
        family="deberta-v2",
        dropout_attr="pooler_dropout",
        params_millions=40.8,
        eval_batch_size=256,
        epochs=5,
        notes="O repositório declara DebertaV2Tokenizer (lento) mas só tem tokenizer.json: use sempre o rápido.",
    ),
}

"""Construção do classificador: camada linear sobre a representação `[CLS]` (`AutoModelForSequenceClassification`).

Dois cuidados que, se esquecidos, degradam o treino sem nenhum erro:
1. dtype explícito em float32. O transformers v5 respeita o dtype gravado
   no checkpoint, e a Albertina está em bfloat16. A precisão mista fica a cargo
   do `Trainer`.
2. Atributo de dropout por família. BERT usa `classifier_dropout`; DeBERTa V1
   e V2 usam `pooler_dropout`.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel

from ddsd.observability.logger import get_logger
from ddsd.training.config import ID2LABEL, LABEL2ID, ModelConfig

log = get_logger("treino.modelo")


def build_model(model: ModelConfig) -> PreTrainedModel:
    """Checkpoint com cabeça binária (DD positivo), em float32 e com o dropout no atributo certo."""
    config = AutoConfig.from_pretrained(model.checkpoint, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
    if hasattr(config, model.dropout_attr):
        setattr(config, model.dropout_attr, model.dropout)
    else:
        log.warn("%s não tem o atributo %r; o dropout do cabeçalho não foi ajustado",
                 model.checkpoint, model.dropout_attr)
    kwargs = dict(model.model_kwargs)
    kwargs["dtype"] = torch.float32
    return AutoModelForSequenceClassification.from_pretrained(model.checkpoint, config=config, **kwargs)


def count_parameters(module: PreTrainedModel) -> dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    head = sum(p.numel() for n, p in module.named_parameters() if n.startswith(("classifier", "pooler")))
    return {"total": total, "treinaveis": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "cabecalho": head}

"""
Descrição:
    Construção do modelo de classificação a partir de um checkpoint pré-treinado.

    Segue a metodologia do TCC: uma camada linear de classificação sobre a
    representação do token `[CLS]`, treinada com entropia cruzada. Isso é o que
    `AutoModelForSequenceClassification` faz — a cabeça é inicializada
    aleatoriamente e ajustada junto com o encoder.

    Concentra dois cuidados que, se ignorados, degradam o treino silenciosamente:

    1. **Dtype explícito.** O transformers v5 respeita o `torch_dtype` gravado no
       checkpoint. A Albertina está armazenada em bfloat16; carregada sem
       intervenção, os pesos ficariam em bf16 puro, sem cópia mestre em float32,
       o que prejudica a otimização quando combinado com precisão mista. Aqui o
       carregamento é sempre em float32 e a precisão mista fica a cargo do
       `Trainer`.

    2. **Atributo de dropout por família.** BERT expõe `classifier_dropout`;
       DeBERTa V1 e V2 expõem `pooler_dropout`. Definir o atributo errado não
       gera erro nem aviso — apenas não tem efeito, e o dropout de 0,1 exigido
       pela metodologia nunca seria aplicado.

Uso:
    from src.training.modeling import build_model, count_parameters
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel

from src.data_prep import console
from src.training.config import ID2LABEL, LABEL2ID, ModelConfig


def apply_dropout(config: object, model: ModelConfig) -> bool:
    """
    Descrição:
        Ajusta o dropout do cabeçalho de classificação na configuração do modelo,
        usando o atributo correto para a família da arquitetura.

    Args:
        config: Objeto de configuração retornado por `AutoConfig.from_pretrained`.
        model: Configuração do modelo, que informa `dropout_attr` e `dropout`.

    Retorna:
        `True` se o atributo existia e foi ajustado; `False` caso contrário.
    """
    if not hasattr(config, model.dropout_attr):
        console.warn(
            f"configuração de {model.checkpoint} não tem o atributo "
            f"{model.dropout_attr!r}; dropout do cabeçalho não foi alterado"
        )
        return False
    setattr(config, model.dropout_attr, model.dropout)
    return True


def build_model(model: ModelConfig) -> PreTrainedModel:
    """
    Descrição:
        Carrega o checkpoint com um cabeçalho de classificação binária.

        Fixa `num_labels=2` e o mapeamento `id2label`/`label2id` do projeto (com
        DD como classe positiva), força o carregamento em float32 e aplica o
        dropout do cabeçalho.

        É esperado o transformers avisar que parte dos pesos foi inicializada de
        novo (`classifier`, e `pooler` no caso do DeBERTinha, cujo checkpoint é
        um `DebertaV2Model` sem cabeçalho): é exatamente a camada que o
        fine-tuning vai treinar.

    Args:
        model: Configuração do modelo a carregar.

    Retorna:
        Modelo pronto para o `Trainer`.
    """
    config = AutoConfig.from_pretrained(
        model.checkpoint,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    apply_dropout(config, model)

    kwargs = dict(model.model_kwargs)
    # O dtype é definido aqui, e não pelo config, para sobrepor o `torch_dtype`
    # gravado no checkpoint (relevante para a Albertina, salva em bfloat16).
    kwargs["dtype"] = torch.float32

    return AutoModelForSequenceClassification.from_pretrained(
        model.checkpoint, config=config, **kwargs
    )


def count_parameters(module: PreTrainedModel) -> dict[str, int]:
    """
    Descrição:
        Conta os parâmetros do modelo, separando o cabeçalho de classificação do
        restante.

    Args:
        module: Modelo carregado.

    Retorna:
        Dicionário com `total`, `treinaveis` e `cabecalho`.
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    head = sum(
        p.numel()
        for name, p in module.named_parameters()
        if name.startswith(("classifier", "pooler"))
    )
    return {"total": total, "treinaveis": trainable, "cabecalho": head}


def log_model(model: ModelConfig, module: PreTrainedModel) -> dict[str, int]:
    """
    Descrição:
        Imprime a identificação do modelo e a contagem de parâmetros.

    Args:
        model: Configuração do modelo.
        module: Modelo carregado.

    Retorna:
        Contagem de parâmetros, como em `count_parameters`.
    """
    counts = count_parameters(module)
    console.header(f"Modelo: {model.name}")
    console.info(f"checkpoint: {model.checkpoint}")
    console.info(f"família: {model.family}  dropout via `{model.dropout_attr}`={model.dropout}")
    console.info(
        f"parâmetros: {counts['total'] / 1e6:.1f}M "
        f"(cabeçalho: {counts['cabecalho'] / 1e3:.1f}K)"
    )
    console.info(
        f"lote {model.batch_size} x {model.grad_accum_steps} acum = "
        f"{model.effective_batch_size} efetivo   lr={model.learning_rate}   "
        f"épocas={model.epochs}   max_length={model.max_length}"
    )
    if model.notes:
        console.detail(model.notes)
    return counts

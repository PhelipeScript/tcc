"""
Descrição:
    Configuração do treino: identidade de cada modelo, hiperparâmetros e
    caminhos.

    Separa duas responsabilidades:

    - `ModelConfig` — o que é específico de um checkpoint (id no Hub, família da
      arquitetura, atributo de dropout, batch, learning rate, épocas). É o único
      objeto que os três scripts de treino declaram de forma diferente.
    - `TrainingConfig` — o que é comum a qualquer execução (semente, diretórios,
      precisão, early stopping, campo de texto). Igual para os três modelos.

    Os hiperparâmetros padrão seguem a tabela de hiperparâmetros definida na
    metodologia do TCC: `lr = 2e-5`, batch efetivo 32, 3 a 5 épocas,
    `max_length = 128`, warmup de 10%, dropout 0,1.

Uso:
    from src.training.config import MODELS, TrainingConfig
    cfg = MODELS["bertimbau"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

#: Raiz do projeto (`project/`), resolvida a partir deste arquivo.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

#: Diretório com as partições produzidas pelo Stage 7 do pipeline de dados.
DEFAULT_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "07_splits"

#: Diretório raiz das execuções de treino (um subdiretório por modelo).
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"

#: Mapeamento rótulo -> índice. **DD é a classe positiva (1).**
#:
#: A escolha não é arbitrária e não deve ser invertida: com DD positivo,
#: `precision`/`recall` e o EER passam a ser lidos na direção que interessa ao
#: sistema — "ativar o assistente quando não devia" é um falso positivo. Inverter
#: isso troca silenciosamente o significado de metade das métricas do TCC.
LABEL2ID: Final[dict[str, int]] = {"NDD": 0, "DD": 1}

#: Mapeamento inverso, usado na configuração do modelo e nos relatórios.
ID2LABEL: Final[dict[int, str]] = {index: label for label, index in LABEL2ID.items()}

#: Nomes das classes na ordem dos índices, para matrizes de confusão e relatórios.
LABEL_NAMES: Final[tuple[str, str]] = ("NDD", "DD")

#: Índice da classe positiva.
POSITIVE_ID: Final[int] = LABEL2ID["DD"]


@dataclass(frozen=True)
class ModelConfig:
    """
    Descrição:
        Identidade e hiperparâmetros de um checkpoint a ser ajustado.

    Atributos:
        name: Nome curto usado no diretório de saída e nos relatórios.
        checkpoint: Identificador do modelo no Hugging Face Hub.
        family: Família da arquitetura (`"bert"`, `"deberta"`, `"deberta-v2"`).
            Determina qual atributo de dropout precisa ser ajustado.
        dropout_attr: Nome do atributo de dropout do cabeçalho de classificação
            na configuração do modelo. Difere entre famílias: BERT usa
            `classifier_dropout`, DeBERTa V1 e V2 usam `pooler_dropout`. Ajustar
            o atributo errado não gera erro — simplesmente não tem efeito.
        params_millions: Número aproximado de parâmetros, em milhões. Apenas
            informativo, exibido no cabeçalho da execução.
        batch_size: Tamanho do lote por dispositivo.
        grad_accum_steps: Passos de acumulação de gradiente. O lote efetivo é
            `batch_size * grad_accum_steps` e deve ser 32 nos três modelos, para
            que a comparação entre eles seja justa.
        eval_batch_size: Tamanho do lote na avaliação (não afeta o resultado,
            só a velocidade).
        learning_rate: Taxa de aprendizado de pico.
        epochs: Número de épocas.
        max_length: Comprimento máximo em subtokens; o excedente é truncado.
        warmup_ratio: Fração dos passos totais usada como warmup linear.
        dropout: Dropout do cabeçalho de classificação.
        model_kwargs: Argumentos extras para `from_pretrained` do modelo.
        tokenizer_kwargs: Argumentos extras para `from_pretrained` do tokenizador.
        notes: Observações sobre particularidades do checkpoint.
    """

    name: str
    checkpoint: str
    family: str
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
    notes: str = ""

    @property
    def effective_batch_size(self) -> int:
        """
        Descrição:
            Calcula o lote efetivo de otimização.

        Retorna:
            `batch_size * grad_accum_steps`.
        """
        return self.batch_size * self.grad_accum_steps


@dataclass
class TrainingConfig:
    """
    Descrição:
        Configuração comum a qualquer execução de treino, independente do modelo.

    Atributos:
        seed: Semente global (Python, NumPy, PyTorch e CUDA).
        data_dir: Diretório com `train.jsonl`, `val.jsonl` e `test.jsonl`.
        output_dir: Diretório raiz das execuções.
        text_field: Campo do JSONL usado como entrada do modelo. O padrão é
            `"text"` (normalizado, que é o dataset definido para o TCC);
            `"text_raw"` permite a ablação com acentuação e pontuação
            preservadas, sem reprocessar o pipeline.
        precision: Precisão numérica: `"bf16"`, `"fp16"` ou `"fp32"`. Resolvida
            contra a capacidade real da GPU em `hardware.resolve_precision`.
        allow_cpu: Se `True`, permite executar sem CUDA. Existe apenas para
            testes de fumaça; o treino de verdade é em GPU.
        early_stopping_patience: Épocas sem melhora antes de interromper. Zero
            desliga o early stopping.
        metric_for_best_model: Métrica de seleção do melhor checkpoint.
        weight_decay: Decaimento de peso do AdamW.
        max_grad_norm: Recorte de norma do gradiente.
        lr_scheduler_type: Tipo de agendamento da taxa de aprendizado.
        save_total_limit: Número máximo de checkpoints mantidos em disco.
        dataloader_num_workers: Processos do DataLoader. Zero é o padrão porque
            no Windows workers usam `spawn`, que reimporta o script de entrada;
            como os dados já estão tokenizados em Arrow, workers pouco ajudam.
        tune_threshold: Se `True`, escolhe o limiar de decisão que maximiza o
            F1 macro **na validação** e o aplica congelado ao teste.
        max_train_samples: Limita o treino a N exemplos (teste de fumaça).
        max_eval_samples: Limita validação e teste a N exemplos.
        max_steps: Limita o número de passos de otimização (teste de fumaça).
        run_name: Nome do diretório da execução. `None` usa o nome do modelo.
    """

    seed: int = 42
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    text_field: Literal["text", "text_raw"] = "text"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    allow_cpu: bool = False
    early_stopping_patience: int = 2
    metric_for_best_model: str = "f1_macro"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    tune_threshold: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_steps: int | None = None
    run_name: str | None = None

    def run_dir(self, model: ModelConfig) -> Path:
        """
        Descrição:
            Monta o diretório de saída da execução de um modelo.

        Args:
            model: Configuração do modelo em treino.

        Retorna:
            `output_dir/<run_name ou nome do modelo>`.
        """
        return self.output_dir / (self.run_name or model.name)


#: Configuração dos três modelos do TCC.
#:
#: Os valores de lote foram dimensionados para 12 GB de VRAM (RTX A2000) com
#: `max_length = 128` e precisão mista. Nenhum dos três precisa de acumulação de
#: gradiente nessa placa: o lote efetivo 32 cabe fisicamente nos três.
MODELS: Final[dict[str, ModelConfig]] = {
    "bertimbau": ModelConfig(
        name="bertimbau",
        checkpoint="neuralmind/bert-base-portuguese-cased",
        family="bert",
        dropout_attr="classifier_dropout",
        params_millions=108.0,
        batch_size=32,
        grad_accum_steps=1,
        learning_rate=2e-5,
        epochs=4,
        notes=(
            "Modelo principal da metodologia. Vocabulário WordPiece de 29.794 "
            "tokens, cased. Como o dataset é normalizado (sem acento, minúsculo), "
            "espera-se mais fragmentação em subtokens do que no pré-treino."
        ),
    ),
    "albertina": ModelConfig(
        name="albertina",
        checkpoint="PORTULAN/albertina-100m-portuguese-ptbr-encoder",
        family="deberta",
        # DeBERTa V1 não tem `classifier_dropout`; o atributo correto é
        # `pooler_dropout`. Ajustar o atributo errado não gera erro, só não tem efeito.
        dropout_attr="pooler_dropout",
        params_millions=139.0,
        batch_size=32,
        grad_accum_steps=1,
        eval_batch_size=96,
        learning_rate=2e-5,
        epochs=4,
        # O checkpoint está armazenado em bfloat16 (`torch_dtype: bfloat16` no
        # config.json) e o transformers v5 respeita o dtype do checkpoint por
        # padrão. Sem forçar float32 aqui, os pesos ficariam em bf16 puro, sem
        # cópia mestre em fp32, degradando a otimização de forma silenciosa.
        model_kwargs={"dtype": "float32"},
        notes=(
            "É DeBERTa V1 (não V2), com tokenizador BPE byte-level herdado do "
            "deberta-base em inglês. Checkpoint armazenado em bfloat16."
        ),
    ),
    "debertinha": ModelConfig(
        name="debertinha",
        checkpoint="sagui-nlp/debertinha-ptbr-xsmall",
        family="deberta-v2",
        dropout_attr="pooler_dropout",
        params_millions=40.8,
        batch_size=32,
        grad_accum_steps=1,
        eval_batch_size=256,
        learning_rate=2e-5,
        # Modelo bem menor: 5 épocas, ainda dentro da faixa 3-5 da metodologia.
        epochs=5,
        notes=(
            "O repositório declara `DebertaV2Tokenizer` (classe lenta) mas NÃO "
            "contém `spm.model` — só `tokenizer.json`. Instanciar o tokenizador "
            "lento falha, portanto nunca passar `use_fast=False`."
        ),
    ),
}

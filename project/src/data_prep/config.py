"""
Descrição:
    Configuração central do pipeline de preparação do dataset.

    Todos os caminhos, constantes de filtragem, proporções de split e a semente
    aleatória vivem aqui. Nenhum outro módulo do pipeline deve declarar caminhos
    ou números mágicos — se um valor precisa ser ajustado, é neste arquivo.

Uso:
    from src.data_prep import config
    print(config.STAGE7_SPLITS_DIR)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Raízes do projeto
# ---------------------------------------------------------------------------

#: Raiz do projeto Python (a pasta `project/`), resolvida a partir deste arquivo.
#: `config.py` -> `data_prep/` -> `src/` -> `project/`
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

PROVIDERS_DIR: Final[Path] = PROJECT_ROOT / "src/dataset"

#: Diretório onde todos os artefatos gerados pelo pipeline são gravados.
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Fontes do dataset (SOMENTE LEITURA)
# ---------------------------------------------------------------------------

#: Provedores de LLM cujos corpora compõem o dataset, em ordem fixa para garantir
#: reprodutibilidade do merge e do split.
PROVIDERS: Final[tuple[str, ...]] = (
    "glm",
    "groq",
    "lmstudio",
    "mistral",
    "nvidia",
    "opencode",
    "openrouter",
    "qwen",
    "sonnet",
)

#: Nome do arquivo de cada classe dentro da pasta de um provedor.
CLASS_FILES: Final[dict[str, str]] = {
    "DD": "dd.jsonl",
    "NDD": "ndd.jsonl",
}

#: Rótulos do problema, em ordem canônica.
LABELS: Final[tuple[str, ...]] = ("DD", "NDD")


def provider_dir(provider: str) -> Path:
    """
    Descrição:
        Retorna o diretório de origem (somente leitura) de um provedor.

    Args:
        provider: Nome do provedor, obrigatoriamente presente em `PROVIDERS`.

    Retorna:
        Caminho absoluto para `project/<provider>/`.

    Levanta:
        ValueError: Se o provedor não estiver em `PROVIDERS`.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"provedor desconhecido: {provider!r}. Esperado um de: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS_DIR / provider


# ---------------------------------------------------------------------------
# Diretórios de saída, um por estágio
# ---------------------------------------------------------------------------

STAGE1_SNAPSHOT_DIR: Final[Path] = DATA_DIR / "01_snapshot"
STAGE2_NORMALIZED_DIR: Final[Path] = DATA_DIR / "02_normalized"
STAGE3_DEDUP_FILE_DIR: Final[Path] = DATA_DIR / "03_dedup_file"
STAGE4_MERGED_DIR: Final[Path] = DATA_DIR / "04_merged"
STAGE5_DEDUP_FINAL_DIR: Final[Path] = DATA_DIR / "05_dedup_final"
STAGE6_BALANCED_DIR: Final[Path] = DATA_DIR / "06_balanced"
STAGE7_SPLITS_DIR: Final[Path] = DATA_DIR / "07_splits"
REPORTS_DIR: Final[Path] = DATA_DIR / "reports"

#: Manifest do snapshot: contagens, mtime e sha256 de cada arquivo de origem no
#: instante da cópia. É o que torna uma rodada auditável mesmo com os scripts de
#: geração ainda escrevendo nos arquivos originais.
MANIFEST_PATH: Final[Path] = STAGE1_SNAPSHOT_DIR / "manifest.json"

#: Textos que apareceram nas duas classes e foram removidos de ambas.
CONFLICTS_PATH: Final[Path] = STAGE5_DEDUP_FINAL_DIR / "conflicts.jsonl"

#: Registros da classe majoritária descartados pelo undersample.
DISCARDED_PATH: Final[Path] = STAGE6_BALANCED_DIR / "discarded.jsonl"

# ---------------------------------------------------------------------------
# Regras de limpeza
# ---------------------------------------------------------------------------

#: Comprimento mínimo (exclusivo) do texto normalizado: `len(texto) > MIN_CHARS`.
MIN_CHARS: Final[int] = 5

#: Número mínimo (inclusivo) de palavras do texto normalizado.
MIN_WORDS: Final[int] = 2

# ---------------------------------------------------------------------------
# Split e reprodutibilidade
# ---------------------------------------------------------------------------

#: Proporções (treino, validação, teste). Devem somar 1.0.
SPLIT_RATIOS: Final[tuple[float, float, float]] = (0.8, 0.1, 0.1)

#: Nomes das partições, na mesma ordem de `SPLIT_RATIOS`.
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")

#: Semente global usada no undersample e no split.
SEED: Final[int] = 8


# ---------------------------------------------------------------------------
# Helpers de caminho
# ---------------------------------------------------------------------------


def provider_class_path(stage_dir: Path, provider: str, label: str) -> Path:
    """
    Descrição:
        Monta o caminho do arquivo de uma classe, de um provedor, dentro do
        diretório de um estágio que ainda mantém os provedores separados
        (estágios 1 a 3).

    Args:
        stage_dir: Diretório do estágio, ex.: `STAGE2_NORMALIZED_DIR`.
        provider: Nome do provedor.
        label: Rótulo da classe, `"DD"` ou `"NDD"`.

    Retorna:
        Caminho para `<stage_dir>/<provider>/<dd|ndd>.jsonl`.

    Levanta:
        KeyError: Se o rótulo não existir em `CLASS_FILES`.
    """
    return stage_dir / provider / CLASS_FILES[label]


def merged_class_path(stage_dir: Path, label: str) -> Path:
    """
    Descrição:
        Monta o caminho do arquivo de uma classe dentro do diretório de um
        estágio já consolidado, sem separação por provedor (estágios 4 a 6).

    Args:
        stage_dir: Diretório do estágio, ex.: `STAGE4_MERGED_DIR`.
        label: Rótulo da classe, `"DD"` ou `"NDD"`.

    Retorna:
        Caminho para `<stage_dir>/<dd|ndd>.jsonl`.

    Levanta:
        KeyError: Se o rótulo não existir em `CLASS_FILES`.
    """
    return stage_dir / CLASS_FILES[label]


def split_path(split_name: str) -> Path:
    """
    Descrição:
        Monta o caminho de uma partição do dataset final.

    Args:
        split_name: Nome da partição, um de `SPLIT_NAMES`.

    Retorna:
        Caminho para `data/07_splits/<split_name>.jsonl`.

    Levanta:
        ValueError: Se o nome da partição for desconhecido.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(
            f"partição desconhecida: {split_name!r}. Esperado um de: {', '.join(SPLIT_NAMES)}"
        )
    return STAGE7_SPLITS_DIR / f"{split_name}.jsonl"

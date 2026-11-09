"""Caminhos do projeto, resolvidos a partir da localização do pacote."""

from __future__ import annotations

from pathlib import Path
from typing import Final

#: Raiz do projeto (a pasta do `pyproject.toml`)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

CONFIG_DIR: Final[Path] = PROJECT_ROOT / "config"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"


def resolve_path(value: str | Path) -> Path:
    """Resolve um caminho relativo à raiz do projeto (absolutos passam intactos)."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

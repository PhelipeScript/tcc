"""Classificadores publicados no Hugging Face.

Cada repositório de `[hub] repos` (`config/pipeline.toml`) tem uma cópia local em
`models/<nome do repositório>`. `ddsd download-models` baixa todos; o pipeline baixa
sozinho o repositório de um classificador que ainda não esteja no disco. Os
repositórios são públicos, então o download não usa token.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from ddsd.common.paths import MODELS_DIR
from ddsd.observability.logger import get_logger

log = get_logger("hub")


def local_dir(repo_id: str, models_dir: Path = MODELS_DIR) -> Path:
    """Cópia local do repositório: `models/<nome do repositório>`."""
    return models_dir / repo_id.split("/")[-1]


def repo_for(path: Path, repos: Iterable[str], models_dir: Path = MODELS_DIR) -> str | None:
    """Repositório cuja cópia local contém `path`; `None` para um caminho só local (ex.: execução de treino).

    Compara caminhos resolvidos, porque a configuração resolve os links simbólicos (ex.: `models/` em outro disco).
    """
    path = path.resolve()
    for repo in repos:
        root = local_dir(repo, models_dir).resolve()
        if path == root or root in path.parents:
            return repo
    return None


def download(repo_id: str, models_dir: Path = MODELS_DIR) -> Path:
    """Baixa o repositório para a cópia local, completando o que faltar."""
    from huggingface_hub import snapshot_download

    target = local_dir(repo_id, models_dir)
    started = time.perf_counter()
    snapshot_download(repo_id, local_dir=target, token=False)
    log.info("%s em %s (%.0f s)", repo_id, target, time.perf_counter() - started)
    return target


def ensure(path: Path, repos: Iterable[str], models_dir: Path = MODELS_DIR) -> Path:
    """Devolve `path`, baixando antes o repositório que o contém se ele ainda não estiver no disco."""
    if not path.exists() and (repo := repo_for(path, repos, models_dir)) is not None:
        log.info("%s não está no disco; baixando %s do Hugging Face", path.name, repo)
        download(repo, models_dir)
    return path

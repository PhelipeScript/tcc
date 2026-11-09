"""Condições da medição registradas junto de benchmarks, treinos e builds"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


def parse_pmset(batt: str, settings: str) -> dict[str, Any]:
    """Interpreta a saída de `pmset -g batt` e de `pmset -g`."""
    if "AC Power" in batt:
        source = "tomada"
    elif "Battery Power" in batt:
        source = "bateria"
    else:
        source = "desconhecida"
    level = re.search(r"(\d+)%", batt)
    low_power = re.search(r"lowpowermode\s+(\d)", settings)
    return {"fonte_energia": source, "bateria_pct": int(level.group(1)) if level else None,
            "modo_pouca_energia": bool(int(low_power.group(1))) if low_power else None}


def power_state() -> dict[str, Any]:
    """Fonte de energia, carga da bateria e Modo de Pouca Energia; vazio fora do macOS."""
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_pmset(batt, settings)


#: Caminhos cujo conteúdo afeta resultados; alterações fora deles (ex.: `paper/`) não marcam o código como alterado.
CODE_PATHS = ("src", "config", "scripts", "pyproject.toml", "uv.lock", ".python-version")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)


def code_version(root: Path | None = None) -> dict[str, Any]:
    from ddsd.common.paths import PROJECT_ROOT

    root = root or PROJECT_ROOT
    try:
        head = _git(root, "rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None, "erro": head.stderr.strip()[:200] or "sem repositório git"}
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "erro": repr(exc)[:200]}
    changed = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    return {"commit": head.stdout.strip(), "alteracoes_nao_commitadas": bool(changed),
            "arquivos_alterados": changed[:50]}

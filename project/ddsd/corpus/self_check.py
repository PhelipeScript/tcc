"""Taxa da autoverificação do rótulo (`ddsd corpus autoverificacao`), somada dos logs da geração.

Cada lote gerado passa pelos filtros determinísticos e, depois, pela autoverificação: o mesmo modelo
reclassifica as frases em temperatura 0, e só ficam as que ele classifica com o rótulo pretendido. O log de
cada lote traz quantas frases entraram e quantas saíram em cada motivo:
- `juiz_discorda`: o modelo deu o outro rótulo;
- `juiz_invalido`: a resposta veio inválida, e o lote inteiro saiu;
- `celula_cheia`: a frase foi verificada, mas a célula já estava completa.

A taxa de discordância é juiz_discorda / (aceitas + celula_cheia + juiz_discorda + juiz_invalido).
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any

#: Linha de lote do log da geração (`generator.run_generation`).
BATCH_LINE = re.compile(r"lote \d+ (DD|NDD)\|\d+p: \+(\d+) \(faltam \d+\) rejeitadas (\{.*\})")


def strategy_of(log_name: str) -> str:
    """Estratégia pelo nome do log: as fontes de negativos difíceis terminam em `_negdif`."""
    return "negativos_dificeis" if log_name.endswith("_negdif") else "comprimento"


def batch_counts(text: str) -> dict[str, Counter]:
    """Contagens por classe (DD/NDD) das linhas de lote de um log."""
    counts = {"DD": Counter(), "NDD": Counter()}
    for line in text.splitlines():
        match = BATCH_LINE.search(line)
        if not match:
            continue
        label, accepted, rejected = match.group(1), int(match.group(2)), ast.literal_eval(match.group(3))
        counts[label]["aceitas"] += accepted
        counts[label]["lotes"] += 1
        for reason, n in rejected.items():
            counts[label][reason] += n
    return counts


def _rates(c: Counter) -> dict[str, Any]:
    judged = c["aceitas"] + c["celula_cheia"] + c["juiz_discorda"] + c["juiz_invalido"]
    return {"lotes": c["lotes"], "aceitas": c["aceitas"], "julgadas": judged, "juiz_discorda": c["juiz_discorda"],
            "juiz_invalido": c["juiz_invalido"],
            "taxa_discorda": c["juiz_discorda"] / judged if judged else None}


def self_check_stats(log_paths: list[Path]) -> dict[str, Any]:
    """Taxas por log, por estratégia e no total, separadas por classe."""
    per_log: dict[str, Any] = {}
    by_strategy: dict[str, dict[str, Counter]] = {}
    total = {"DD": Counter(), "NDD": Counter()}
    for path in sorted(log_paths):
        name = path.stem.removeprefix("geracao_")
        counts = batch_counts(path.read_text(errors="replace"))
        if not any(c["lotes"] for c in counts.values()):
            continue
        per_log[name] = {label: _rates(c) for label, c in counts.items()}
        strategy = by_strategy.setdefault(strategy_of(name), {"DD": Counter(), "NDD": Counter()})
        for label, c in counts.items():
            strategy[label].update(c)
            total[label].update(c)
    return {"por_log": per_log,
            "por_estrategia": {s: {label: _rates(c) for label, c in by.items()} for s, by in by_strategy.items()},
            "total": {label: _rates(c) for label, c in total.items()}}

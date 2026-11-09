"""De onde vêm os erros do sistema completo (`ddsd eval comparar`, conjuntos com texto de referência).

Para cada item que o sistema errou, a origem é a primeira que se aplica:
1. `segmentacao`: o detector de voz não achou fala, ou dividiu o item em mais de uma fala;
2. `classificador`: o modelo também erra no texto lido (impacto do ASR, `<modelo>:reference_norm`). O erro
   já está no texto;
3. `transcricao`: o modelo acerta no texto lido e erra na transcrição que o sistema produziu. O erro vem do
   ASR.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

ORIGINS = ("segmentacao", "classificador", "transcricao")


def origin_of(record: dict[str, Any], asr_record: dict[str, Any] | None, asr_label: str) -> str | None:
    """Origem do erro de um item, ou `None` se o sistema acertou (ou o item não tem rótulo DD/NDD)."""
    if record["label"] not in ("DD", "NDD"):
        return None
    if record["activated"] == (record["label"] == "DD"):
        return None
    if record["utterances"] != 1:
        return "segmentacao"
    if asr_record is not None and asr_record.get(f"{asr_label}:reference_norm") not in (None, record["label"]):
        return "classificador"
    return "transcricao"


def error_origins(records: list[dict[str, Any]], asr_records: list[dict[str, Any]], asr_label: str
                  ) -> dict[str, Any]:
    """Contagem por rótulo (DD: pedidos perdidos; NDD: falsas ativações) e por origem, com os itens."""
    by_id = {r["id"]: r for r in asr_records}
    counts = {label: Counter() for label in ("DD", "NDD")}
    items: dict[str, list[str]] = {origin: [] for origin in ORIGINS}
    for record in records:
        origin = origin_of(record, by_id.get(record["id"]), asr_label)
        if origin is not None:
            counts[record["label"]][origin] += 1
            items[origin].append(record["id"])
    return {"por_rotulo": {label: {o: c[o] for o in ORIGINS} for label, c in counts.items()}, "itens": items}

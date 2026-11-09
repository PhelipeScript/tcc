"""Balanceamento 1:1 por faixa de comprimento, controlando também a fonte geradora.

Modos (`balance.mode` em `config/corpus.toml`):

- `faixa_fonte_origem` (padrão): como `faixa_fonte`, mas a célula também separa
  a origem (corpus inicial × geração nova de mitigação). Assim o estilo do
  prompt de geração não prediz a classe: dentro de cada célula, DD e NDD vêm do
  mesmo modelo e da mesma versão de prompt (um prompt por classe).
- `faixa_fonte`: cada célula (faixa, fonte) mantém min(|DD|, |NDD|)
  exemplos por classe. Dentro de uma faixa, nem o comprimento nem a fonte
  predizem a classe (AUC = 0,5 por construção). Custa mais dados que o modo
  `faixa`, porque fontes que só geraram uma classe naquela faixa não contribuem.
- `faixa`: cada faixa mantém min(|DD|, |NDD|) por classe. A subamostragem da
  majoritária reproduz a mistura de fontes da minoritária quando há material,
  mas o atalho de fonte pode sobrar onde não há (no corpus inicial, AUC de
  0,58 contra 0,78 antes do balanceamento).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ddsd.corpus.config import LengthBin

#: Modos aceitos; nos modos de célula, `sources` recebe a chave do grupo (fonte ou fonte|origem).
BALANCE_MODES: frozenset[str] = frozenset({"global", "faixa", "faixa_fonte", "faixa_fonte_origem"})


def _fill(weights: Mapping[str, float], remaining: int, caps: Mapping[str, int],
          quotas: dict[str, int]) -> int:
    """Enchimento proporcional com tetos ("water-filling"); devolve o que não coube."""
    active = {k for k in caps if weights.get(k, 0.0) > 0 and quotas[k] < caps[k]}
    while remaining > 0 and active:
        total_weight = sum(weights[k] for k in active)
        exact = {k: remaining * weights[k] / total_weight for k in active}
        saturated = sorted(k for k in active if exact[k] >= caps[k] - quotas[k])
        if saturated:
            # Quem não comporta a própria parte é preenchido até o teto, e o resto
            # é redistribuído entre os demais na rodada seguinte.
            for k in saturated:
                remaining -= caps[k] - quotas[k]
                quotas[k] = caps[k]
            active.difference_update(saturated)
            continue
        floors = {k: int(exact[k]) for k in active}
        for k, value in floors.items():
            quotas[k] += value
        leftover = remaining - sum(floors.values())
        # Maior resto fracionário primeiro; o nome da chave desempata (determinístico).
        for k in sorted(active, key=lambda key: (-(exact[key] - floors[key]), key))[:leftover]:
            quotas[k] += 1
        remaining = 0
    return remaining


def largest_remainder(weights: Mapping[str, float], total: int, caps: Mapping[str, int]) -> dict[str, int]:
    """Distribui `total` unidades proporcionalmente a `weights`, sem passar de `caps`.

    Quem bate no teto tem o excedente redistribuído entre os demais, na
    proporção dos pesos. Só quando as chaves de peso positivo se esgotam é que
    as de peso zero contribuem, proporcionalmente à capacidade restante.
    Levanta `ValueError` se a soma dos tetos não comportar `total`.
    """
    if total > sum(caps.values()):
        raise ValueError(f"total {total} excede a capacidade {sum(caps.values())}")
    quotas = dict.fromkeys(caps, 0)
    remaining = _fill(weights, total, caps, quotas)
    if remaining > 0:
        spare = {k: float(caps[k] - quotas[k]) for k in caps}
        remaining = _fill(spare, remaining, caps, quotas)
    assert remaining == 0 and sum(quotas.values()) == total
    return quotas


@dataclass
class BinOutcome:
    """Resultado do balanceamento de uma faixa."""

    name: str
    before: dict[str, int]
    kept_per_class: int
    majority: str | None
    source_quotas: dict[str, int] = field(default_factory=dict)
    minority_mix: dict[str, int] = field(default_factory=dict)


@dataclass
class BalanceOutcome:
    kept: list[int]
    discarded: list[int]
    bins: list[BinOutcome]


def _balance_cells(labels: Sequence[str], sources: Sequence[str], bin_of: Sequence[int],
                   bins: Sequence[LengthBin], seed: int) -> tuple[list[int], list[BinOutcome]]:
    """Modos de célula: 1:1 dentro de cada (faixa, grupo), em que o grupo é a fonte ou fonte|origem."""
    cells: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index, (label, source) in enumerate(zip(labels, sources, strict=True)):
        cells[(bin_of[index], source, label)].append(index)
    kept: list[int] = []
    outcomes: list[BinOutcome] = []
    for bin_index, length_bin in enumerate(bins):
        before = {"DD": 0, "NDD": 0}
        quotas: dict[str, int] = {}
        for source in sorted({s for (b, s, _) in cells if b == bin_index}):
            dd = cells.get((bin_index, source, "DD"), [])
            ndd = cells.get((bin_index, source, "NDD"), [])
            before["DD"] += len(dd)
            before["NDD"] += len(ndd)
            target = min(len(dd), len(ndd))
            quotas[source] = target
            rng = random.Random(f"{seed}|{bin_index}|{source}")
            for group in (dd, ndd):
                kept.extend(group if len(group) == target else rng.sample(group, target))
        outcomes.append(BinOutcome(name=length_bin.name, before=before,
                                   kept_per_class=sum(quotas.values()), majority=None,
                                   source_quotas=quotas))
    return kept, outcomes


def balance_global(labels: Sequence[str], sources: Sequence[str], *, per_class: int | None,
                   seed: int) -> BalanceOutcome:
    """Modo `global`: 1:1 no total, sem olhar comprimento.

    Cada classe fica com `per_class` exemplos (padrão: o tamanho da minoritária), sorteados com quotas
    proporcionais à participação de cada fonte naquela classe.
    """
    by_label: dict[str, dict[str, list[int]]] = {"DD": defaultdict(list), "NDD": defaultdict(list)}
    for index, (label, source) in enumerate(zip(labels, sources, strict=True)):
        by_label[label][source].append(index)
    sizes = {label: sum(len(ix) for ix in groups.values()) for label, groups in by_label.items()}
    target = min(sizes.values()) if per_class is None else per_class
    if target > min(sizes.values()):
        raise ValueError(f"alvo de {target} por classe excede a classe menor ({sizes})")
    kept: list[int] = []
    quotas_by_label: dict[str, dict[str, int]] = {}
    for label, groups in by_label.items():
        caps = {source: len(ix) for source, ix in groups.items()}
        quotas = largest_remainder({k: float(v) for k, v in caps.items()}, target, caps)
        quotas_by_label[label] = quotas
        for source in sorted(groups):
            rng = random.Random(f"{seed}|global|{label}|{source}")
            kept.extend(rng.sample(groups[source], quotas[source]))
    kept_sorted = sorted(kept)
    kept_set = set(kept_sorted)
    outcome = BinOutcome(name="todas", before=sizes, kept_per_class=target, majority=None,
                         source_quotas={f"{lab}|{src}": q for lab, qs in quotas_by_label.items() for src, q in qs.items()})
    return BalanceOutcome(kept=kept_sorted, discarded=[i for i in range(len(labels)) if i not in kept_set],
                          bins=[outcome])


def balance_by_length(
    labels: Sequence[str],
    sources: Sequence[str],
    words: Sequence[int],
    bins: Sequence[LengthBin],
    *,
    mode: str,
    seed: int,
) -> BalanceOutcome:
    """Índices mantidos e descartados pelo balanceamento.

    A ordem dos índices mantidos segue a ordem original, para que a saída seja
    reprodutível byte a byte com a mesma semente.
    """
    if mode not in BALANCE_MODES:
        raise ValueError(f"modo de balanceamento desconhecido: {mode!r}")
    bin_of = [next(i for i, b in enumerate(bins) if b.contains(count)) for count in words]
    if mode in {"faixa_fonte", "faixa_fonte_origem"}:
        kept, outcomes = _balance_cells(labels, sources, bin_of, bins, seed)
        kept_sorted = sorted(kept)
        kept_set = set(kept_sorted)
        return BalanceOutcome(kept=kept_sorted, discarded=[i for i in range(len(labels)) if i not in kept_set],
                              bins=outcomes)

    members: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members[(bin_of[index], label)].append(index)

    kept: list[int] = []
    outcomes: list[BinOutcome] = []
    for bin_index, length_bin in enumerate(bins):
        dd, ndd = members.get((bin_index, "DD"), []), members.get((bin_index, "NDD"), [])
        target = min(len(dd), len(ndd))
        outcome = BinOutcome(name=length_bin.name, before={"DD": len(dd), "NDD": len(ndd)},
                             kept_per_class=target, majority=None)
        if target == 0:
            outcomes.append(outcome)
            continue
        minority, majority = (dd, ndd) if len(dd) <= len(ndd) else (ndd, dd)
        outcome.majority = labels[majority[0]] if len(dd) != len(ndd) else None
        kept.extend(minority)
        if len(majority) == target:
            kept.extend(majority)
            outcomes.append(outcome)
            continue

        by_source: dict[str, list[int]] = defaultdict(list)
        for index in majority:
            by_source[sources[index]].append(index)
        caps = {s: len(ix) for s, ix in by_source.items()}
        minority_mix = Counter(sources[i] for i in minority)
        weights = {s: float(minority_mix.get(s, 0)) for s in caps}
        quotas = largest_remainder(weights, target, caps)
        rng = random.Random(seed * 1000 + bin_index)
        for source in sorted(by_source):
            kept.extend(rng.sample(by_source[source], quotas[source]))
        outcome.source_quotas = quotas
        outcome.minority_mix = dict(minority_mix)
        outcomes.append(outcome)

    kept_sorted = sorted(kept)
    kept_set = set(kept_sorted)
    discarded = [i for i in range(len(labels)) if i not in kept_set]
    return BalanceOutcome(kept=kept_sorted, discarded=discarded, bins=outcomes)

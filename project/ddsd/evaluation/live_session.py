"""Pontuação de uma sessão ao vivo (`ddsd run --gravar`) contra o roteiro (`ddsd eval sessao`).

O roteiro (`docs/roteiro_sessao_ao_vivo.md`) numera cada fala e diz para quem ela é. As falas para o
computador são DD, e as demais, NDD. As falas que o sistema detectou na sessão são casadas com as linhas
do roteiro lidas nela, em ordem e pelo texto (`align`):

Unidade de pontuação: o grupo casado, ou seja, as falas que o sistema decidiu com as linhas do roteiro
dentro delas.
- O grupo é DD se alguma linha dele for DD. Ele é ativado se alguma fala dele foi decidida DD (a convenção
  de `error_origin`).
- Num grupo DD, a ativação é um pedido atendido. Num grupo NDD, é uma falsa ativação.
- As falas fora do roteiro não entram nas taxas, porque o roteiro não lhes dá rótulo. Elas são listadas à
  parte.

Origem de cada erro, a primeira que se aplica:
1. `segmentacao`: o grupo não é uma linha para uma fala (junção, divisão ou linha sem fala);
2. as de `error_origin.origin_of`, com o texto do roteiro no lugar do texto lido:
   - `classificador`: o modelo também erra no texto do roteiro;
   - `transcricao`: o modelo acerta no texto do roteiro e erra na transcrição. Sem uma transcrição manual
     do áudio, essa origem junta o erro do ASR e o desvio da leitura (improviso ou palavra trocada).
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jiwer
import numpy as np

from ddsd.common.report import MarkdownReport
from ddsd.evaluation.asr_impact import FILLED_PAUSES, without_filled_pauses
from ddsd.evaluation.error_origin import ORIGINS, origin_of
from ddsd.text.normalization import normalize_text

COMPUTER = "computador"
THINKING = "(pensando alto)"
#: Quem fala por chamada de vídeo ou telefone no roteiro. Os demais estão no quarto.
REMOTE = frozenset({"Lucas", "Fernanda"})
#: Casamento aceito só com similaridade acima disto (`difflib`, caracteres do texto normalizado).
MATCH_THRESHOLD = 0.5
#: Junção ou divisão de até 3 falas ou linhas num grupo.
MAX_JOIN = 3
#: Desempate: com a mesma similaridade, prefere uma linha para uma fala.
_TIE = 1e-6
KINDS = ("1:1", "juncao", "divisao", "sem_fala", "fora_do_roteiro")
ADDRESSEE_TYPES = ("computador", "pensando alto", "pessoa no quarto", "pessoa na chamada", "misto")

_ROW = re.compile(r"^\| (\d+) \| (.+?) → (.+?) \| (.+) \|$")
_SCENE = re.compile(r"^## (Cena .+)$")
_CASES = "- Casos difíceis de propósito:"


@dataclass(frozen=True)
class ScriptLine:
    number: int
    scene: str
    speaker: str
    addressee: str
    text: str

    @property
    def label(self) -> str:
        return "DD" if self.addressee == COMPUTER else "NDD"

    @property
    def addressee_type(self) -> str:
        if self.addressee == COMPUTER:
            return "computador"
        if self.addressee == THINKING:
            return "pensando alto"
        return "pessoa na chamada" if self.addressee in REMOTE or self.speaker in REMOTE else "pessoa no quarto"


@dataclass(frozen=True)
class Script:
    lines: tuple[ScriptLine, ...]
    #: Casos difíceis do resumo do roteiro: descrição → números das linhas.
    categories: dict[str, tuple[int, ...]]

    def spoken_by(self, speakers: Collection[str]) -> list[ScriptLine]:
        """As linhas lidas na sessão: as de quem estava falando (por padrão, só "você")."""
        return [line for line in self.lines if line.speaker in speakers]


def parse_script(path: Path) -> Script:
    """Linhas numeradas das tabelas de cena e os casos difíceis listados no resumo."""
    lines: list[ScriptLine] = []
    bullets: list[str] = []
    scene, in_cases = "", False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if match := _SCENE.match(raw):
            scene = match[1]
        elif match := _ROW.match(raw):
            lines.append(ScriptLine(int(match[1]), scene, match[2].strip(), match[3].strip().strip("*"),
                                    match[4].strip()))
        elif raw == _CASES:
            in_cases = True
        elif in_cases and raw.startswith("  - "):
            bullets.append(raw[4:])
        elif in_cases and raw.startswith("    ") and bullets:
            bullets[-1] += " " + raw.strip()
        elif in_cases:
            in_cases = False
    numbers = [line.number for line in lines]
    if not lines or numbers != list(range(1, len(lines) + 1)):
        raise ValueError(f"{path}: linhas do roteiro ausentes, fora de ordem ou repetidas")
    categories: dict[str, tuple[int, ...]] = {}
    for bullet in bullets:
        name = re.split(r"[:(]", bullet, maxsplit=1)[0].strip()
        found = tuple(int(n) for group in re.findall(r"\(([^)]*)\)", bullet) for n in re.findall(r"\d+", group))
        if not found or any(n not in numbers for n in found):
            raise ValueError(f"{path}: caso difícil sem linhas válidas: {bullet!r}")
        categories[name] = found
    return Script(tuple(lines), categories)


def similarity(a: str, b: str) -> float:
    """Similaridade de caracteres (0 a 1) entre dois textos normalizados."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass(frozen=True)
class Group:
    """Linhas do roteiro (índices) casadas com falas da sessão (índices); um dos lados pode ser vazio."""

    lines: tuple[int, ...]
    utterances: tuple[int, ...]
    similarity: float | None

    @property
    def kind(self) -> str:
        if not self.utterances:
            return "sem_fala"
        if not self.lines:
            return "fora_do_roteiro"
        if len(self.lines) > 1:
            return "juncao"
        return "divisao" if len(self.utterances) > 1 else "1:1"


def align(script: Sequence[str], spoken: Sequence[str], threshold: float = MATCH_THRESHOLD,
          max_join: int = MAX_JOIN) -> list[Group]:
    """Casamento monotônico das linhas do roteiro com as falas, pelo texto normalizado.

    Cada passo casa k linhas com uma fala (junção) ou uma linha com m falas (divisão), k, m ≤ `max_join`, ou
    deixa uma linha sem fala ou uma fala fora do roteiro. Maximiza a soma de (similaridade - `threshold`)
    nos casamentos; as lacunas valem zero.
    """
    n, m = len(script), len(spoken)
    best = np.full((n + 1, m + 1), -np.inf)
    best[0, 0] = 0.0
    back: dict[tuple[int, int], tuple[int, int, float | None]] = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            options: list[tuple[float, int, int, float | None]] = []
            for k in range(1, min(max_join, i) + 1):
                if j:
                    s = similarity(" ".join(script[i - k : i]), spoken[j - 1])
                    options.append((best[i - k, j - 1] + s - threshold - _TIE * (k - 1), i - k, j - 1, s))
            for size in range(2, min(max_join, j) + 1):
                if i:
                    s = similarity(script[i - 1], " ".join(spoken[j - size : j]))
                    options.append((best[i - 1, j - size] + s - threshold - _TIE * (size - 1), i - 1, j - size, s))
            if i:
                options.append((best[i - 1, j], i - 1, j, None))
            if j:
                options.append((best[i, j - 1], i, j - 1, None))
            chosen = options[0]
            for option in options[1:]:
                if option[0] > chosen[0]:
                    chosen = option
            best[i, j] = chosen[0]
            back[i, j] = chosen[1:]
    groups: list[Group] = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, s = back[i, j]
        groups.append(Group(tuple(range(pi, i)), tuple(range(pj, j)), s))
        i, j = pi, pj
    return groups[::-1]


def _wer(reference: str, hypothesis: str) -> float:
    return float(jiwer.wer(reference, hypothesis)) if reference else (0.0 if not hypothesis else 1.0)


def _next_line(groups: Sequence[Group], index: int, lines: Sequence[ScriptLine]) -> ScriptLine | None:
    """A próxima linha do roteiro depois do grupo `index` (ou a última, no fim): dá a cena de um improviso."""
    for later in groups[index + 1 :]:
        if later.lines:
            return lines[later.lines[0]]
    return lines[-1] if lines else None


def score(groups: Sequence[Group], lines: Sequence[ScriptLine], evidences: Sequence[dict[str, Any]],
          script: Script, classify_reference: Callable[[str], tuple[str, float]]) -> list[dict[str, Any]]:
    """Um registro por grupo: linhas, falas, decisão do sistema, decisão no texto do roteiro e origem do erro.

    `classify_reference(texto normalizado) -> (rótulo, p(DD))` é o mesmo classificador da sessão.
    """
    records: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        group_lines = [lines[i] for i in group.lines]
        group_ev = [evidences[j] for j in group.utterances]
        label = None
        if group_lines:
            label = "DD" if any(line.label == "DD" for line in group_lines) else "NDD"
        # Linhas que dão o rótulo ao grupo; numa junção DD + NDD, a NDD fica absorvida no pedido.
        own = [line for line in group_lines if line.label == label]
        anchor = group_lines[0] if group_lines else _next_line(groups, index, lines)
        reference = normalize_text(" ".join(line.text for line in group_lines))
        hypothesis = " ".join(e["normalized"] for e in group_ev)
        types = {line.addressee_type for line in own}
        both = bool(group_lines and group_ev)
        record: dict[str, Any] = {
            "linhas": [line.number for line in group_lines],
            "rotulos_das_linhas": [line.label for line in group_lines],
            "falas": [e["utterance"] for e in group_ev],
            "tipo": group.kind,
            "cena": anchor.scene if anchor else "",
            "quem": [f"{line.speaker} → {line.addressee}" for line in group_lines],
            "destinatario": (next(iter(types)) if len(types) == 1 else "misto") if own else None,
            "label": label,
            "roteiro": " ".join(line.text for line in group_lines),
            "roteiro_normalizado": reference,
            "transcricao": " ".join(e["text_raw"].strip() for e in group_ev),
            "transcricao_normalizada": hypothesis,
            "decisoes": [e["label"] for e in group_ev],
            "p_dd": [e["votes"][e["classifier"]]["p_dd"] if e["classifier"] in e["votes"] else None
                     for e in group_ev],
            "confiabilidade": [e["reliability"]["label"] for e in group_ev],
            "audio": [e["audio"] for e in group_ev],
            "activated": any(e["label"] == "DD" for e in group_ev),
            "utterances": len(group_ev),
            "similaridade": None if group.similarity is None else round(group.similarity, 4),
            "wer": _wer(reference, hypothesis) if both else None,
            "wer_sem_pausas": _wer(without_filled_pauses(reference), without_filled_pauses(hypothesis)) if both
            else None,
            "categorias": [name for name, numbers in script.categories.items()
                           if any(line.number in numbers for line in own)],
            "label_roteiro": None, "p_dd_roteiro": None, "acerto": None, "origem": None,
        }
        if label is not None:
            ref_label, ref_p = classify_reference(reference)
            record["label_roteiro"], record["p_dd_roteiro"] = ref_label, round(ref_p, 6)
            record["acerto"] = record["activated"] == (label == "DD")
            if not record["acerto"]:
                record["origem"] = ("segmentacao" if group.kind != "1:1" else
                                    origin_of(record, {"roteiro:reference_norm": ref_label}, "roteiro"))
        records.append(record)
    return records


def _rate(records: Sequence[dict[str, Any]], key: str = "activated") -> dict[str, Any]:
    hits = sum(bool(r[key]) for r in records)
    return {"n": len(records), "ativados": hits, "taxa": hits / len(records) if records else None}


def _by_label(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {label: _rate([r for r in records if r["label"] == label]) for label in ("DD", "NDD")}


def summarize(records: Sequence[dict[str, Any]], spoken: Sequence[ScriptLine], script: Script) -> dict[str, Any]:
    """Taxas por grupo (no sistema e no texto do roteiro), casamento, origem dos erros e recortes."""
    scored = [r for r in records if r["label"] is not None]
    on_script = [dict(r, activated=r["label_roteiro"] == "DD") for r in scored]
    by_kind = {kind: [r for r in records if r["tipo"] == kind] for kind in KINDS}
    errors = {label: Counter(r["origem"] for r in scored if r["label"] == label and r["origem"])
              for label in ("DD", "NDD")}
    # Contando por linha: toda linha NDD dentro de um grupo ativado (inclusive a absorvida num pedido).
    ndd_in_activated = [n for r in scored if r["activated"]
                        for n, label in zip(r["linhas"], r["rotulos_das_linhas"], strict=True) if label == "NDD"]
    absorbed = [n for r in scored if r["label"] == "DD"
                for n, label in zip(r["linhas"], r["rotulos_das_linhas"], strict=True) if label == "NDD"]
    matched = [r for r in scored if r["wer"] is not None]
    refs = [r["roteiro_normalizado"] for r in matched]
    hyps = [r["transcricao_normalizada"] for r in matched]
    pairs_np = [(without_filled_pauses(a), without_filled_pauses(b)) for a, b in zip(refs, hyps, strict=True)]
    pairs_np = [(a, b) for a, b in pairs_np if a]
    sims = sorted((r["similaridade"], r["linhas"]) for r in scored if r["similaridade"] is not None)
    extras = by_kind["fora_do_roteiro"]
    spoken_numbers = {line.number for line in spoken}
    others = [line for line in script.lines if line.number not in spoken_numbers]
    return {
        "linhas_do_roteiro": len(script.lines),
        "linhas_lidas": {"n": len(spoken), "DD": sum(line.label == "DD" for line in spoken),
                         "NDD": sum(line.label == "NDD" for line in spoken)},
        "linhas_nao_lidas": {"n": len(others), "falantes": sorted({line.speaker for line in others}),
                             "DD": [line.number for line in others if line.label == "DD"]},
        "falas_detectadas": sum(r["utterances"] for r in records),
        "pedidos_atendidos": _rate([r for r in scored if r["label"] == "DD"]),
        "falsas_ativacoes": _rate([r for r in scored if r["label"] == "NDD"]),
        "no_roteiro": {"pedidos_atendidos": _rate([r for r in on_script if r["label"] == "DD"]),
                       "falsas_ativacoes": _rate([r for r in on_script if r["label"] == "NDD"]),
                       "pedidos_perdidos": [r["linhas"] for r in on_script if r["label"] == "DD" and not r["activated"]],
                       "linhas_ativadas_ndd": [r["linhas"] for r in on_script if r["label"] == "NDD" and r["activated"]]},
        "por_linha": {"linhas_ndd": sum(line.label == "NDD" for line in spoken),
                      "linhas_ndd_em_falas_ativadas": ndd_in_activated, "absorvidas_em_pedidos": absorbed},
        "casamento": {kind: len(rs) for kind, rs in by_kind.items()}
        | {"juncoes": [r["linhas"] for r in by_kind["juncao"]], "divisoes": [r["linhas"] for r in by_kind["divisao"]],
           "linhas_sem_fala": [r["linhas"] for r in by_kind["sem_fala"]],
           "similaridade_minima": sims[0][0] if sims else None,
           "linhas_da_similaridade_minima": sims[0][1] if sims else None},
        "erros": {"por_rotulo": {label: {o: c[o] for o in ORIGINS} for label, c in errors.items()},
                  "itens": [{"linhas": r["linhas"], "falas": r["falas"], "label": r["label"], "origem": r["origem"]}
                            for r in scored if r["origem"]]},
        "por_destinatario": {kind: _rate([r for r in scored if r["destinatario"] == kind])
                             for kind in ADDRESSEE_TYPES if any(r["destinatario"] == kind for r in scored)},
        "por_cena": {scene: _by_label([r for r in scored if r["cena"] == scene])
                     for scene in dict.fromkeys(r["cena"] for r in scored)},
        "por_caso_dificil": {
            name: _by_label([r for r in scored if name in r["categorias"]])
            | {"linhas_do_roteiro": len(numbers), "linhas_lidas": sum(n in spoken_numbers for n in numbers),
               "nao_lidas": [n for n in numbers if n not in spoken_numbers]}
            for name, numbers in script.categories.items()},
        "fora_do_roteiro": {"falas": sum(r["utterances"] for r in extras),
                            "ativadas": sum(d == "DD" for r in extras for d in r["decisoes"])},
        "wer": float(jiwer.wer(refs, hyps)) if refs else None,
        "wer_sem_pausas": float(jiwer.wer([a for a, _ in pairs_np], [b for _, b in pairs_np])) if pairs_np else None,
        "grupos_com_wer": len(matched),
    }


def latency_summary(evidences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fim da fala → decisão, Whisper e classificador (p50, p95 e máximo, em segundos)."""
    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {"n": len(values), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                "max": float(max(values))}

    finals = [next(t for t in e["tasks"] if t["task"].endswith("-F")) for e in evidences]
    return {"fim_da_fala_a_decisao_s": stats([e["latency_s"] for e in evidences if e["latency_s"] is not None]),
            "whisper_s": stats([t["asr_latency_s"] for t in finals if t.get("asr_latency_s") is not None]),
            "classificador_s": stats([p["latency_ms"] / 1000 for t in finals
                                      for p in (t.get("predictions") or {}).values()
                                      if p.get("latency_ms") is not None])}


def session_checks(evidences: Sequence[dict[str, Any]], session: dict[str, Any],
                   audio_s: dict[str, float]) -> dict[str, Any]:
    """Saúde da sessão: código, classificador, falhas, limite de 30 s, áudio gravado e resumo do gravador."""
    meta = session["metadados"]
    active = meta["config"]["classifiers"]["active"]
    spans = {e["audio"]: e["span_s"][1] - e["span_s"][0] for e in evidences}
    return {
        "codigo": meta["codigo"],
        "classificador_ativo": active,
        "classificador_em_todas": all(e["classifier"] == active
                                      and (e.get("contexto") or {}).get("classificador", active) == active
                                      for e in evidences),
        "motivos": dict(Counter(e["reason"] for e in evidences)),
        "fechamento": dict(Counter(e["close_reason"] for e in evidences)),
        "prazo_estourado": sum(bool(e["timed_out"]) for e in evidences),
        "confiabilidade": dict(Counter(e["reliability"]["label"] for e in evidences)),
        "audios": len(audio_s),
        "audios_com_duracao_do_trecho": sum(abs(audio_s.get(name, -1.0) - dur) <= 0.05 for name, dur in spans.items()),
        "gravador": session.get("resumo"),
    }


# ----------------------------------------------------------------------------------------------- relatório
def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}".replace(".", ",") + "%"


def _frac(rate: dict[str, Any]) -> str:
    return f"{rate['ativados']} de {rate['n']} ({_pct(rate['taxa'])})" if rate["n"] else "-"


def _num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _lines(numbers: Sequence[int]) -> str:
    return " + ".join(f"{n:02d}" for n in numbers)


def _listing(numbers: Sequence[int]) -> str:
    return ", ".join(f"{n:02d}" for n in numbers) or "nenhuma"


def _decision(record: dict[str, Any]) -> str:
    return " + ".join(f"{d} ({_num(p)})" for d, p in zip(record["decisoes"], record["p_dd"], strict=True))


def build_report(payload: dict[str, Any], records: Sequence[dict[str, Any]]) -> MarkdownReport:
    """Relatório em Markdown da sessão pontuada (`reports/sessao_ao_vivo/resumo.md`)."""
    s, checks, lat, parity = payload["resumo"], payload["verificacoes"], payload["latencia"], payload["paridade"]
    start, end = datetime.fromisoformat(payload["inicio"]), datetime.fromisoformat(payload["fim"])
    minutes = (end - start).total_seconds() / 60
    clf = payload["classificador"]
    report = MarkdownReport(f"Sessão ao vivo com roteiro - {payload['sessao']}")
    report.paragraph(
        f"Sessão de {start:%Y-%m-%d}, das {start:%H:%M} às {end:%H:%M} ({_num(minutes, 1)} min), com o sistema "
        f"final (`ddsd run --gravar`, código `{checks['codigo']['commit'][:7]}`"
        f"{', sem alterações' if not checks['codigo']['alteracoes_nao_commitadas'] else ', COM ALTERAÇÕES'}), "
        f"o classificador `{clf['execucao']}` (limiar {_num(clf['limiar'], 3)}) e o Whisper "
        f"`{payload['asr']['modelo']}` ({payload['asr']['backend']}, semente {payload['asr']['semente']}). "
        f"Roteiro: `{payload['roteiro']}`. Pontuação: `ddsd eval sessao` (código `{payload['codigo']['commit'][:7]}`"
        f"{', COM ALTERAÇÕES' if payload['codigo']['alteracoes_nao_commitadas'] else ''}).")
    report.heading("Protocolo realizado")
    lidas, nao = s["linhas_lidas"], s["linhas_nao_lidas"]
    who = " e ".join(f"\"{name}\"" for name in payload["falantes"])
    items = [f"Lidas na sessão: só as linhas de {who} no roteiro, {lidas['n']} das {s['linhas_do_roteiro']} "
             f"({lidas['DD']} DD e {lidas['NDD']} NDD)."]
    if nao["n"]:
        items.append(f"As {nao['n']} falas das outras pessoas ({', '.join(nao['falantes'])}) não foram lidas. "
                     f"Falas dirigidas ao computador entre elas, não testadas: {_listing(nao['DD'])}.")
        for name, cut in s["por_caso_dificil"].items():
            if cut["nao_lidas"]:
                items.append(f"Caso difícil \"{name}\": {cut['linhas_lidas']} de {cut['linhas_do_roteiro']} "
                             f"linhas lidas (não lidas: {_listing(cut['nao_lidas'])}).")
    report.bullets(items)
    report.heading("Casamento com o roteiro")
    c = s["casamento"]
    def count(n: int, one: str, many: str, groups: Sequence[Sequence[int]] = ()) -> str:
        text = f"nenhuma {one}" if n == 0 else (f"1 {one}" if n == 1 else f"{n} {many}")
        return text + (f" ({'; '.join(_lines(x) for x in groups)})" if groups else "")

    report.bullets([
        f"{s['falas_detectadas']} falas detectadas para {lidas['n']} linhas lidas: {c['1:1']} casadas uma a uma, "
        f"{count(c['juncao'], 'junção', 'junções', c['juncoes'])}, "
        f"{count(c['divisao'], 'divisão', 'divisões', c['divisoes'])}, "
        f"{count(c['sem_fala'], 'linha sem fala', 'linhas sem fala', c['linhas_sem_fala'])} e "
        f"{count(c['fora_do_roteiro'], 'fala fora do roteiro', 'falas fora do roteiro')}.",
        f"Menor similaridade aceita num casamento: {_num(c['similaridade_minima'])} (linha "
        f"{_lines(c['linhas_da_similaridade_minima'] or [])}); o limiar é {_num(MATCH_THRESHOLD)}.",
    ])
    report.heading("Resultado")
    report.table(["medida", "no texto do roteiro", "na sessão (transcrição do Whisper)"], [
        ["Pedidos atendidos (DD)", _frac(s["no_roteiro"]["pedidos_atendidos"]), _frac(s["pedidos_atendidos"])],
        ["Falsas ativações (NDD)", _frac(s["no_roteiro"]["falsas_ativacoes"]), _frac(s["falsas_ativacoes"])],
    ])
    by_line = s["por_linha"]
    report.paragraph(
        "Unidade: o grupo casado (as falas que o sistema decidiu, com as linhas do roteiro dentro delas). O grupo "
        "é DD se alguma linha dele for DD, e é ativado se alguma fala dele foi decidida DD. \"No texto do "
        "roteiro\" é o mesmo classificador julgando o texto que deveria ter sido lido. As falas fora do roteiro "
        "não entram nas taxas.")
    lost, fired = s["no_roteiro"]["pedidos_perdidos"], s["no_roteiro"]["linhas_ativadas_ndd"]
    report.paragraph(
        "No texto do roteiro, o classificador perderia "
        + ("nenhum pedido" if not lost else ("o pedido " if len(lost) == 1 else "os pedidos ")
           + ", ".join(_lines(x) for x in lost))
        + " e ativaria " + ("em nenhuma linha NDD" if not fired else ("na linha NDD " if len(fired) == 1 else
                                                                       "nas linhas NDD ")
                           + ", ".join(_lines(x) for x in fired)) + ".")
    absorbed = by_line["absorvidas_em_pedidos"]
    if absorbed:
        verb = "foi dita" if len(absorbed) == 1 else "foram ditas"
        report.paragraph(
            f"Contando por linha, as linhas NDD dentro de falas ativadas são "
            f"{len(by_line['linhas_ndd_em_falas_ativadas'])} de {by_line['linhas_ndd']} "
            f"({_listing(by_line['linhas_ndd_em_falas_ativadas'])}), porque "
            f"{'a linha' if len(absorbed) == 1 else 'as linhas'} {_listing(absorbed)} {verb} na mesma fala que "
            "um pedido ao computador.")
    report.heading("Erros e origem")
    wrong = [r for r in records if r["origem"]]
    report.table(["linha", "quem → para quem", "roteiro", "transcrição", "decisão (p DD)", "no roteiro (p DD)",
                  "origem", "áudio"],
                 [[_lines(r["linhas"]), " + ".join(r["quem"]), r["roteiro"], r["transcricao"], _decision(r),
                   f"{r['label_roteiro']} ({_num(r['p_dd_roteiro'])})", r["origem"], ", ".join(r["audio"])]
                  for r in wrong])
    counts = s["erros"]["por_rotulo"]
    report.bullets([
        f"Pedidos perdidos (DD): {sum(counts['DD'].values())}; "
        + ", ".join(f"{o} {counts['DD'][o]}" for o in ORIGINS) + ".",
        f"Falsas ativações (NDD): {sum(counts['NDD'].values())}; "
        + ", ".join(f"{o} {counts['NDD'][o]}" for o in ORIGINS) + ".",
        "`classificador`: o modelo erra também no texto do roteiro. `transcricao`: acerta no texto do roteiro e "
        "erra na transcrição. Sem uma transcrição manual, essa origem junta o erro do ASR e o desvio da leitura "
        f"(improviso ou palavra trocada); para separar, é preciso ouvir os áudios listados, em "
        f"`data/sessoes/{payload['sessao']}/`. `segmentacao`: junção, divisão ou linha sem fala.",
    ])
    report.heading("Recortes")
    report.paragraph("Por destinatário da fala (grupos casados):")
    report.table(["destinatário", "grupos", "ativados", "taxa"],
                 [[k, v["n"], v["ativados"], _pct(v["taxa"])] for k, v in s["por_destinatario"].items()])
    report.paragraph("Casos difíceis do roteiro, só nas linhas lidas:")
    report.table(["caso", "linhas lidas", "DD: pedidos atendidos", "NDD: falsas ativações"],
                 [[name, f"{cut['linhas_lidas']} de {cut['linhas_do_roteiro']}", _frac(cut["DD"]), _frac(cut["NDD"])]
                  for name, cut in s["por_caso_dificil"].items()])
    report.paragraph("Por cena:")
    report.table(["cena", "pedidos atendidos", "falsas ativações"],
                 [[scene, _frac(cut["DD"]), _frac(cut["NDD"])] for scene, cut in s["por_cena"].items()])
    report.heading("Falas fora do roteiro")
    extras = [r for r in records if r["tipo"] == "fora_do_roteiro"]
    report.paragraph(f"{s['fora_do_roteiro']['falas']} falas sem linha no roteiro (improvisos de quem leu); "
                     f"{s['fora_do_roteiro']['ativadas']} delas ativaram o sistema.")
    report.table(["fala", "cena", "transcrição", "decisão (p DD)", "confiabilidade"],
                 [[", ".join(r["falas"]), r["cena"], r["transcricao"], _decision(r), ", ".join(r["confiabilidade"])]
                  for r in extras])
    report.heading("Transcrição")
    report.bullets([
        f"WER contra o roteiro: {_pct(s['wer'])} ({_pct(s['wer_sem_pausas'])} sem as pausas preenchidas: "
        f"{', '.join(sorted(FILLED_PAUSES))}), em {s['grupos_com_wer']} grupos casados.",
        "O WER mede a transcrição contra o texto do roteiro. Por isso inclui os desvios da própria leitura "
        "(improvisos e palavras trocadas), além dos erros do ASR.",
    ])
    report.heading("Tempo de resposta")
    rows = [("Fim da fala → decisão (inclui a espera de 0,7 s do detector de voz)", lat["fim_da_fala_a_decisao_s"]),
            ("Whisper", lat["whisper_s"]), ("Classificador", lat["classificador_s"])]
    report.table(["etapa", "falas", "p50", "p95", "máximo"],
                 [[name, v["n"], f"{_num(v['p50'], 3)} s", f"{_num(v['p95'], 3)} s", f"{_num(v['max'], 3)} s"]
                  for name, v in rows if v])
    report.heading("Verificações")
    rec = checks["gravador"] or {}
    report.bullets([
        f"Paridade: julgando de novo as {parity['falas']} transcrições, o classificador repetiu {parity['identicas']} "
        f"decisões (max |Δp| {parity['max_delta_p']:.2g}).",
        f"Classificador ativo em todas as falas: {'sim' if checks['classificador_em_todas'] else 'não'} "
        f"(`{checks['classificador_ativo']}`). Motivos das decisões: "
        + ", ".join(f"{k} {v}" for k, v in checks["motivos"].items())
        + ". Fechamento das falas: " + ", ".join(f"{k} {v}" for k, v in checks["fechamento"].items())
        + f". Prazo de segurança estourado: {checks['prazo_estourado']}.",
        f"Áudio: {checks['audios']} arquivos WAV, {checks['audios_com_duracao_do_trecho']} com a duração do trecho "
        "da decisão (±0,05 s).",
        f"Gravador: {rec.get('erros')} erros de gravação, {rec.get('pendentes_no_fim')} pendentes no fim, thread "
        f"{'encerrada' if rec.get('thread_encerrada') else 'ativa'}.",
        "Transbordamento da captura: não registrado (a gravação da sessão não guarda o contador do painel).",
        "Confiabilidade das transcrições: " + ", ".join(f"{k} {v}" for k, v in checks["confiabilidade"].items()) + ".",
        f"SHA-256 de `evidencias.jsonl`: `{payload['evidencias_sha256']}`; do roteiro: `{payload['roteiro_sha256']}`.",
    ])
    report.heading("Todas as falas")
    report.table(["linha", "fala", "quem → para quem", "roteiro", "transcrição", "decisão (p DD)", "certo"],
                 [[_lines(r["linhas"]) or "-", ", ".join(r["falas"]) or "-", " + ".join(r["quem"]) or "fora do roteiro",
                   r["roteiro"] or "-", r["transcricao"] or "-", _decision(r) or "-",
                   "-" if r["acerto"] is None else ("sim" if r["acerto"] else "não")] for r in records])
    return report

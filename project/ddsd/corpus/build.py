"""Construção do corpus: fontes → filtros → dedup exato → conflitos → dedup aproximado
→ balanceamento por faixa → split, com relatório e manifesto.

Tudo roda em memória (o corpus tem poucos MB). Cada descarte vai para um JSONL de
artefatos com o motivo, para que qualquer número do relatório possa ser auditado.
"""

from __future__ import annotations

import platform
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ddsd.common.io import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from ddsd.common.report import MarkdownReport
from ddsd.common.system import code_version
from ddsd.corpus import diagnostics as diag
from ddsd.corpus.balance import balance_by_length, balance_global
from ddsd.corpus.config import LABELS, CorpusConfig, FilterConfig
from ddsd.corpus.near_dedup import connected_groups, find_near_duplicates, tfidf_matrix, NearDuplicatePairs
from ddsd.corpus.sources import SourceRow, load_sources
from ddsd.corpus.split import SPLIT_NAMES, stratified_split, stratum_keys, verify_disjoint
from ddsd.observability.logger import get_logger
from ddsd.text.normalization import normalize_text, word_count

log = get_logger("corpus.build")


@dataclass(slots=True)
class Item:
    """Registro em processamento: linha de origem + texto normalizado."""

    row: SourceRow
    text: str
    words: int
    bin: str

    @property
    def label(self) -> str:
        return self.row.label

    def ref(self) -> str:
        return f"{self.row.file}:{self.row.line}"

    def to_split_row(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_raw": self.row.text_raw,
            "label": self.row.label,
            "source": self.row.source,
            "origin": self.row.origin,
            "strategy": self.row.strategy,
            "model": self.row.model,
            "words": self.words,
            "bin": self.bin,
        }


#: Registro que é uma lista serializada de várias frases (defeito de gerador do corpus inicial), ex.:
#: "['Por gentileza, poderia capturar a tela?', 'Seria possível...']". Abre com colchete e aspas ou
#: tem dois ou mais separadores de lista; uma citação isolada ("pastas 'Projetos', 'Notas'") passa.
_LIST_START = re.compile(r"^\s*\[\s*['\"]")
_LIST_SEPARATOR = re.compile(r"['\"]\s*,\s*['\"]")


def rejection_reason(text: str, filters: FilterConfig, raw: str = "") -> str | None:
    """Motivo de descarte do texto normalizado (e do original, para defeitos de formato), ou `None`."""
    if raw and (_LIST_START.search(raw) or len(_LIST_SEPARATOR.findall(raw)) >= 2):
        return "lista_serializada"
    if not text:
        return "vazio"
    words = word_count(text)
    if len(text) <= filters.min_chars or words < filters.min_words:
        return "tamanho"
    if words > filters.max_words:
        return "longo_demais"
    if filters.require_ascii and not text.isascii():
        return "nao_ascii"
    return None


def _funnel_row(stage: str, items: list[Item], note: str = "") -> dict[str, Any]:
    counts = Counter(item.label for item in items)
    return {"etapa": stage, "DD": counts["DD"], "NDD": counts["NDD"], "total": len(items), "nota": note}


def _shortcut_diagnostics(items: list[Item]) -> dict[str, Any]:
    labels = [i.label for i in items]
    words = [i.words for i in items]
    bins = [i.bin for i in items]
    source_auc, source_per_bin = diag.categorical_auc_within_bins(labels, [i.row.source for i in items], bins)
    origin_auc, origin_per_bin = diag.categorical_auc_within_bins(labels, [i.row.origin for i in items], bins)
    length = diag.length_auc(labels, words)
    return {
        "n": len(items),
        "auc_comprimento": length,
        "informatividade_comprimento": diag.informativeness(length),
        "auc_fonte_na_faixa": source_auc,
        "auc_fonte_por_faixa": source_per_bin,
        "auc_origem_na_faixa": origin_auc,
        "auc_origem_por_faixa": origin_per_bin,
        "mediana_palavras": {lab: float(np.median([i.words for i in items if i.label == lab] or [0]))
                             for lab in LABELS},
    }


def _exact_dedup(items: list[Item]) -> tuple[list[Item], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Dedup por classe (primeira ocorrência vence) e remoção de conflitos entre classes."""
    first: dict[tuple[str, str], Item] = {}
    sources_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    duplicates: list[dict[str, Any]] = []
    kinds = Counter()
    for item in items:
        key = (item.label, item.text)
        sources_of[key].add(item.row.source)
        kept = first.get(key)
        if kept is None:
            first[key] = item
            continue
        kind = "mesmo_arquivo" if kept.row.file == item.row.file else "entre_arquivos"
        kinds[f"{item.label}:{kind}"] += 1
        duplicates.append({"text": item.text, "label": item.label, "kept": kept.ref(),
                           "dropped": item.ref(), "source": item.row.source, "kind": kind})
    dd_texts = {text for (label, text) in first if label == "DD"}
    conflicts = sorted(dd_texts & {text for (label, text) in first if label == "NDD"})
    conflict_set = set(conflicts)
    conflict_rows = [{
        "text": text,
        "dd_sources": sorted(sources_of[("DD", text)]),
        "ndd_sources": sorted(sources_of[("NDD", text)]),
        "dd_raw": first[("DD", text)].row.text_raw,
        "ndd_raw": first[("NDD", text)].row.text_raw,
    } for text in conflicts]
    survivors = [item for key, item in first.items() if key[1] not in conflict_set]
    # Mantém a ordem canônica de leitura (dict preserva a ordem de inserção).
    return survivors, duplicates, conflict_rows, dict(kinds)


def _near_dedup(items: list[Item], cfg: CorpusConfig) -> tuple[list[Item], dict[str, Any], list[dict[str, Any]]]:
    """Remove conflitos aproximados (pares entre classes) e quase-duplicatas dentro da classe."""
    started = time.perf_counter()
    texts = [i.text for i in items]
    labels = np.array([i.label for i in items])
    matrix = tfidf_matrix(texts, (cfg.near_dedup.ngram_min, cfg.near_dedup.ngram_max))
    pairs = find_near_duplicates(matrix, cfg.near_dedup.threshold)
    cross = labels[pairs.left] != labels[pairs.right]
    conflict_ids = set(pairs.left[cross].tolist()) | set(pairs.right[cross].tolist())

    same = ~cross
    keep_pair = np.array([a not in conflict_ids and b not in conflict_ids
                          for a, b in zip(pairs.left[same].tolist(), pairs.right[same].tolist(), strict=True)],
                         dtype=bool)
    within = NearDuplicatePairs(pairs.left[same][keep_pair], pairs.right[same][keep_pair],
                                pairs.cosine[same][keep_pair])
    roots = connected_groups(len(items), within)

    artifacts: list[dict[str, Any]] = []
    for a, b, cos in zip(pairs.left[cross].tolist(), pairs.right[cross].tolist(),
                         pairs.cosine[cross].tolist(), strict=True):
        artifacts.append({"reason": "conflito_aproximado", "cosine": round(cos, 4),
                          "a": {"text": texts[a], "label": items[a].label, "source": items[a].row.source},
                          "b": {"text": texts[b], "label": items[b].label, "source": items[b].row.source}})
    survivors: list[Item] = []
    removed_near = Counter()
    for index, item in enumerate(items):
        if index in conflict_ids:
            continue
        root = int(roots[index])
        if root != index:
            removed_near[(item.label, item.bin)] += 1
            artifacts.append({"reason": "quase_duplicata", "text": item.text, "label": item.label,
                              "source": item.row.source, "kept_text": texts[root]})
            continue
        survivors.append(item)

    stats = {
        "pares_acima_do_limiar": int(len(pairs.left)),
        "pares_entre_classes": int(cross.sum()),
        "itens_em_conflito": len(conflict_ids),
        "quase_duplicatas_removidas": int(sum(removed_near.values())),
        "removidas_por_classe_faixa": {f"{lab}|{b}": n for (lab, b), n in sorted(removed_near.items())},
        "segundos": round(time.perf_counter() - started, 1),
    }
    return survivors, stats, artifacts


def _exclude_foreign_test(items: list[Item], cfg: CorpusConfig) -> tuple[list[Item], dict[str, Any]]:
    """Remove textos do teste de outro corpus e suas quase-duplicatas (cosseno > limiar)."""
    test_path = cfg.control.exclude_test_of / "test.jsonl"
    foreign = [row["text"] for row in read_jsonl(test_path)]
    foreign_set = set(foreign)
    exact = {i for i, item in enumerate(items) if item.text in foreign_set}
    texts = [item.text for item in items] + foreign
    matrix = tfidf_matrix(texts, (cfg.near_dedup.ngram_min, cfg.near_dedup.ngram_max))
    pairs = find_near_duplicates(matrix, cfg.near_dedup.threshold)
    n = len(items)
    near = {int(a) for a, b in zip(pairs.left, pairs.right, strict=True) if a < n <= b}
    near |= {int(b) for a, b in zip(pairs.left, pairs.right, strict=True) if b < n <= a}
    removed = exact | near
    stats = {"teste_excluido": str(test_path), "textos_no_teste": len(foreign_set), "iguais": len(exact),
             "quase_iguais": len(near - exact), "removidos": len(removed)}
    log.info("controle: %d textos do teste principal removidos (%d iguais, %d quase iguais)", len(removed),
             len(exact), len(near - exact))
    return [item for i, item in enumerate(items) if i not in removed], stats


def _versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "scikit-learn")
    out = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "ausente"
    return out


def build_corpus(cfg: CorpusConfig, output_dir: Path | None = None) -> dict[str, Any]:
    """Executa a construção completa e grava saídas em `output_dir` (padrão: `paths.build`)."""
    output_dir = output_dir or cfg.paths.build
    started = time.perf_counter()
    bin_order = [b.name for b in cfg.balance.length_bins]

    rows, sources_manifest, read_rejects = load_sources(cfg)
    if not rows:
        raise RuntimeError(f"nenhuma fonte em {cfg.paths.sources}; rode `ddsd corpus import-legacy`")
    log.info("fontes: %d linhas em %d arquivos", len(rows), len(sources_manifest))
    funnel = [{"etapa": "fontes", "DD": sum(r.label == "DD" for r in rows),
               "NDD": sum(r.label == "NDD" for r in rows), "total": len(rows),
               "nota": f"descartadas na leitura: {read_rejects}"}]

    items: list[Item] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        text = normalize_text(row.text_raw)
        reason = rejection_reason(text, cfg.filters, row.text_raw)
        words = word_count(text)
        length_bin = cfg.balance.bin_of(words) if reason is None else None
        if reason is None and length_bin is None:
            reason = "fora_das_faixas"
        if reason is not None:
            rejected.append({"reason": reason, "text_raw": row.text_raw, "normalized": text,
                             "label": row.label, "source": row.source, "ref": f"{row.file}:{row.line}"})
            continue
        items.append(Item(row=row, text=text, words=words, bin=length_bin.name))
    reasons = Counter(r["reason"] for r in rejected)
    funnel.append(_funnel_row("normalização e filtros", items, f"descartes: {dict(reasons)}"))
    diag_raw = _shortcut_diagnostics(items)

    items, duplicates, conflicts, dup_kinds = _exact_dedup(items)
    funnel.append(_funnel_row("dedup exato + conflitos", items,
                              f"duplicatas: {len(duplicates)}; conflitos de rótulo: {len(conflicts)} textos"))

    near_stats: dict[str, Any] = {"habilitado": cfg.near_dedup.enabled}
    near_artifacts: list[dict[str, Any]] = []
    if cfg.near_dedup.enabled:
        log.info("dedup aproximado (cosseno > %.2f) em %d textos...", cfg.near_dedup.threshold, len(items))
        items, stats, near_artifacts = _near_dedup(items, cfg)
        near_stats.update(stats)
        funnel.append(_funnel_row("dedup aproximado", items,
                                  f"{stats['quase_duplicatas_removidas']} quase-duplicatas; "
                                  f"{stats['itens_em_conflito']} itens em conflito entre classes"))
    control_stats: dict[str, Any] = {}
    if cfg.control.exclude_test_of is not None:
        items, control_stats = _exclude_foreign_test(items, cfg)
        funnel.append(_funnel_row("exclusão do teste do corpus principal", items,
                                  f"{control_stats['removidos']} textos (iguais ou quase iguais)"))
    diag_before_balance = _shortcut_diagnostics(items)
    hist_before = diag.length_histogram([i.label for i in items], [i.bin for i in items], bin_order)

    groups = ([f"{i.row.source}|{i.row.origin}" for i in items] if cfg.balance.mode == "faixa_fonte_origem"
              else [i.row.source for i in items])
    if cfg.balance.mode == "global":
        per_class = None
        if cfg.control.match_size_of is not None:
            reference = read_json(cfg.control.match_size_of)
            per_class = sum(reference["split"].values()) // 2
            control_stats["tamanho_casado_por_classe"] = per_class
        balance = balance_global([i.label for i in items], groups, per_class=per_class, seed=cfg.balance.seed)
    else:
        balance = balance_by_length(
            [i.label for i in items], groups, [i.words for i in items],
            cfg.balance.length_bins, mode=cfg.balance.mode, seed=cfg.balance.seed,
        )
    discarded_balance = [items[i] for i in balance.discarded]
    items = [items[i] for i in balance.kept]
    funnel.append(_funnel_row(f"balanceamento ({cfg.balance.mode})", items,
                              f"descartados: {len(discarded_balance)}"))
    diag_final = _shortcut_diagnostics(items)
    hist_after = diag.length_histogram([i.label for i in items], [i.bin for i in items], bin_order)

    keys = stratum_keys([i.label for i in items], [i.row.source for i in items], [i.bin for i in items],
                        cfg.split.min_stratum)
    parts = stratified_split(keys, cfg.split.ratios, cfg.split.seed)
    verify_disjoint([i.text for i in items], parts)

    # ---- asserções do corpus final ----
    texts = [i.text for i in items]
    assert len(set(texts)) == len(texts), "texto repetido no corpus final"
    if cfg.balance.mode != "global":
        for b, dd, ndd in hist_after:
            assert dd == ndd, f"faixa {b} desbalanceada: DD={dd} NDD={ndd}"
    counts = Counter(i.label for i in items)
    assert counts["DD"] == counts["NDD"], f"classes desbalanceadas: {counts}"
    source_share = {
        name: Counter(items[i].row.source for i in idx) for name, idx in parts.items()
    }
    total_share = Counter(i.row.source for i in items)
    max_dev_pp = max(
        abs(100 * source_share[name][s] / max(len(parts[name]), 1) - 100 * total_share[s] / len(items))
        for name in SPLIT_NAMES for s in total_share
    )
    split_diag = {name: _shortcut_diagnostics([items[i] for i in idx]) for name, idx in parts.items()}

    # ---- gravação ----
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "splits").mkdir(parents=True)
    split_files: dict[str, dict[str, Any]] = {}
    for name, idx in parts.items():
        path = output_dir / "splits" / f"{name}.jsonl"
        write_jsonl(path, (items[i].to_split_row() for i in idx))
        split_files[name] = {"rows": len(idx), "sha256": file_sha256(path)}
    art = output_dir / "artifacts"
    write_jsonl(art / "rejected.jsonl", rejected)
    write_jsonl(art / "exact_duplicates.jsonl", duplicates)
    write_jsonl(art / "conflicts_exact.jsonl", conflicts)
    write_jsonl(art / "near_duplicates.jsonl", near_artifacts)
    write_jsonl(art / "balance_discarded.jsonl", (i.to_split_row() for i in discarded_balance))

    summary: dict[str, Any] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.perf_counter() - started, 1),
        "config_file": str(cfg.source_file),
        "config_sha256": file_sha256(cfg.source_file),
        "versions": _versions(),
        "codigo": code_version(),
        "funnel": funnel,
        "rejections": dict(reasons),
        "exact_duplicates": dup_kinds,
        "exact_conflicts": len(conflicts),
        "near_dedup": near_stats,
        "include_origins": list(cfg.sources.include_origins),
        "control": control_stats,
        "length_histogram_before_balance": hist_before,
        "length_histogram_final": hist_after,
        "balance_mode": cfg.balance.mode,
        "balance_bins": [{"faixa": b.name, "antes": b.before, "por_classe": b.kept_per_class,
                          "quotas_por_fonte": b.source_quotas} for b in balance.bins],
        "diagnostics": {"apos_filtros": diag_raw, "antes_balanceamento": diag_before_balance,
                        "final": diag_final, "por_particao": split_diag},
        "split": {name: len(idx) for name, idx in parts.items()},
        "split_source_max_deviation_pp": round(max_dev_pp, 4),
        "final_by_source_label": {f"{s}|{lab}": n for (s, lab), n in
                                  sorted(Counter((i.row.source, i.label) for i in items).items())},
        "final_by_origin_label": {f"{o}|{lab}": n for (o, lab), n in
                                  sorted(Counter((i.row.origin, i.label) for i in items).items())},
        "split_files": split_files,
        "sources": sources_manifest,
    }
    write_json(output_dir / "manifest.json", summary)
    _write_report(output_dir / "report.md", summary, bin_order)
    log.info("corpus construído em %.1f s: %s", summary["seconds"], summary["split"])
    return summary


def _write_report(path: Path, s: dict[str, Any], bin_order: list[str]) -> None:
    r = MarkdownReport("Relatório de construção do corpus DDSD")
    r.paragraph(f"Gerado em {s['built_at']} em {s['seconds']} s. Configuração: `{s['config_file']}` "
                f"(SHA-256 `{s['config_sha256'][:12]}`).")
    r.heading("Funil")
    r.table(["etapa", "DD", "NDD", "total", "nota"],
            [[f["etapa"], f["DD"], f["NDD"], f["total"], f["nota"]] for f in s["funnel"]])
    r.heading("Atalhos: o quanto cada atributo prediz a classe")
    r.paragraph("AUC com DD positivo; 0,5 = sem informação. Para o comprimento, o score é -palavras "
                "(AUC > 0,5 indica que frases curtas tendem a DD). Fonte e origem são medidas dentro "
                "de cada faixa de comprimento.")
    d = s["diagnostics"]
    rows = []
    for label, key in (("após filtros", "apos_filtros"), ("antes do balanceamento", "antes_balanceamento"),
                       ("final", "final")):
        x = d[key]
        rows.append([label, x["n"], x["auc_comprimento"], x["auc_fonte_na_faixa"], x["auc_origem_na_faixa"],
                     f"{x['mediana_palavras']['DD']:.0f} / {x['mediana_palavras']['NDD']:.0f}"])
    for name, x in d["por_particao"].items():
        rows.append([f"final - {name}", x["n"], x["auc_comprimento"], x["auc_fonte_na_faixa"],
                     x["auc_origem_na_faixa"],
                     f"{x['mediana_palavras']['DD']:.0f} / {x['mediana_palavras']['NDD']:.0f}"])
    r.table(["conjunto", "n", "AUC comprimento", "AUC fonte|faixa", "AUC origem|faixa",
             "mediana palavras DD/NDD"], rows)
    r.heading("Histograma de comprimento (palavras)")
    before = {b: (dd, ndd) for b, dd, ndd in s["length_histogram_before_balance"]}
    after = {b: (dd, ndd) for b, dd, ndd in s["length_histogram_final"]}
    r.table(["faixa", "DD antes", "NDD antes", "% DD antes", "DD no final", "NDD no final"],
            [[b, before[b][0], before[b][1],
              f"{100 * before[b][0] / max(sum(before[b]), 1):.1f}%", after[b][0], after[b][1]] for b in bin_order])
    r.heading("Dedup aproximado")
    r.code("\n".join(f"{k}: {v}" for k, v in s["near_dedup"].items() if k != "removidas_por_classe_faixa"))
    r.heading("Composição final por origem")
    r.table(["origem|classe", "n"], list(s["final_by_origin_label"].items()))
    r.heading("Composição final por fonte")
    r.table(["fonte|classe", "n"], list(s["final_by_source_label"].items()))
    r.heading("Partições")
    r.table(["partição", "n", "SHA-256"],
            [[name, info["rows"], info["sha256"][:16]] for name, info in s["split_files"].items()])
    r.paragraph(f"Desvio máximo da participação de cada fonte entre partições: "
                f"{s['split_source_max_deviation_pp']:.3f} p.p.")
    r.save(path)

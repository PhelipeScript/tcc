"""Composição do corpus final congelado (`uv run python scripts/relatorio_corpus.py`).

Lê as partições congeladas em `data/corpus/frozen/splits/` (fora do git) e confere o SHA-256 de cada uma com o
manifesto versionado (`reports/corpus/principal/manifest.json`). Grava `reports/corpus/principal/composicao.json`
e `composicao.md`: frases por origem, modelo gerador e estratégia, por classe e por partição, e o tamanho das
frases de cada classe.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "corpus" / "frozen" / "splits"
OUT = ROOT / "reports" / "corpus" / "principal"
PARTS = {"train": "Treino", "val": "Validação", "test": "Teste"}
ORIGINS = {"legacy": "Corpus inicial", "mitigation": "Frases novas da correção"}
STRATEGIES = {"legado": "-", "comprimento": "geração por tamanho", "negativos_dificeis": "negativos difíceis"}
LABELS = ("DD", "NDD")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def decimal(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float) -> str:
    return "< 0,1%" if 0 < value < 0.0005 else decimal(100 * value, 1) + "%"


def load_rows(manifest: dict) -> list[dict]:
    """Frases das 3 partições, depois de conferir que os arquivos são os do manifesto."""
    rows = []
    for part in PARTS:
        path = SPLITS / f"{part}.jsonl"
        expected = manifest["split_files"][part]["sha256"]
        found = sha256(path)
        if found != expected:
            raise SystemExit(f"{path}: SHA-256 {found[:16]} diferente do manifesto ({expected[:16]})")
        with path.open(encoding="utf-8") as handle:
            rows += [{**json.loads(line), "part": part} for line in handle]
        if sum(r["part"] == part for r in rows) != manifest["split_files"][part]["rows"]:
            raise SystemExit(f"{path}: número de frases diferente do manifesto")
    return rows


def build(rows: list[dict], manifest: dict) -> dict:
    groups = Counter((r["origin"], r["model"], r["strategy"], r["label"]) for r in rows)
    keys = sorted({k[:3] for k in groups}, key=lambda k: (list(ORIGINS).index(k[0]),
                                                         -sum(groups[(*k, lab)] for lab in LABELS), k[1], k[2]))
    total = len(rows)
    by_generator = [{"origem": origin, "modelo": model, "estrategia": strategy,
                     **{lab: groups[(origin, model, strategy, lab)] for lab in LABELS},
                     "fracao_do_corpus": sum(groups[(origin, model, strategy, lab)] for lab in LABELS) / total}
                    for origin, model, strategy in keys]
    by_origin = {origin: {lab: sum(g[lab] for g in by_generator if g["origem"] == origin) for lab in LABELS}
                 for origin in ORIGINS}
    expected_origin = manifest["final_by_origin_label"]
    for origin, counts in by_origin.items():
        for lab in LABELS:
            if counts[lab] != expected_origin[f"{origin}|{lab}"]:
                raise SystemExit(f"{origin}|{lab}: {counts[lab]} frases, o manifesto diz {expected_origin[f'{origin}|{lab}']}")
    parts = {part: {lab: sum(r["part"] == part and r["label"] == lab for r in rows) for lab in LABELS}
             for part in PARTS}
    words = {lab: np.array([r["words"] for r in rows if r["label"] == lab]) for lab in LABELS}
    length = {lab: {"mediana": float(np.median(w)), "p10": float(np.percentile(w, 10)),
                    "p90": float(np.percentile(w, 90)), "minimo": int(w.min()), "maximo": int(w.max())}
              for lab, w in words.items()}
    new = [g for g in by_generator if g["origem"] == "mitigation"]
    new_total = sum(g["DD"] + g["NDD"] for g in new)
    new_by_model = Counter()
    for g in new:
        new_by_model[g["modelo"]] += g["DD"] + g["NDD"]
    return {
        "particoes_sha256": {part: manifest["split_files"][part]["sha256"] for part in PARTS},
        "total": total,
        "por_origem": by_origin,
        "por_gerador": by_generator,
        "frases_novas_por_modelo": {m: {"frases": n, "fracao": n / new_total} for m, n in new_by_model.most_common()},
        "particoes": parts,
        "palavras": length,
        "auc_comprimento": {"depois_dos_filtros": manifest["diagnostics"]["apos_filtros"]["auc_comprimento"],
                            "corpus_final": manifest["diagnostics"]["final"]["auc_comprimento"]},
    }


def markdown(c: dict) -> str:
    lines = ["# Composição do corpus final", "",
             "Gerado por `scripts/relatorio_corpus.py` a partir "
             "das partições congeladas (SHA-256 conferido com `manifest.json`). Texto normalizado; tamanho em "
             "palavras.", "",
             "## Por origem, modelo gerador e estratégia", "",
             "| Origem | Modelo gerador | Estratégia | DD | NDD | Parte do corpus |", "|---|---|---|---|---|---|"]
    for g in c["por_gerador"]:
        lines.append(f"| {ORIGINS[g['origem']]} | {g['modelo']} | {STRATEGIES[g['estrategia']]} | "
                     f"{thousands(g['DD'])} | {thousands(g['NDD'])} | {pct(g['fracao_do_corpus'])} |")
    for origin, counts in c["por_origem"].items():
        n = counts["DD"] + counts["NDD"]
        lines.append(f"| **{ORIGINS[origin]}** | | | **{thousands(counts['DD'])}** | **{thousands(counts['NDD'])}** | "
                     f"**{pct(n / c['total'])}** |")
    lines += ["", "Nas frases novas: " + "; ".join(f"{m}, {pct(v['fracao'])}" for m, v in
                                                     c["frases_novas_por_modelo"].items()) + ".", "",
              "## Partições", "", "| Partição | DD | NDD | Total |", "|---|---|---|---|"]
    for part, counts in c["particoes"].items():
        lines.append(f"| {PARTS[part]} | {thousands(counts['DD'])} | {thousands(counts['NDD'])} | "
                     f"{thousands(counts['DD'] + counts['NDD'])} |")
    lines += ["", "## Tamanho das frases (palavras)", "", "| Classe | Mediana | 10% até | 90% até | Mínimo | Máximo |",
              "|---|---|---|---|---|---|"]
    for lab, w in c["palavras"].items():
        lines.append(f"| {lab} | {w['mediana']:.0f} | {w['p10']:.0f} | {w['p90']:.0f} | {w['minimo']} | {w['maximo']} |")
    auc = c["auc_comprimento"]
    lines += ["", f"AUC com que só o tamanho prevê a classe: {decimal(auc['depois_dos_filtros'], 3)} depois dos "
              f"filtros e {decimal(auc['corpus_final'], 3)} no corpus final.", ""]
    return "\n".join(lines)


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    composition = build(load_rows(manifest), manifest)
    (OUT / "composicao.json").write_text(json.dumps(composition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "composicao.md").write_text(markdown(composition), encoding="utf-8")
    print(f"composição em {OUT / 'composicao.md'}")


if __name__ == "__main__":
    main()

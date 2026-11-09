"""Revisão humana cega do corpus e concordância (kappa de Cohen) com os rótulos dos LLMs.

`ddsd corpus annotate`: sorteia uma amostra estratificada (fonte × classe × faixa de
comprimento) do corpus congelado e mostra cada frase sem rótulo, sem fonte, em
ordem aleatória. O anotador marca DD ou NDD, pode voltar e corrigir, e o
progresso fica salvo a cada resposta (pode parar e continuar depois).

`ddsd corpus agreement`: calcula o kappa de Cohen (geral, por origem, por fonte e por
faixa), a matriz de concordância e a lista de divergências, em
`reports/corpus/anotacao_<anotador>.{md,json}`.
"""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ddsd.common.io import append_jsonl, read_jsonl, write_json, write_jsonl
from ddsd.common.paths import DATA_DIR, REPORTS_DIR
from ddsd.common.report import MarkdownReport

ANNOTATION_DIR = DATA_DIR / "annotation"


def stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Amostra com cotas iguais por estrato (fonte|classe|faixa), completando com sorteio global."""
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[f"{row['source']}|{row['label']}|{row['bin']}"].append(row)
    rng = random.Random(seed)
    per_stratum = max(1, n // len(strata))
    chosen: list[dict[str, Any]] = []
    for key in sorted(strata):
        group = strata[key]
        chosen.extend(rng.sample(group, min(per_stratum, len(group))))
    taken = {id(r) for r in chosen}
    remaining = [r for r in rows if id(r) not in taken]
    if len(chosen) < n:
        chosen.extend(rng.sample(remaining, min(n - len(chosen), len(remaining))))
    rng.shuffle(chosen)
    return chosen[:n]


def build_sample(frozen_splits: Path, n: int, seed: int) -> Path:
    """Grava `data/annotation/amostra.jsonl` (uma vez; a amostra é fixa para todos os anotadores)."""
    path = ANNOTATION_DIR / "amostra.jsonl"
    if path.exists():
        return path
    rows = [row for split in ("train", "val", "test") for row in read_jsonl(frozen_splits / f"{split}.jsonl")]
    sample = stratified_sample(rows, n, seed)
    write_jsonl(path, ({"id": f"A{i:04d}", **{k: r[k] for k in ("text_raw", "text", "label", "source", "origin",
                                                                  "strategy", "bin")}}
                       for i, r in enumerate(sample)))
    return path


def _answers(annotator: str) -> dict[str, dict[str, Any]]:
    path = ANNOTATION_DIR / f"respostas_{annotator}.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in read_jsonl(path):
            latest[row["id"]] = row  # a última resposta vale (permite corrigir)
    return latest


def annotate(annotator: str, frozen_splits: Path, n: int = 300, seed: int = 20261101) -> int:
    """Sessão interativa de anotação (exige um terminal de verdade)."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    sample = list(read_jsonl(build_sample(frozen_splits, n, seed)))
    answers_path = ANNOTATION_DIR / f"respostas_{annotator}.jsonl"
    answers = _answers(annotator)
    index = next((i for i, row in enumerate(sample) if row["id"] not in answers), len(sample))
    console.print(Panel(
        "Para cada frase, imagine-a dita em voz alta perto de um computador com assistente de voz sempre "
        "ouvindo.\n[bold]d[/bold] = DD (pedido ao computador) · [bold]n[/bold] = NDD (não é para o "
        "computador) · [bold]v[/bold] = voltar · [bold]s[/bold] = sair (o progresso fica salvo)",
        title=f"Anotação cega - {annotator}"))
    while index < len(sample):
        row = sample[index]
        previous = answers.get(row["id"], {}).get("answer")
        console.rule(f"{index + 1}/{len(sample)}" + (f" (resposta anterior: {previous})" if previous else ""))
        console.print(f"\n   [bold]{row['text_raw']}[/bold]\n")
        choice = Prompt.ask("DD ou NDD?", choices=["d", "n", "v", "s"], show_choices=True).lower()
        if choice == "s":
            break
        if choice == "v":
            index = max(0, index - 1)
            continue
        answer = "DD" if choice == "d" else "NDD"
        record = {"id": row["id"], "answer": answer, "annotator": annotator,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        append_jsonl(answers_path, [record])
        answers[row["id"]] = record
        index += 1
    done = sum(1 for row in sample if row["id"] in answers)
    console.print(f"\n{done}/{len(sample)} anotadas. Rode `uv run ddsd corpus agreement --annotator {annotator}`.")
    return 0


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Kappa de Cohen para dois avaliadores e rótulos categóricos."""
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def agreement_report(annotator: str) -> Path:
    """Relatório de concordância do anotador com os rótulos do corpus (gerados pelos LLMs)."""
    sample = {row["id"]: row for row in read_jsonl(ANNOTATION_DIR / "amostra.jsonl")}
    answers = _answers(annotator)
    rows = [(sample[i], a["answer"]) for i, a in answers.items() if i in sample]
    pairs = [(row["label"], answer) for row, answer in rows]

    def by(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row, answer in rows:
            groups[str(row[key])].append((row["label"], answer))
        return {g: {"n": len(p), "concordancia": sum(a == b for a, b in p) / len(p), "kappa": cohen_kappa(p)}
                for g, p in sorted(groups.items())}

    matrix = Counter(pairs)
    divergences = [{"id": row["id"], "texto": row["text_raw"], "rotulo_llm": row["label"], "humano": answer,
                    "fonte": row["source"], "origem": row["origin"], "faixa": row["bin"]}
                   for row, answer in rows if row["label"] != answer]
    summary = {
        "anotador": annotator, "anotadas": len(rows), "amostra": len(sample),
        "concordancia": sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan"),
        "kappa": cohen_kappa(pairs),
        "matriz": {f"{llm}->{hum}": matrix[(llm, hum)] for llm in ("DD", "NDD") for hum in ("DD", "NDD")},
        "por_origem": by("origin"), "por_fonte": by("source"), "por_estrategia": by("strategy"),
        "por_faixa": by("bin"), "divergencias": divergences,
    }
    out = REPORTS_DIR / "corpus"
    write_json(out / f"anotacao_{annotator}.json", summary)
    report = MarkdownReport(f"Revisão humana cega - {annotator}")
    report.paragraph(f"{len(rows)} de {len(sample)} frases anotadas. Concordância {summary['concordancia']:.3f}; "
                     f"kappa de Cohen {summary['kappa']:.3f}.")
    report.table(["LLM \\ humano", "DD", "NDD"],
                 [[llm, matrix[(llm, "DD")], matrix[(llm, "NDD")]] for llm in ("DD", "NDD")])
    for title, key in (("Por origem", "por_origem"), ("Por fonte", "por_fonte"), ("Por estratégia", "por_estrategia"),
                       ("Por faixa", "por_faixa")):
        report.heading(title, 3)
        report.table(["grupo", "n", "concordância", "kappa"],
                     [[g, v["n"], v["concordancia"], v["kappa"]] for g, v in summary[key].items()])
    report.heading("Divergências", 3)
    report.table(["id", "texto", "LLM", "humano", "fonte", "faixa"],
                 [[d["id"], d["texto"], d["rotulo_llm"], d["humano"], d["fonte"], d["faixa"]] for d in divergences])
    return report.save(out / f"anotacao_{annotator}.md")

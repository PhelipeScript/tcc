"""Figuras do TCC, geradas a partir dos relatórios em `reports/` (`uv run python scripts/figuras.py`).

Saída: `reports/figuras/<nome>.png` (200 dpi) e `<nome>.pdf` (vetorial, para o LaTeX).

Regras de todas as figuras:
- cada figura responde a uma pergunta. Título, notas e fonte ficam na legenda do texto, não na imagem;
- 16 cm de largura e fonte de 8 pt ou mais;
- cada condição tem cor e marcador fixos, então a figura também se lê em tons de cinza:
  - com correção do viés de comprimento: azul, círculo;
  - sem correção: laranja, quadrado;
  - TF-IDF: verde, triângulo;
- na M4 (Whisper), as opções usam tons de cinza, marcador e traço próprios, com o nome escrito ao lado;
- eixos em porcentagem começam do zero, e os decimais usam vírgula;
- valores escritos só nos pontos que o texto cita.

Uma figura cujos dados ainda não existem é pulada com aviso.
"""

from __future__ import annotations

import json
import locale
import logging
import math
import os
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

# PDFs idênticos byte a byte a cada execução: data fixa (criação e fontes embutidas) e ordem fixa dos glifos
# das fontes embutidas, que depende do hash das strings. O script se reinicia uma vez com os dois fixos.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ.update(PYTHONHASHSEED="0", SOURCE_DATE_EPOCH="0")
    os.execv(sys.executable, [sys.executable, *sys.argv])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # avisos inofensivos ao embutir a Helvetica nos PDFs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "figuras"

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COM, SEM, TFIDF = "#2a78d6", "#eb6834", "#1baf7a"  # paleta categórica de referência, posições 1-3 (validada)
STYLE = {"com": (COM, "o"), "sem": (SEM, "s"), "tfidf": (TFIDF, "^")}
WIDTH = 16 / 2.54  # 16 cm em polegadas

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.formatter.use_locale": True,
})
for _name in ("pt_BR.UTF-8", "pt_BR.utf8", "pt_BR"):  # vírgula decimal nos eixos, como no texto do TCC
    try:
        locale.setlocale(locale.LC_NUMERIC, _name)
        break
    except locale.Error:
        continue

TRANSFORMERS = {"bertimbau": "BERTimbau", "albertina": "Albertina", "debertinha": "DeBERTinha"}
TFIDF_NAMES = {"baseline_logreg": "TF-IDF + Regressão Logística", "baseline_svm": "TF-IDF + SVM"}
PIPELINE_CFG = tomllib.loads((ROOT / "config" / "pipeline.toml").read_text(encoding="utf-8"))
SCENARIOS = {"quieto_interno": "Sala quieta", "ruido_interno": "Ruído de fundo", "quieto_iphone": "iPhone",
             "distante_interno": "~2 m", "conversa_interno": "Roteiro de conversa"}


# ------------------------------------------------------------------------------------------ utilitários
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_label(model: str) -> str:
    """Execução do Transformer no sistema (ex.: `bertimbau-s44`), a marcada em `reports/treino/treino.json`."""
    runs = load_json(REPORTS / "treino" / "treino.json")["transformers"]
    return next(r["execucao"] for r in runs if r["modelo"] == model and r["no_sistema"])


def num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct_text(value: float, digits: int = 1) -> str:
    return f"{num(value, digits)}%"


def percent_axis(ax, axis: str = "y") -> None:
    """Marcas em porcentagem, com uma casa decimal só quando a marca não é inteira (2,5% não vira 2%)."""
    def fmt(v: float, _p) -> str:
        return f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else pct_text(v)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def needs(*paths: Path) -> bool:
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"  (pulada: falta {p.relative_to(ROOT)})")
    return not missing


def dot(ax, x, y, style: str, label: str | None = None, size: float = 6.5, **kw) -> None:
    color, marker = STYLE[style]
    ax.plot(x, y, linestyle="none", marker=marker, color=color, markersize=size, markeredgecolor="white",
            markeredgewidth=1.0, label=label, zorder=3, **kw)


def latest_benchmark(part: str) -> Path | None:
    """O benchmark mais recente que mediu `part` (o nome dos arquivos traz data e hora)."""
    found = [p for p in sorted((REPORTS / "benchmark").glob("benchmark-*.json")) if part in load_json(p)]
    return found[-1] if found else None


# --------------------------------------------------------------------- M1 · o sistema em tempo real
def m1_sistema() -> None:
    """Caminho de uma fala, do microfone à decisão. Os tempos medidos ficam na R7; aqui só o parâmetro do VAD."""
    wait = PIPELINE_CFG["vad"]["min_silence_ms"] / 1000
    boxes = [
        ("Microfone", "áudio contínuo"),
        ("Detector de voz", f"fecha a fala após\n{num(wait)} s de silêncio"),
        ("Whisper", "transcreve a\nfala completa"),
        ("Classificador", "um dos 5 modelos\nde texto"),
        ("Decisão", "DD ou NDD"),
    ]
    fig = plt.figure(figsize=(WIDTH, 2.0))
    ax = fig.add_axes((0, 0, 1, 1))  # o diagrama ocupa os 16 cm, sem as margens de um gráfico
    ax.set_xlim(0, len(boxes))
    ax.set_ylim(-0.04, 1)
    ax.axis("off")
    w, h, y0 = 0.84, 0.50, 0.46

    def box(cx: float, cy: float, width: float, height: float, title: str, detail: str, dashed: bool = False):
        ax.add_patch(FancyBboxPatch((cx - width / 2, cy - height / 2), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=0.9, edgecolor=INK_2,
                                    facecolor="white" if dashed else "#f4f3f0", linestyle="--" if dashed else "-"))
        ax.text(cx, cy + height / 2 - 0.12, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(cx, cy - 0.07, detail, ha="center", va="center", fontsize=8, color=INK_2, linespacing=1.2)

    for i, (title, detail) in enumerate(boxes):
        box(i + 0.5, y0 + h / 2, w, h, title, detail)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 1 + (1 - w) / 2, y0 + h / 2), xytext=(i + 0.5 + w / 2, y0 + h / 2),
                        arrowprops={"arrowstyle": "-|>", "color": INK_2, "linewidth": 0.9})
    # O índice de confiança sai dos metadados do Whisper e é registrado com a decisão, sem mudá-la (tracejado).
    fy, fh = 0.17, 0.30
    box(3.5, fy, 1.5, fh, "Índice de confiança", "lógica fuzzy", dashed=True)
    dashed = {"arrowstyle": "-|>", "color": INK_2, "linewidth": 0.9, "linestyle": "--"}
    ax.annotate("", xy=(2.75, fy), xytext=(2.5, y0), arrowprops={**dashed, "connectionstyle": "angle,angleA=-90,angleB=180"})
    ax.annotate("", xy=(4.5, y0), xytext=(4.25, fy), arrowprops={**dashed, "connectionstyle": "angle,angleA=0,angleB=-90"})
    save(fig, "m1_sistema")


# --------------------------------------------------------------------- M2 · funil do corpus
def m2_funil_corpus() -> None:
    manifest = load_json(REPORTS / "corpus" / "principal" / "manifest.json")
    names = ["Frases das fontes", "Depois da normalização e dos filtros", "Sem duplicatas nem conflitos de rótulo",
             "Sem quase-duplicatas", "Balanceado por faixa de tamanho (corpus final)"]
    totals = [stage["total"] for stage in manifest["funnel"]]
    assert len(totals) == len(names), "o funil do relatório mudou de etapas"
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    y = np.arange(len(names))[::-1]
    ax.barh(y, totals, height=0.58, color=[INK_2] * (len(names) - 1) + [COM])
    for yy, total in zip(y, totals, strict=True):
        ax.text(total + 3000, yy, f"{total:,}".replace(",", "."), va="center", fontsize=8.5)
    ax.set_yticks(y, names)
    ax.set_xlim(0, max(totals) * 1.18)
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v / 1000:.0f} mil" if v else "0")
    ax.set_xlabel("Frases")
    ax.grid(axis="y", visible=False)
    save(fig, "m2_funil_corpus")  # a divisão treino/validação/teste fica na legenda


# --------------------------------------------------------------------- M3 · o treino
def m3_treino() -> None:
    """F1 macro e perda de validação ao fim de cada época, nas 9 execuções com correção (uma linha por semente).

    Os modelos sem correção ficam de fora: a validação deles é a do corpus com o atalho do tamanho.
    """
    path = REPORTS / "treino" / "treino.json"
    if not needs(path):
        return
    runs = [r for r in load_json(path)["transformers"] if r["condicao"] == "principal"]
    fig, axes = plt.subplots(2, len(TRANSFORMERS), figsize=(WIDTH, 3.9), sharex="col", sharey="row")
    rows = (("f1_macro_validacao", "F1 macro de validação"), ("perda_validacao", "Perda de validação"))
    for col, (family, name) in enumerate(TRANSFORMERS.items()):
        family_runs = [r for r in runs if r["modelo"] == family]
        for run in family_runs:
            epochs = [e["epoca"] for e in run["por_epoca"]]
            best = epochs.index(round(run["melhor_checkpoint"]["epoca"]))
            chosen = run["execucao"] == run_label(family)  # a semente usada no sistema: linha cheia e mais grossa
            line = {"linewidth": 1.6, "linestyle": "-"} if chosen else {"linewidth": 1.0, "linestyle": (0, (3, 2))}
            for row, (key, _label) in enumerate(rows):
                values = [e[key] for e in run["por_epoca"]]
                axes[row, col].plot(epochs, values, color=COM, marker="o", markersize=3.8, markerfacecolor="white",
                                    markeredgecolor=COM, markeredgewidth=1.0, zorder=2 + chosen, **line)
                axes[row, col].plot([epochs[best]], [values[best]], linestyle="none", marker="o", markersize=6.5,
                                    color=COM, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        axes[0, col].set_title(name, fontsize=8.5, loc="left")
        last = max(len(r["por_epoca"]) for r in family_runs)
        axes[1, col].set_xticks(range(1, last + 1))
        axes[1, col].set_xlim(0.6, last + 0.4)
    for row, (_key, label) in enumerate(rows):
        axes[row, 0].set_ylabel(label)
    axes[1, 1].set_xlabel("Época")
    handles = [Line2D([], [], color=COM, linewidth=1.6, marker="o", markersize=3.8, markerfacecolor="white",
                      markeredgecolor=COM, label="semente usada no sistema"),
               Line2D([], [], color=COM, linewidth=1.0, linestyle=(0, (3, 2)), marker="o", markersize=3.8,
                      markerfacecolor="white", markeredgecolor=COM, label="outras sementes"),
               Line2D([], [], linestyle="none", marker="o", markersize=6.5, color=COM, markeredgecolor="white",
                      label="melhor checkpoint (maior F1 de validação)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
               columnspacing=1.6)
    fig.tight_layout(h_pad=1.0, w_pad=1.2, rect=(0, 0, 1, 0.93))
    save(fig, "m3_treino")


# --------------------------------------------------------------------- M4 · Whisper na GPU × na CPU
WHISPER_SETUPS = {"mlx": {"biblioteca": "mlx-whisper", "dispositivo": "GPU", "precisao": "fp16", "threads": None},
                  "faster-whisper-fp32": {"biblioteca": "faster-whisper", "dispositivo": "CPU", "precisao": "fp32",
                                          "threads": 4},
                  "faster-whisper-int8": {"biblioteca": "faster-whisper", "dispositivo": "CPU", "precisao": "int8",
                                          "threads": 4}}


def m4_whisper() -> None:
    """Tempo para transcrever uma fala com o Whisper large-v3-turbo, por duração da fala (mediana).

    O mlx roda na GPU; o faster-whisper, só na CPU no macOS. A diagonal é o tempo real: acima dela, transcrever
    leva mais tempo que a própria fala.
    """
    path = latest_benchmark("whisper")
    if path is None:
        print("  (pulada: falta reports/benchmark/benchmark-*.json com o Whisper)")
        return
    bench = load_json(path)["whisper"]
    styles = {"mlx": (INK, "o", "-", 2.0), "faster-whisper-fp32": (INK_2, "s", "--", 1.4),
              "faster-whisper-int8": ("#8f8d88", "^", ":", 1.6)}  # cor, marcador, traço, espessura
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    top = 1.15 * max(v["p50_s"] for b in bench.values() for v in b["por_duracao"].values())
    ax.plot([0, top], [0, top], color=INK_2, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.text(top - 1.2, top - 0.35, "tempo igual à\nduração da fala", ha="right", va="center", fontsize=8,
            color=INK_2, linespacing=1.1)
    for key in ("faster-whisper-int8", "faster-whisper-fp32", "mlx"):
        b, (color, marker, dash, width) = bench[key], styles[key]
        setup = {**WHISPER_SETUPS[key], **{k: b[k] for k in WHISPER_SETUPS[key] if k in b}}
        xs = [float(d.removesuffix("s")) for d in b["por_duracao"]]
        ys = [v["p50_s"] for v in b["por_duracao"].values()]
        ax.plot(xs, ys, color=color, linestyle=dash, linewidth=width, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        where = "GPU" if setup["dispositivo"] == "GPU" else f"CPU, {setup['threads']} threads"
        ax.text(14.5, max(y for x, y in zip(xs, ys, strict=True) if x >= 10) + 0.15, f"{setup['biblioteca']} {setup['precisao']} · {where}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks([1, 3, 5, 10, 20])
    ax.set_xlim(0, 21)
    ax.set_ylim(0, top)
    ax.set_xlabel("Duração da fala (s)")
    ax.set_ylabel("Tempo para transcrever (s)")
    save(fig, "m4_whisper")


# --------------------------------------------------------------------- R1 · viés de comprimento no corpus
def r1_vies_comprimento() -> None:
    principal = load_json(REPORTS / "corpus" / "principal" / "manifest.json")
    control = load_json(REPORTS / "corpus" / "controle" / "manifest.json")

    def share(hist):
        return [100 * dd / (dd + ndd) if dd + ndd else np.nan for _, dd, ndd in hist]

    bins = [b for b, _, _ in principal["length_histogram_final"]]
    x = np.arange(len(bins))
    sem, com = share(control["length_histogram_final"]), share(principal["length_histogram_final"])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.axhline(50, color=INK_2, linewidth=0.8, zorder=1)
    for values, style, label in ((sem, "sem", "Sem correção (corpus inicial)"),
                                 (com, "com", "Com correção (corpus final)")):
        color, marker = STYLE[style]
        ax.plot(x, values, color=color, linewidth=2, marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=1.0, label=label, zorder=3)
    for i, offset, ha in ((1, (0, 8), "center"), (9, (6, 5), "left")):  # citados no texto: 3 e 18-24 palavras
        ax.annotate(pct_text(sem[i], 1), (x[i], sem[i]), xytext=offset, textcoords="offset points", ha=ha,
                    fontsize=8)
    ax.text(x[8], 53, "50% em todas as faixas", ha="center", va="bottom", fontsize=8)  # longe da linha laranja
    ax.set_xticks(x, bins)
    ax.set_xlabel("Tamanho da frase (palavras)")
    ax.set_ylabel("Frases dirigidas ao computador")
    ax.set_ylim(0, 105)
    percent_axis(ax)
    ax.legend(loc="lower left")
    save(fig, "r1_vies_comprimento")


# --------------------------------------------------------------------- R2 · teste interno
def r2_teste_interno() -> None:
    t = load_json(REPORTS / "experimento" / "teste_interno.json")
    rows = []  # (rótulo, estilo, erro %, desvio %, EER %, desvio %)
    for key, name in TRANSFORMERS.items():
        p = t["principal"][key]
        rows.append((name, "com", 100 * (1 - p["accuracy"]["media"]), 100 * p["accuracy"]["desvio"],
                     100 * p["eer"]["media"], 100 * p["eer"]["desvio"]))
    for key, name in TFIDF_NAMES.items():
        b = t["baselines"][key]
        rows.append((name, "tfidf", 100 * (1 - b["accuracy"]), 0.0, 100 * b["eer"], 0.0))
    for key, name in TRANSFORMERS.items():
        c = t["controle_no_teste_principal"][key]["metricas"]
        rows.append((f"{name} sem correção", "sem", 100 * (1 - c["accuracy"]), 0.0, 100 * c["eer"], 0.0))
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    legend = {"com": "Com correção (média ± desvio de 3 sementes)", "tfidf": "TF-IDF",
              "sem": "Sem correção"}
    seen: set[str] = set()
    for (label, style, err, err_sd, eer, eer_sd), yy in zip(rows, y, strict=True):
        for ax, value, sd in ((axes[0], err, err_sd), (axes[1], eer, eer_sd)):
            dot(ax, [value], [yy], style, label=legend[style] if ax is axes[0] and style not in seen else None)
            # o desvio (≤ 0,12 p.p.) é menor que o marcador: vai no rótulo, em p.p., e não numa barra invisível
            text = pct_text(value) + (f" ± {num(sd)}" if style == "com" else "")
            ax.annotate(text, (value, yy), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
        seen.add(style)
    axes[0].set_yticks(y, [r[0] for r in rows])
    axes[0].set_xlabel("Frases classificadas errado")
    axes[1].set_xlabel("Taxa de erro igual (EER)")
    for ax in axes:
        ax.set_xlim(0, 11)
        ax.set_xticks(range(0, 11, 2))
        percent_axis(ax, "x")
        ax.grid(axis="y", visible=False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1.06), ncol=3, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "r2_teste_interno")


# --------------------------------------------------------------------- R3 · efeito da correção fora do laboratório
def r3_correcao() -> None:
    """Falsas ativações sem × com correção, para cada Transformer, em texto, voz do autor e áudio público."""
    paths = [REPORTS / "externa_texto" / "resultado.json", REPORTS / "pipeline_gravacoes" / "comparacao.json",
             REPORTS / "pipeline_publico" / "comparacao.json"]
    if not needs(*paths):
        return
    fp_text = load_json(paths[0])["fp"]

    def text_rate(run: str) -> float:
        return 100 * fp_text[run]["geral"]["fp_rate"]

    def system_rate(path: Path, name: str) -> float:
        groups = load_json(path)["comparacao"]["por_grupo"][name]
        n = sum(g["NDD"]["n"] for g in groups.values() if "NDD" in g)
        hits = sum(g["NDD"]["n"] * g["NDD"]["ativacao"] for g in groups.values() if "NDD" in g)
        return 100 * hits / n

    panels = [("Transcrições dos\ncorpora públicos", lambda m: (text_rate(f"controle/{m}-s42"),
                                                                text_rate(run_label(m)))),
              ("Voz do autor\n", lambda m: (system_rate(paths[1], f"sem_correcao_{m}"), system_rate(paths[1], m))),
              ("Áudio dos\ncorpora públicos", lambda m: (system_rate(paths[2], f"sem_correcao_{m}"),
                                                         system_rate(paths[2], m)))]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharey=True)
    y = np.arange(len(TRANSFORMERS))[::-1]
    for ax, (title, rates) in zip(axes, panels, strict=True):
        for yy, model in zip(y, TRANSFORMERS, strict=True):
            sem, com = rates(model)
            ax.plot([com, sem], [yy, yy], color=GRID, linewidth=2.5, zorder=1)
            dot(ax, [sem], [yy], "sem", label="Sem correção" if ax is axes[0] and yy == y[0] else None)
            dot(ax, [com], [yy], "com", label="Com correção" if ax is axes[0] and yy == y[0] else None)
            ax.annotate(pct_text(sem), (sem, yy), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=8)
            ax.annotate(pct_text(com), (com, yy), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlim(0, 45)
        ax.set_ylim(-0.6, len(TRANSFORMERS) - 0.3)
        percent_axis(ax, "x")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, list(TRANSFORMERS.values()))
    axes[1].set_xlabel("Falsas ativações (frases não dirigidas que o sistema aceitou)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    fig.tight_layout()
    save(fig, "r3_correcao")


# --------------------------------------------------------------------- R4 · falsas ativações por tamanho
def r4_falsas_ativacoes_por_tamanho() -> None:
    fp = load_json(REPORTS / "externa_texto" / "resultado.json")["fp"]
    corpora = ("coraa", "nurcsp", "tagarela")
    bins = [b for b in fp[run_label("bertimbau")]["coraa"]["por_faixa"] if b != ">50"]  # >50: só 12 frases

    def pooled(run: str) -> list[float]:
        out = []
        for b in bins:
            n = sum(fp[run][c]["por_faixa"][b]["n"] for c in corpora)
            hits = sum(fp[run][c]["por_faixa"][b]["n"] * fp[run][c]["por_faixa"][b]["fp_rate"] for c in corpora)
            out.append(100 * hits / n)
        return out

    sem = np.mean([pooled(f"controle/{m}-s42") for m in TRANSFORMERS], axis=0)
    com = np.mean([pooled(run_label(m)) for m in TRANSFORMERS], axis=0)
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    last = len(bins) - 1
    ax.axvspan(last - 0.5, last + 0.5, color="#f4f3f0", zorder=0)
    ax.text(last, 62, "fora do\ntreino com\ncorreção", ha="center", va="center", fontsize=8, color=INK_2)
    for values, style, label in ((sem, "sem", "Sem correção"), (com, "com", "Com correção")):
        color, marker = STYLE[style]
        ax.plot(x, values, color=color, linewidth=2, marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=1.0, label=label, zorder=3)
    for i, values, dy in ((0, sem, 8), (0, com, 8), (last, com, 8)):  # os pontos citados no texto
        ax.annotate(pct_text(values[i]), (x[i], values[i]), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(x, bins)
    ax.set_xlabel("Tamanho da frase (palavras)")
    ax.set_ylabel("Falsas ativações")
    ax.set_ylim(0, 100)
    percent_axis(ax)
    ax.legend(loc="upper center")
    save(fig, "r4_falsas_ativacoes_por_tamanho")


# --------------------------------------------------------------------- R5 · impacto do ASR
WER_BANDS = ((0.0, 0.0, "0"), (1e-9, 0.1, "até 0,1"), (0.1 + 1e-9, 0.3, "0,1 a 0,3"), (0.3 + 1e-9, 99.0, "acima de 0,3"))


def r5_asr() -> None:
    """(a) WER por condição, com o áudio público; (b) queda do acerto por faixa de WER na voz sintética e na do autor.

    O áudio público só entra em (a): ele só tem falas não dirigidas, e a queda dele está na tabela 7.
    """
    tts_path, voice_path = REPORTS / "asr_tts" / "registros.jsonl", REPORTS / "asr_gravacoes" / "registros.jsonl"
    public_path = REPORTS / "asr_publico" / "registros.jsonl"
    if not needs(tts_path, voice_path, public_path):
        return
    tts = [r for r in load_jsonl(tts_path) if r["wer"] is not None]
    voice = [r for r in load_jsonl(voice_path) if r["wer"] is not None]
    public = [r for r in load_jsonl(public_path) if r["wer"] is not None]
    groups = [("Voz sintética", tts)] + [(name, [r for r in voice if r["group"] == key])
                                         for key, name in SCENARIOS.items()] + [("Áudio público", public)]
    groups = [(name, rows) for name, rows in groups if rows]
    wer = [100 * float(np.mean([r["wer"] for r in rows])) for _, rows in groups]

    transformers = [run_label(m) for m in TRANSFORMERS]
    rows = tts + voice

    def drops(models: list[str]) -> list[float]:
        out = []
        for lo, hi, _ in WER_BANDS:
            sel = [r for r in rows if lo <= r["wer"] <= hi]
            out.append(100 * float(np.mean([np.mean([float(r[f"{m}:reference_norm"] == r["label"])
                                                     - float(r[f"{m}:hypothesis"] == r["label"]) for m in models])
                                            for r in sel])))
        return out

    n_band = [sum(lo <= r["wer"] <= hi for r in rows) for lo, hi, _ in WER_BANDS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
    xb = np.arange(len(groups))
    a1.bar(xb, wer, width=0.6, color=INK_2)
    for xi, v in zip(xb, wer, strict=True):
        a1.text(xi, v + 0.6, pct_text(v, 1), ha="center", va="bottom", fontsize=8)
    a1.set_xticks(xb, [name for name, _ in groups], rotation=30, ha="right")
    a1.set_ylabel("Palavras erradas na transcrição (WER)")
    a1.set_ylim(0, max(30, max(wer) + 5))
    percent_axis(a1)
    a1.grid(axis="x", visible=False)
    x = np.arange(len(WER_BANDS))
    for models, style, label in ((transformers, "com", "Transformers (média dos 3)"),
                                 (list(TFIDF_NAMES), "tfidf", "TF-IDF (média dos 2)")):
        color, marker = STYLE[style]
        values = drops(models)
        a2.plot(x, values, color=color, linewidth=2, marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=1.0, label=label, zorder=3)
        a2.annotate(f"{num(values[-1])} p.p.", (x[-1], values[-1]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    a2.set_xticks(x, [f"{name}\n(n = {n})" for (_, _, name), n in zip(WER_BANDS, n_band, strict=True)])
    a2.set_xlabel("WER da frase")
    a2.set_ylabel("Queda do acerto (p.p.)")
    a2.axhline(0, color=INK_2, linewidth=0.8)
    a2.set_ylim(-2, 20)  # espaço para a legenda acima dos rótulos dos pontos
    a2.legend(loc="upper left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "r5_asr")


# --------------------------------------------------------------------- R6 · os modelos no sistema completo
def r6_sistema_completo() -> None:
    """Diferença de cada modelo para o BERTimbau (pareada, IC 95%), nos três conjuntos de áudio."""
    paths = {ds: REPORTS / f"pipeline_{ds}" / "comparacao.json" for ds in ("gravacoes", "tts", "publico")}
    if not needs(*paths.values()):
        return
    others = {"baseline_logreg": ("TF-IDF + Reg. Logística", "tfidf"), "baseline_svm": ("TF-IDF + SVM", "tfidf"),
              "debertinha": ("DeBERTinha", "com"), "albertina": ("Albertina", "com")}
    panels = [("gravacoes", "DD", "Voz do autor"), ("tts", "DD", "Voz sintética"),
              ("gravacoes", "NDD", "Voz do autor"), ("tts", "NDD", "Voz sintética"), ("publico", "NDD", "Áudio público")]
    # cabeçalho, lado em que o modelo é pior que o BERTimbau, primeiro e último painel
    groups = (("Pedidos atendidos", "pior à esquerda", 0, 1), ("Falsas ativações", "pior à direita", 2, 4))
    data = {ds: load_json(p)["comparacao"]["pareado"] for ds, p in paths.items()}
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 2.5), sharey=True)
    for *_texts, first, last in groups:  # mesma escala dentro de cada medida, para comparar a largura dos IC
        for k in range(first + 1, last + 1):
            axes[k].sharex(axes[first])
    y = np.arange(len(others))[::-1]
    for ax, (ds, label, title) in zip(axes, panels, strict=True):
        ax.axvline(0, color=INK_2, linewidth=0.8, zorder=1)
        for yy, (name, (display, style)) in zip(y, others.items(), strict=True):
            v = data[ds][f"{name} vs bertimbau"][label]
            d, lo, hi = 100 * v["diferenca"], 100 * v["ic95_inf"], 100 * v["ic95_sup"]
            ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5)
            dot(ax, [d], [yy], style)
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(lambda v, _p: f"{v:+.0f}" if v else "0")
    axes[0].set_yticks(y, [d for d, _ in others.values()])
    axes[2].set_xlabel("Diferença para o BERTimbau (p.p.), com IC 95%")
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.88))
    for text, worse, first, last in groups:  # cabeçalho sobre os painéis da mesma medida, com um traço embaixo
        x0, x1, top = axes[first].get_position().x0, axes[last].get_position().x1, axes[first].get_position().y1
        fig.add_artist(Line2D([x0, x1], [top + 0.115] * 2, color=INK_2, linewidth=0.6))
        fig.text(x0, top + 0.13, text, fontsize=8.5, ha="left", va="bottom")
        fig.text(x1, top + 0.13, worse, fontsize=8, ha="right", va="bottom", color=INK_2)
    save(fig, "r6_sistema_completo")


# --------------------------------------------------------------------- R7 · tempo de resposta
def r7_latencia() -> None:
    """De onde vem o tempo entre o fim da fala e a decisão (mediana, execução do BERTimbau).

    O resto (filas e processamento do detector de voz) fica abaixo de 0,02 s em 95% das falas e não aparece.
    """
    sets = {"Voz do autor": "gravacoes", "Voz sintética": "tts", "Áudio público": "publico"}
    paths = {name: REPORTS / f"pipeline_{ds}" / "bertimbau" / "registros.jsonl" for name, ds in sets.items()}
    if not needs(*paths.values()):
        return
    wait = PIPELINE_CFG["vad"]["min_silence_ms"] / 1000
    parts = []  # (nome, espera, whisper, classificador, p50, p95)
    for name, path in paths.items():
        decisions = [d for r in load_jsonl(path) for d in r["decisions"] if d.get("latency_s") is not None]
        tasks = [t for d in decisions for t in d["tasks"]]
        total = np.array([d["latency_s"] for d in decisions])
        asr = float(np.median([t["asr_latency_s"] for t in tasks if t.get("asr_latency_s") is not None]))
        clf = float(np.median([p["latency_ms"] for t in tasks for p in (t.get("predictions") or {}).values()])) / 1000
        p50, p95 = float(np.median(total)), float(np.percentile(total, 95))
        parts.append((name, wait, asr, clf, p50, p95))
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    y = np.arange(len(parts))[::-1]
    shades = ("#c9c8c3", COM, "#0b0b0b")
    labels = ("Espera de silêncio do detector de voz", "Whisper", "Classificador")
    for yy, (name, *segments, p50, p95) in zip(y, parts, strict=True):
        left = 0.0
        for k, (value, color) in enumerate(zip(segments, shades, strict=True)):
            ax.barh(yy, value, left=left, height=0.55, color=color, edgecolor="white", linewidth=1.5,
                    label=labels[k] if yy == y[0] else None)
            left += value
        ax.text(left + 0.03, yy, f"mediana {num(p50, 2)} s · 95% até {num(p95, 2)} s", va="center", fontsize=8)
    ax.set_yticks(y, [p[0] for p in parts])
    ax.set_xlim(0, 2.6)
    ax.set_xlabel("Tempo do fim da fala até a decisão (s)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.32), ncol=3, handlelength=1.2, columnspacing=1.5)
    save(fig, "r7_latencia")


FIGURES: tuple[Callable[[], None], ...] = (m1_sistema, m2_funil_corpus, m3_treino, m4_whisper,
                                           r1_vies_comprimento, r2_teste_interno,
                                           r3_correcao, r4_falsas_ativacoes_por_tamanho, r5_asr,
                                           r6_sistema_completo, r7_latencia)


def main() -> None:
    print(f"figuras em {OUT}:")
    for figure in FIGURES:
        figure()


if __name__ == "__main__":
    main()

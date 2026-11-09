"""Tabelas do TCC num arquivo só (`uv run python scripts/tabelas_tcc.py`).

Lê os relatórios de `reports/` e grava `reports/tabelas_tcc.md`. Cada tabela cita o relatório de onde vem. Uma
tabela cujos dados ainda não existem sai como "pendente".
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PIPELINE_CFG = tomllib.loads((ROOT / "config" / "pipeline.toml").read_text(encoding="utf-8"))

TRANSFORMERS = {"bertimbau": "BERTimbau", "albertina": "Albertina", "debertinha": "DeBERTinha"}
TFIDF = {"baseline_logreg": "TF-IDF + Regressão Logística", "baseline_svm": "TF-IDF + SVM"}
LIVE = {"bertimbau": "BERTimbau", "baseline_logreg": TFIDF["baseline_logreg"], "baseline_svm": TFIDF["baseline_svm"],
        "debertinha": "DeBERTinha", "albertina": "Albertina"}
WITHOUT = {f"sem_correcao_{m}": f"{name} sem correção" for m, name in TRANSFORMERS.items()}
SCENARIOS = {"quieto_interno": "Sala quieta", "ruido_interno": "Ruído de fundo", "quieto_iphone": "iPhone",
             "distante_interno": "~2 m", "conversa_interno": "Roteiro de conversa"}
DATASETS = {"gravacoes": "Voz do autor", "tts": "Voz sintética", "publico": "Áudio dos corpora públicos"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_label(model: str) -> str:
    """Execução do Transformer no sistema (ex.: `bertimbau-s44`), a marcada em `reports/treino/treino.json`."""
    runs = load_json(REPORTS / "treino" / "treino.json")["transformers"]
    return next(r["execucao"] for r in runs if r["modelo"] == model and r["no_sistema"])


def display(label: str) -> str:
    """Nome usado no texto para o rótulo de uma execução dos relatórios."""
    for model, name in TRANSFORMERS.items():
        if label == run_label(model) or label == model:
            return name
        if label.startswith(f"controle/{model}") or label == f"sem_correcao_{model}":
            return f"{name} sem correção"
    return TFIDF.get(label, label)


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f}%".replace(".", ",")


def num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}".replace(".", ",")


def pp(value: float, digits: int = 1) -> str:
    return f"{100 * value:+.{digits}f}".replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def model_order(label: str) -> int:
    """BERTimbau, Albertina, DeBERTinha e depois os TF-IDF, como no texto do TCC."""
    names = [*TRANSFORMERS.values(), *TFIDF.values()]
    shown = display(label)
    return names.index(shown) if shown in names else len(names)


def ci(v: dict, key_lo: str = "ic95_inf", key_hi: str = "ic95_sup") -> str:
    return f"[{pp(v[key_lo])}; {pp(v[key_hi])}]"


def significant(v: dict) -> str:
    """IC 95% que não inclui o zero. Um limite igual a zero (com erro de arredondamento) não conta."""
    eps = 1e-9
    return "sim" if v["ic95_inf"] > eps or v["ic95_sup"] < -eps else "não"


def table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


class Missing(Exception):
    """Os dados da tabela ainda não existem."""


def need(*paths: Path) -> None:
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise Missing(", ".join(missing))


# ---------------------------------------------------------------------------------------------- tabelas
def t1_revisao_humana() -> tuple[str, str]:
    a = load_json(REPORTS / "corpus" / "anotacao_phelipe.json")
    m = a["matriz"]
    rows = [["Geral", a["anotadas"], pct(a["concordancia"]), num(a["kappa"])]]
    names = {"legacy": "Corpus inicial", "mitigation": "Frases novas da correção"}
    rows += [[names.get(k, k), v["n"], pct(v["concordancia"]), num(v["kappa"])] for k, v in a["por_origem"].items()]
    strategies = {"comprimento": "Geração por tamanho", "negativos_dificeis": "Negativos difíceis"}
    rows += [[strategies[k], v["n"], pct(v["concordancia"]), num(v["kappa"])]
             for k, v in a["por_estrategia"].items() if k in strategies]
    text = table(["Recorte", "Frases", "Concordância", "Kappa de Cohen"], rows)
    text += "\n\n" + table(["Rótulo do LLM", "Autor: DD", "Autor: NDD"],
                           [["DD", m["DD->DD"], m["DD->NDD"]], ["NDD", m["NDD->DD"], m["NDD->NDD"]]])
    return "Revisão humana cega (300 frases, um anotador)", text + "\n\nFonte: `reports/corpus/anotacao_phelipe.md`."


def t2_teste_interno() -> tuple[str, str]:
    t = load_json(REPORTS / "experimento" / "teste_interno.json")
    keys = ("accuracy", "f1_macro", "eer")

    def mean_sd(block: dict, k: str) -> str:
        return f"{num(block[k]['media'], 4)} ± {num(block[k]['desvio'], 4)}"

    rows = [[name, *[mean_sd(t["principal"][m], k) for k in keys]] for m, name in TRANSFORMERS.items()]
    rows += [[name, *[num(t["baselines"][m][k], 4) for k in keys]] for m, name in TFIDF.items()]
    rows += [[f"{name} sem correção", *[num(t["controle_no_teste_principal"][m]["metricas"][k], 4) for k in keys]]
             for m, name in TRANSFORMERS.items()]
    text = table(["Modelo", "Acurácia", "F1 macro", "EER"], rows)
    note = ("Transformers com correção: média ± desvio de 3 sementes. TF-IDF: 1 execução. Sem correção: 1 semente, "
            "avaliada no mesmo teste. Precisão e sensibilidade estão na tabela 3. ")
    n_test = next(iter(load_json(REPORTS / "experimento" / "erros.json")["matrizes"].values()))["n"]
    for m, name in TRANSFORMERS.items():  # desvio 0,0000: explicar, para não parecer erro de cálculo
        block = t["principal"][m]
        if block["accuracy"]["desvio"] == 0:
            wrong = round((1 - block["accuracy"]["media"]) * n_test)
            note += (f"{name}: a acurácia é a mesma nas {block['n_sementes']} sementes ({thousands(wrong)} frases "
                     "erradas em cada)")
            note += ("; o desvio da F1 macro fica abaixo de 0,00005. " if block["f1_macro"]["desvio"] < 5e-5
                     else ". ")
    note += "Fonte: `reports/experimento/teste_interno.md`."
    own = table(["Modelo sem correção", "Acurácia no teste do próprio corpus", "Acurácia no teste com correção",
                 "EER no próprio teste", "EER no teste com correção"],
                [[f"{name} sem correção", num(t["controle"][m]["accuracy"]["media"], 4),
                  num(t["controle_no_teste_principal"][m]["metricas"]["accuracy"], 4),
                  num(t["controle"][m]["eer"]["media"], 4),
                  num(t["controle_no_teste_principal"][m]["metricas"]["eer"], 4)] for m, name in TRANSFORMERS.items()])
    own_note = ("O teste do próprio corpus sem correção ainda tem o atalho do tamanho; o teste com correção, não. A "
                "queda mostra quanto do acerto vinha do atalho.")
    return ("Teste interno (11.616 frases)",
            text + "\n\n" + note + "\n\n**Modelos sem correção nos dois testes**\n\n" + own + "\n\n" + own_note)


def t3_matriz_confusao() -> tuple[str, str]:
    e = load_json(REPORTS / "experimento" / "erros.json")["matrizes"]
    rows = [[display(m), *[thousands(v[k]) for k in ("DD->DD", "DD->NDD", "NDD->DD", "NDD->NDD")],
             pct(v["DD->DD"] / (v["DD->DD"] + v["NDD->DD"]), 2), pct(v["DD->DD"] / (v["DD->DD"] + v["DD->NDD"]), 2)]
            for m, v in sorted(e.items(), key=lambda item: model_order(item[0]))]
    text = table(["Modelo", "DD aceito", "DD perdido", "NDD aceito (falsa ativação)", "NDD ignorado",
                  "Precisão (DD)", "Sensibilidade (DD)"], rows)
    note = "Semente escolhida pela validação. Fonte: `reports/experimento/erros.md`."
    return "Matriz de confusão no teste interno", text + "\n\n" + note


def _pairs_vs_tfidf(result: dict, name_of: Callable[[str], str]) -> list[list[object]]:
    """Os pares Transformer - TF-IDF, na ordem BERTimbau, Albertina, DeBERTinha."""
    order = list(TRANSFORMERS.values())
    rows = []
    for pair, by in result.items():
        a, b = pair.split(" vs ")
        if not b.startswith("baseline_") or a.startswith("baseline_"):
            continue
        rows.append([f"{name_of(a)} - {name_of(b)}",
                     *[f"{pp(by[k]['diferenca_media'])} {ci(by[k])}" for k in ("accuracy", "f1_macro", "eer")],
                     significant(by["f1_macro"])])
    return sorted(rows, key=lambda r: (order.index(r[0].split(" - ")[0]), r[0]))


def t4_bootstrap() -> tuple[str, str]:
    parts = []
    header = ["Comparação", "Acurácia (p.p.)", "F1 macro (p.p.)", "EER (p.p.)", "F1 significativo?"]
    internal = load_json(REPORTS / "experimento" / "bootstrap.json")
    parts.append("**Teste interno**\n\n" + table(header, _pairs_vs_tfidf(internal, display)))
    for ds, title in (("tts", "Transcrições da voz sintética"), ("gravacoes", "Transcrições da voz do autor")):
        path = REPORTS / f"asr_{ds}" / "hipotese.json"
        if path.exists():
            h = load_json(path)
            parts.append(f"**{title}** ({h['n']} itens)\n\n"
                         + table(header, _pairs_vs_tfidf(h["pareado_transcricao"], display)))
    note = ("Diferença A - B com IC 95% (bootstrap pareado, 2.000 reamostras). No EER, negativo favorece A. Na voz do "
            "autor, as mesmas frases foram lidas nos 4 cenários, então o bootstrap sorteia as frases (as leituras de "
            "cada uma entram juntas). Fontes: `reports/experimento/bootstrap.md` e `reports/asr_*/hipotese.md`.")
    return "Transformers × TF-IDF: bootstrap pareado", "\n\n".join(parts) + "\n\n" + note


def t5_erros() -> tuple[str, str]:
    e = load_json(REPORTS / "experimento" / "erros.json")["erros"]
    rows = [["Rotulada NDD, decidida DD", x["text"], num(x["score_dd"], 4)] for x in e["falsos_positivos"][:5]]
    rows += [["Rotulada DD, decidida NDD", x["text"], num(x["score_dd"], 4)] for x in e["falsos_negativos"][:5]]
    text = table(["Erro", "Frase (normalizada)", "p(DD) do BERTimbau"], rows)
    note = ("Os 5 erros mais confiantes de cada tipo. O teste mede a concordância com o rótulo do LLM gerador, e um "
            "erro confiante pode ser um rótulo errado no corpus. Fonte: `reports/experimento/erros.md`.")
    return "Erros mais confiantes do BERTimbau no teste interno", text + "\n\n" + note


def t6_transcricoes_publicas() -> tuple[str, str]:
    r = load_json(REPORTS / "externa_texto" / "resultado.json")
    fp, seeds = r["fp"], r["fp_sementes"]
    runs = [run_label(m) for m in TRANSFORMERS] + list(TFIDF) + [f"controle/{m}-s42" for m in TRANSFORMERS]
    rows = [[display(x), *[pct(fp[x][c]["fp_rate"]) for c in ("coraa", "nurcsp", "tagarela", "geral")]]
            for x in runs]
    text = table(["Modelo", "CORAA", "NURC-SP", "TAGARELA", "Geral"], rows)
    text += "\n\n" + table(["Modelo", "s42", "s43", "s44", "Média ± desvio"],
                           [[TRANSFORMERS[m], *[pct(v["sementes"][s]) for s in ("42", "43", "44")],
                             f"{pct(v['media'])} ± {num(100 * v['desvio'], 1)} p.p."]
                            for m, v in sorted(seeds.items(), key=lambda item: model_order(item[0]))])
    note = ("2.912 transcrições dos próprios corpora (não do Whisper), todas não dirigidas: toda ativação é falsa. "
            "Fonte: `reports/externa_texto/resultado.md`.")
    return "Falsas ativações nas transcrições dos corpora públicos", text + "\n\n" + note


def t7_asr() -> tuple[str, str]:
    need(REPORTS / "asr_tts" / "hipotese.json", REPORTS / "asr_gravacoes" / "hipotese.json",
         REPORTS / "asr_publico" / "resumo.json")
    voice = [r for r in load_jsonl(REPORTS / "asr_gravacoes" / "registros.jsonl") if r["wer"] is not None]
    tts = load_json(REPORTS / "asr_tts" / "resumo.json")
    public = load_json(REPORTS / "asr_publico" / "resumo.json")
    wer_rows = [["Voz sintética", tts["n_com_referencia"], num(tts["wer_medio"])]]
    wer_rows += [[name, len(sel), num(float(np.mean([r["wer"] for r in sel])))]
                 for key, name in SCENARIOS.items() if (sel := [r for r in voice if r["group"] == key])]
    wer_rows.append(["Áudio dos corpora públicos", public["n_com_referencia"], num(public["wer_medio"])])
    text = table(["Condição", "Itens", "WER médio"], wer_rows)
    text += (f"\n\nNo áudio dos corpora públicos, sem as pausas preenchidas (\"eh\", \"ahn\"…) da referência e da "
             f"transcrição, o WER médio fica em {num(public['wer_medio_sem_pausas'])}.")
    for ds, title in (("tts", "Voz sintética"), ("gravacoes", "Voz do autor")):
        h = load_json(REPORTS / f"asr_{ds}" / "hipotese.json")
        rows = [[display(m), pct(h["metricas"]["reference_norm"][m]["accuracy"]),
                 pct(h["metricas"]["hypothesis"][m]["accuracy"]),
                 f"{num(100 * h['queda_por_wer'][m]['queda_media'], 1)} p.p.",
                 num(h["queda_por_wer"][m]["spearman_wer_queda"]["rho"], 2)]
                for m in sorted(h["metricas"]["hypothesis"], key=model_order)]
        text += f"\n\n**{title}** ({h['n']} itens)\n\n" + table(
            ["Modelo", "Acerto no texto lido", "Acerto na transcrição", "Queda", "Spearman (WER × queda)"], rows)
    drops = public["queda_por_wer"]
    models = [m for m in public if isinstance(public[m], dict) and "asr" in public[m]]
    rows = [[display(m), pct(public[m]["referencia"]["fp_rate"]), pct(public[m]["asr"]["fp_rate"]),
             f"{pp(public[m]['asr']['fp_rate'] - public[m]['referencia']['fp_rate'])} p.p.",
             num(drops[m]["spearman_wer_queda"]["rho"], 2)]
            for m in sorted(models, key=lambda m: (m.startswith("controle/"),
                                                   model_order(m.removeprefix("controle/").split("-s")[0])))]
    text += (f"\n\n**Áudio dos corpora públicos** ({public['n_com_referencia']} trechos, todos não dirigidos)\n\n"
             + table(["Modelo", "Falsas ativações na transcrição do corpus", "Falsas ativações na transcrição do "
                      "Whisper", "Diferença", "Spearman (WER × queda)"], rows))
    note = ("Nos três blocos, os mesmos itens nas duas versões do texto. No áudio público, a referência é a "
            "transcrição do próprio corpus, e o Whisper transcreve cada trecho inteiro, sem o detector de voz.")
    return "Impacto do ASR", (text + "\n\n" + note + "\n\nFontes: `reports/asr_*/resumo.md` e "
                              "`reports/asr_{tts,gravacoes}/hipotese.md`.")


def t8_sistema_completo() -> tuple[str, str]:
    paths = {ds: REPORTS / f"pipeline_{ds}" / "comparacao.json" for ds in DATASETS}
    need(*paths.values())
    parts = []
    names = {**LIVE, **WITHOUT}
    for ds, title in DATASETS.items():
        c = load_json(paths[ds])
        groups = c["comparacao"]["por_grupo"]
        rows = []
        for name, label in names.items():
            if name not in groups:
                continue
            cells = []
            for lab in ("DD", "NDD"):
                n = sum(g[lab]["n"] for g in groups[name].values() if lab in g)
                hits = sum(g[lab]["n"] * g[lab]["ativacao"] for g in groups[name].values() if lab in g)
                cells.append(pct(hits / n) if n else "-")
            rows.append([label, *cells])
        same = c["transcricoes_iguais"]
        identical = all(v["transcricao_identica"] == v["itens"] for v in same.values())
        parts.append(f"**{title}** (transcrições idênticas entre as execuções: {'sim' if identical else 'NÃO'})\n\n"
                     + table(["Modelo", "Pedidos atendidos (DD)", "Falsas ativações (NDD)"], rows))
    note = ("Os modelos com correção rodaram ao vivo, cada um na sua execução, com a semente fixa do Whisper; os sem "
            "correção julgaram as transcrições da execução do BERTimbau. As diferenças pareadas, com IC 95%, estão na "
            "figura R6 e em `reports/pipeline_*/comparacao.md`; na voz do autor, o bootstrap sorteia as frases. "
            "O áudio dos corpora públicos só tem falas não dirigidas, então mede apenas as falsas ativações.")
    return "Sistema completo em tempo real", "\n\n".join(parts) + "\n\n" + note


def _retried(decision: dict) -> bool:
    """A fala teve nova tentativa do Whisper (algum segmento com temperatura acima de zero)."""
    return any(s.get("temperature", 0) > 0 for t in decision["tasks"] for s in t.get("segments") or [])


WHISPER_SETUPS = {"mlx": {"biblioteca": "mlx-whisper", "dispositivo": "GPU", "precisao": "fp16", "threads": None},
                  "faster-whisper-fp32": {"biblioteca": "faster-whisper", "dispositivo": "CPU", "precisao": "fp32",
                                          "threads": 4},
                  "faster-whisper-int8": {"biblioteca": "faster-whisper", "dispositivo": "CPU", "precisao": "int8",
                                          "threads": 4}}


def latest_benchmark(part: str) -> Path:
    """O benchmark mais recente que mediu `part` (o nome dos arquivos traz data e hora)."""
    found = [p for p in sorted((REPORTS / "benchmark").glob("benchmark-*.json")) if part in load_json(p)]
    if not found:
        raise Missing(f"reports/benchmark/benchmark-*.json com {part}")
    return found[-1]


def whisper_setup(name: str, entry: dict) -> str:
    s = {**WHISPER_SETUPS[name], **{k: entry[k] for k in WHISPER_SETUPS[name] if k in entry}}
    where = "GPU" if s["dispositivo"] == "GPU" else f"CPU, {s['threads']} threads"
    return f"{s['biblioteca']} {s['precisao']}, {where}"


def t9_latencia() -> tuple[str, str]:
    paths = {ds: REPORTS / f"pipeline_{ds}" / "bertimbau" / "registros.jsonl" for ds in DATASETS}
    need(*paths.values())
    bench_path = latest_benchmark("whisper")
    wait = PIPELINE_CFG["vad"]["min_silence_ms"] / 1000
    rows, rest_p95 = [], []
    for ds, path in paths.items():
        decisions = [d for r in load_jsonl(path) for d in r["decisions"] if d.get("latency_s") is not None]
        lat = np.array([d["latency_s"] for d in decisions])
        retried = np.array([_retried(d) for d in decisions])
        asr = [t["asr_latency_s"] for d in decisions for t in d["tasks"] if t.get("asr_latency_s") is not None]
        clf = [p["latency_ms"] for d in decisions for t in d["tasks"] for p in (t.get("predictions") or {}).values()]
        rest = [d["latency_s"] - wait - t["asr_latency_s"]
                - max((p["latency_ms"] for p in (t.get("predictions") or {}).values()), default=0.0) / 1000
                for d in decisions if d.get("close_reason") == "fim_da_fala"
                for t in d["tasks"] if t.get("asr_latency_s") is not None]
        rest_p95.append(float(np.percentile(rest, 95)))
        rows.append([DATASETS[ds], len(lat), f"{num(float(np.median(lat)), 2)} s",
                     f"{num(float(np.percentile(lat, 95)), 2)} s", int(retried.sum()),
                     f"{num(float(np.percentile(lat[~retried], 95)), 2)} s", f"{num(float(np.median(asr)), 2)} s",
                     f"{float(np.median(clf)):.0f} ms"])
    text = table(["Conjunto", "Falas", "Mediana", "Percentil 95", "Com nova tentativa do Whisper",
                  "Percentil 95 sem elas", "Whisper (mediana)", "Classificador (mediana)"], rows)
    bench_all = load_json(bench_path)
    bench, env = bench_all["whisper"], bench_all["ambiente"]
    gpu_cpu = table(["Whisper large-v3-turbo", "Latência (mediana)", "RTF"],
                    [[whisper_setup(key, bench[key]) + (" (o do sistema)" if key == "mlx" else ""),
                      f"{num(bench[key]['por_duracao']['5s']['p50_s'], 2)} s",
                      num(bench[key]["por_duracao"]["5s"]["rtf_p50"], 3)]
                     for key in ("mlx", "faster-whisper-fp32", "faster-whisper-int8")])
    note = (f"Do fim da fala até a decisão, com a espera de {num(wait, 1)} s de silêncio do detector de voz. "
            "Execução do BERTimbau. O resto (filas e processamento do detector de voz) tem percentil 95 de "
            f"{min(rest_p95) * 1000:.0f} a {max(rest_p95) * 1000:.0f} ms nas falas fechadas pelo fim da voz. "
            "Fonte: `reports/pipeline_*/bertimbau/`.")
    power = {"tomada": "na tomada", "bateria": "na bateria"}.get(env["energia"].get("fonte_energia"),
                                                                   "com fonte de energia desconhecida")
    bench_note = ("Benchmark com 5 s de fala sintetizada pelo macOS (RTF = tempo de processamento ÷ duração do áudio), "
                  f"{power}, com carga do sistema de {num(env['carga_1min'], 1)} no início. "
                  "Mede só o tempo, não a qualidade da transcrição. "
                  f"Fonte: `reports/benchmark/{bench_path.name}`.")
    return ("Tempo de resposta do sistema completo",
            text + "\n\n" + note + "\n\n**Whisper na GPU × na CPU**\n\n" + gpu_cpu + "\n\n" + bench_note)


def t10_custo() -> tuple[str, str]:
    c = load_json(REPORTS / "custo" / "custo.json")
    live = [load_json(REPORTS / f"pipeline_{ds}" / "comparacao.json")["inferencia_ms"] for ds in DATASETS]

    def live_range(model: str, digits: int) -> str:
        values = [block[model]["p50_ms"] for block in live]
        return f"{num(min(values), digits)}-{num(max(values), digits)} ms"

    rows = [[TRANSFORMERS[m], f"{num(v['parametros'] / 1e6, 1)} mi", f"{v['tamanho_mb']:.0f} MB",
             f"{v['treino_min']['media']:.0f} ± {v['treino_min']['desvio']:.0f} min",
             f"{num(v['pico_memoria_gb']['media'], 1)} GB",
             live_range(m, 1)] for m, v in c["transformers"].items()]
    rows += [[TFIDF[m], f"{v['vocabulario']:,} termos".replace(",", "."), f"{num(v['tamanho_mb'], 1)} MB",
              f"~{v['treino_s']:.0f} s", "-", live_range(m, 2)]
             for m, v in c["baselines"].items()]
    text = table(["Modelo", "Parâmetros", "Tamanho", "Treino por semente", "Pico de memória",
                  "Inferência ao vivo (CPU)"], rows)
    note = ("Parâmetros contados nos pesos do modelo treinado. O \"100m\" do nome da Albertina conta o codificador sem "
            "a camada de embeddings (99,8 milhões); com ela, são 139,2 milhões. Inferência: mediana por fala na CPU, "
            "com as threads do sistema, medida no sistema ao vivo; a faixa vai do menor ao maior valor nos 3 "
            "conjuntos (cada modelo na sua execução). A do BERTimbau variou entre as sessões sem causa conhecida. "
            "Fontes: `reports/custo/custo.md` e `reports/pipeline_*/comparacao.json`.")
    return "Custo dos classificadores", text + "\n\n" + note


ORIGINS = {"legacy": "Corpus inicial", "mitigation": "Frases novas da correção"}
STRATEGIES = {"legado": "-", "comprimento": "por tamanho", "negativos_dificeis": "negativos difíceis"}


def t11_corpus() -> tuple[str, str]:
    path = REPORTS / "corpus" / "principal" / "composicao.json"
    need(path)
    c = load_json(path)
    if any(g["DD"] != g["NDD"] for g in c["por_gerador"]):
        raise ValueError("composição com classes desiguais num gerador; a tabela supõe DD = NDD")
    if any(v["DD"] != v["NDD"] for v in c["particoes"].values()):
        raise ValueError("partição com classes desiguais; a tabela supõe DD = NDD")
    if any(c["palavras"]["DD"][k] != c["palavras"]["NDD"][k] for k in ("mediana", "minimo", "maximo")):
        raise ValueError("tamanho das frases diferente entre as classes; a nota supõe o mesmo")
    rows = [[ORIGINS[g["origem"]], g["modelo"], STRATEGIES[g["estrategia"]], thousands(g["DD"]),
             "< 0,1%" if g["fracao_do_corpus"] < 0.0005 else pct(g["fracao_do_corpus"])] for g in c["por_gerador"]]
    rows += [[f"**{ORIGINS[o]}**", "", "", f"**{thousands(v['DD'])}**", f"**{pct((v['DD'] + v['NDD']) / c['total'])}**"]
             for o, v in c["por_origem"].items()]
    text = table(["Origem", "Modelo gerador", "Estratégia", "Frases por classe", "Parte do corpus"], rows)
    parts = {"train": "Treino", "val": "Validação", "test": "Teste"}
    split = table(["Partição", "Frases por classe", "Total"],
                  [[parts[k], thousands(v["DD"]), thousands(v["DD"] + v["NDD"])] for k, v in c["particoes"].items()])
    words, auc = c["palavras"], c["auc_comprimento"]
    note = (f"Corpus final com {thousands(c['total'])} frases, metade de cada classe em cada linha. Mediana de "
            f"{words['DD']['mediana']:.0f} palavras nas duas classes ({words['DD']['minimo']} a "
            f"{words['DD']['maximo']}). Só o tamanho previa a classe com AUC de {num(auc['depois_dos_filtros'])} "
            f"depois dos filtros; no corpus final, {num(auc['corpus_final'])}. As partições não têm texto em comum. "
            "Fonte: `reports/corpus/principal/composicao.md`.")
    return "Corpus final", text + "\n\n" + split + "\n\n" + note


def t12_treino() -> tuple[str, str]:
    path = REPORTS / "treino" / "treino.json"
    need(path)
    r = load_json(path)
    runs = r["transformers"]
    rows = [[TRANSFORMERS[x["modelo"]] + ("" if x["condicao"] == "principal" else " sem correção"), x["semente"],
             f"{x['melhor_checkpoint']['epoca']:.0f} de {x['hiperparametros']['epocas']}",
             num(x["f1_macro_validacao_limiar_05"], 4), num(x["limiar_escolhido"], 3),
             "sim" if x["no_sistema"] else ""] for x in runs]
    text = table(["Modelo", "Semente", "Melhor checkpoint (época)", "F1 macro de validação", "Limiar escolhido",
                  "No sistema"], rows)
    first = {m: next(x for x in runs if x["modelo"] == m)["hiperparametros"] for m in TRANSFORMERS}
    hyper = table(["Hiperparâmetro", *TRANSFORMERS.values()], [
        ["Taxa de aprendizado", *[f"{first[m]['taxa_aprendizado']:.0e}".replace("e-0", "e-") for m in TRANSFORMERS]],
        ["Lote", *[first[m]["lote"] for m in TRANSFORMERS]],
        ["Épocas (máximo)", *[first[m]["epocas"] for m in TRANSFORMERS]],
        ["Tokens por frase (máximo)", *[first[m]["max_tokens"] for m in TRANSFORMERS]],
        ["Aquecimento", *[f"{100 * first[m]['aquecimento']:.0f}%" for m in TRANSFORMERS]],
        ["Dropout", *[num(first[m]["dropout"], 1) for m in TRANSFORMERS]],
        ["Weight decay", *[num(first[m]["weight_decay"], 2) for m in TRANSFORMERS]],
        ["Parada antecipada (paciência, em épocas)", *[first[m]["paciencia"] for m in TRANSFORMERS]],
        ["Precisão", *[first[m]["precisao"] for m in TRANSFORMERS]]])
    earlier = sum(x["menor_perda_antes_do_melhor"] for x in runs)
    with_c = [x["limiar_escolhido"] for x in runs if x["condicao"] == "principal"]
    without = [x["limiar_escolhido"] for x in runs if x["condicao"] == "controle"]
    tfidf = "; ".join(f"{TFIDF[t['execucao']]}: C = {num(t['c'], 0)}, F1 macro de validação {num(t['f1_macro_validacao'], 4)}"
                      f" e limiar {num(t['limiar_escolhido'], 2)}" for t in r["tfidf"])
    note = ("F1 macro de validação do modelo final, com limiar 0,5. O melhor checkpoint é o de maior F1 macro de "
            "validação; o limiar maximiza a mesma F1 e varia muito entre sementes "
            f"({num(min(with_c), 3)} a {num(max(with_c), 3)} com correção e {num(min(without), 3)} a "
            f"{num(max(without), 3)} sem correção; a tabela 13 mostra o efeito). Em {earlier} de {len(runs)} "
            "execuções, a menor perda de validação vem antes do melhor checkpoint: depois dela, a perda sobe, mas a "
            "F1 ainda melhora. A validação "
            f"dos modelos sem correção é a do corpus deles. TF-IDF (C por validação cruzada de 5 partes): {tfidf}. "
            "Fonte: `reports/treino/treino.md`.")
    return "Treino dos classificadores", text + "\n\n" + note + "\n\n**Hiperparâmetros**\n\n" + hyper


def _ndd_activation(path: Path) -> tuple[float, float, float]:
    """Falsas ativações (itens NDD com alguma fala aceita) com o limiar da execução e com 0,5, e o limiar."""
    frozen = half = n = 0
    thresholds = set()
    for record in load_jsonl(path):
        if record["label"] != "NDD":
            continue
        scores = []
        for d in record["decisions"]:
            if d.get("p_dd") is not None:  # modelos sem correção, julgando as transcrições da execução
                scores.append((d["p_dd"], d["threshold"]))
            else:  # execução ao vivo: a predição fica na tarefa da fala
                scores += [(p["p_dd"], p["threshold"]) for t in d.get("tasks", [])
                           for p in (t.get("predictions") or {}).values() if p.get("p_dd") is not None]
        hit = any(p >= thr for p, thr in scores)
        if hit != record["activated"]:
            raise ValueError(f"{path}: {record['id']} não reproduz a decisão gravada")
        thresholds |= {thr for _p, thr in scores}
        frozen += hit
        half += any(p >= 0.5 for p, _thr in scores)
        n += 1
    if len(thresholds) != 1:
        raise ValueError(f"{path}: limiares diferentes {sorted(thresholds)}")
    return frozen / n, half / n, thresholds.pop()


def t13_limiar() -> tuple[str, str]:
    runs = {**{m: name for m, name in TRANSFORMERS.items()},
            **{f"sem_correcao_{m}": f"{name} sem correção" for m, name in TRANSFORMERS.items()}}
    paths = {(run, ds): REPORTS / f"pipeline_{ds}" / run / "registros.jsonl" for run in runs for ds in DATASETS}
    need(*paths.values())
    rows, half_with, half_without = [], [], []
    for run, name in runs.items():
        cells, threshold = [], None
        for ds in DATASETS:
            frozen, half, threshold = _ndd_activation(paths[(run, ds)])
            cells.append(f"{pct(frozen)} → {pct(half)}")
            (half_without if run.startswith("sem_correcao_") else half_with).append(half)
        rows.append([name, num(threshold, 3), *cells])
    text = table(["Modelo", "Limiar da validação", *DATASETS.values()], rows)
    note = ("Falsas ativações no sistema completo com o limiar escolhido na validação → com 0,5. Com 0,5 em todos, "
            f"os modelos sem correção ficam em {pct(min(half_without))}-{pct(max(half_without))}, contra "
            f"{pct(min(half_with))}-{pct(max(half_with))} com correção. Os p(DD) gravados em cada fala reproduzem as "
            "decisões do sistema. Fonte: `reports/pipeline_*/*/registros.jsonl`.")
    return "Falsas ativações com o limiar da validação e com 0,5", text + "\n\n" + note


TABLES: tuple[Callable[[], tuple[str, str]], ...] = (
    t1_revisao_humana, t2_teste_interno, t3_matriz_confusao, t4_bootstrap, t5_erros, t6_transcricoes_publicas,
    t7_asr, t8_sistema_completo, t9_latencia, t10_custo, t11_corpus, t12_treino, t13_limiar)


#: Título das tabelas que podem sair como pendentes (dados das execuções ao vivo).
PENDING_TITLES = {"t7_asr": "Impacto do ASR", "t8_sistema_completo": "Sistema completo em tempo real",
                  "t9_latencia": "Tempo de resposta do sistema completo"}


def main() -> None:
    out = ["# Tabelas do TCC", "",
           "Geradas por `scripts/tabelas_tcc.py` a partir de `reports/`.", ""]
    for k, build in enumerate(TABLES, start=1):
        try:
            title, body = build()
        except Missing as exc:
            title, body = PENDING_TITLES.get(build.__name__, build.__name__), f"*Pendente: faltam {exc}.*"
        out += [f"## Tabela {k}. {title}", "", body, ""]
    (REPORTS / "tabelas_tcc.md").write_text("\n".join(out), encoding="utf-8")
    print(f"tabelas em {REPORTS / 'tabelas_tcc.md'}")


if __name__ == "__main__":
    main()

"""Subcomandos `ddsd eval ...` (avaliação offline do experimento)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ddsd.observability.logger import configure_logging, get_logger

log = get_logger("avaliacao")


def _setup() -> None:
    configure_logging(level="info")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cmd_baselines(args: argparse.Namespace) -> int:
    _setup()
    from ddsd.common.paths import MODELS_DIR
    from ddsd.corpus.config import load_corpus_config
    from ddsd.evaluation.baselines import train_baseline

    data_dir = args.data_dir or load_corpus_config().paths.frozen / "splits"
    for kind in ("logreg", "svm"):
        train_baseline(kind, data_dir, args.output_dir or MODELS_DIR)
    return 0


def _cmd_external_prepare(args: argparse.Namespace) -> int:
    _setup()
    from ddsd.common.io import write_json
    from ddsd.common.paths import DATA_DIR
    from ddsd.corpus.config import load_corpus_config
    from ddsd.evaluation.external_text import prepare

    out = DATA_DIR / "eval" / "externa_texto"
    stats = prepare(load_corpus_config(), out, sample_size=args.n)
    write_json(out / "preparo.json", stats)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_benchmark(subparsers)
    ev = subparsers.add_parser("eval", help="avaliação offline (baselines, externa, ASR, pipeline)")
    sub = ev.add_subparsers(dest="eval_command", required=True, metavar="<ação>")
    p = sub.add_parser("baselines", help="TF-IDF + SVM / Regressão Logística no corpus congelado")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.set_defaults(handler=_cmd_baselines)
    p = sub.add_parser("external-prepare", help="baixa e prepara as transcrições reais (CORAA/NURC-SP/TAGARELA)")
    p.add_argument("--n", type=int, default=1000, help="frases por corpus")
    p.set_defaults(handler=_cmd_external_prepare)
    _register_more(sub)
    _register_asr(sub)
    _register_parity(sub)
    _register_errors(sub)
    _register_pipeline(sub)
    _register_comparisons(sub)
    _register_session(sub)


METRICS = ("accuracy", "f1_dd", "f1_macro", "roc_auc", "eer", "mcc")


def _mean_sd(mean: float, sd: float) -> str:
    """"média ± desvio"; com uma semente só não há desvio, e o valor aparece sozinho."""
    return f"{mean:.4f}" if sd != sd else f"{mean:.4f} ± {sd:.4f}"  # sd != sd ⇔ NaN


def _bin_order(name: str) -> tuple[int, str]:
    """Ordena faixas pelo limite inferior ("2" < "7-8" < "11-13"); nomes não numéricos vão para o fim."""
    head = name.split("-")[0]
    return (int(head), name) if head.isdigit() else (10**9, name)


def _cmd_internal(args: argparse.Namespace) -> int:
    """Resumo do teste interno: média ± desvio entre sementes, por faixa, e o controle no teste principal."""
    _setup()
    import numpy as np

    from ddsd.classifiers.model import DddClassifier
    from ddsd.common.io import read_json, read_jsonl, write_json
    from ddsd.common.paths import REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.corpus.config import load_corpus_config
    from ddsd.evaluation.runs import best_seed, discover, seed_summary
    from ddsd.training.config import LABEL2ID, TRAIN_ORDER
    from ddsd.training.metrics import classification_metrics

    runs = discover()
    if not runs:
        log.error("nenhuma execução treinada em models/")
        return 1
    out = REPORTS_DIR / "experimento"
    report = MarkdownReport("Resultados no teste interno")
    summary: dict = {"principal": {}, "controle": {}, "baselines": {}, "controle_no_teste_principal": {}}
    rows = []
    for condition in ("principal", "controle"):
        for model in TRAIN_ORDER:
            s = seed_summary(runs, condition, model, METRICS)
            if s["n_sementes"]:
                summary[condition][model] = s
                rows.append([f"{condition}/{model}", s["n_sementes"],
                             *[_mean_sd(s[k]["media"], s[k]["desvio"]) for k in METRICS]])
    for run in (r for r in runs if r.condition == "baseline"):
        summary["baselines"][run.model] = run.test_metrics
        rows.append([run.model, 1, *[f"{run.test_metrics[k]:.4f}" for k in METRICS]])
    report.paragraph("Cada condição é avaliada no próprio teste. Média ± desvio padrão entre sementes.")
    report.table(["modelo", "sementes", *METRICS], rows)

    # Por faixa de comprimento (melhor semente pela validação).
    report.heading("Por faixa de comprimento (melhor semente pela validação)")
    bin_rows = []
    for model in TRAIN_ORDER:
        run = best_seed(runs, "principal", model)
        if run is None:
            continue
        by_bin = read_json(run.path / "metrics_by_bin_test.json")
        for name, m in sorted(by_bin.items(), key=lambda item: _bin_order(item[0])):
            if "accuracy" in m:
                bin_rows.append([run.label, name, m["n"], m["accuracy"], m["f1_macro"], m["eer"]])
    report.table(["execução", "faixa", "n", "acurácia", "f1_macro", "EER"], bin_rows)

    # Controle avaliado no teste principal (balanceado por comprimento): o atalho deixa de ajudar.
    test_path = load_corpus_config().paths.frozen / "splits" / "test.jsonl"
    test_rows = list(read_jsonl(test_path))
    texts = [r["text"] for r in test_rows]
    y = np.array([LABEL2ID[r["label"]] for r in test_rows])
    cross_rows = []
    for model in TRAIN_ORDER:
        run = best_seed(runs, "controle", model)
        if run is None:
            continue
        clf = DddClassifier(model, run.path, device=args.device)
        preds = clf.predict_batch(texts)
        score = np.array([p.p_dd for p in preds])
        pred = np.array([LABEL2ID[p.label] for p in preds])
        metrics = classification_metrics(y, pred, score)
        bins = {}
        for name in sorted({r["bin"] for r in test_rows}):
            mask = np.array([r["bin"] == name for r in test_rows])
            bins[name] = classification_metrics(y[mask], pred[mask], score[mask])["accuracy"]
        summary["controle_no_teste_principal"][model] = {"metricas": metrics, "acuracia_por_faixa": bins,
                                                         "execucao": run.label}
        cross_rows.append([run.label, *[metrics[k] for k in METRICS]])
    if cross_rows:
        report.heading("Controle (sem mitigação) avaliado no teste principal")
        report.table(["execução", *METRICS], cross_rows)
    write_json(out / "teste_interno.json", summary)
    report.save(out / "teste_interno.md")
    log.info("relatório em %s", out / "teste_interno.md")
    return 0


def _cmd_errors(args: argparse.Namespace) -> int:
    """Matriz de confusão e erros mais confiantes no teste interno (lê as predições já gravadas)."""
    _setup()
    from ddsd.common.io import read_json, read_jsonl, write_json
    from ddsd.common.paths import REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.evaluation.runs import best_seed, discover
    from ddsd.training.config import TRAIN_ORDER

    runs = discover()
    chosen = [best_seed(runs, "principal", m) for m in TRAIN_ORDER]
    chosen = [r for r in chosen if r is not None] + [r for r in runs if r.condition == "baseline"]
    result: dict = {"matrizes": {}, "erros": {}}
    for run in chosen:
        rows = list(read_jsonl(run.path / "predictions_test.jsonl"))
        cells = {f"{t}->{p}": sum(r["label"] == t and r["pred"] == p for r in rows)
                 for t in ("DD", "NDD") for p in ("DD", "NDD")}
        result["matrizes"][run.label] = {"n": len(rows), **cells}
    principal = best_seed(runs, "principal", "bertimbau")
    errors = read_json(principal.path / "errors_test.json")
    result["erros"] = {kind: errors[kind][: args.n] for kind in ("falsos_positivos", "falsos_negativos")}
    out = REPORTS_DIR / "experimento"
    write_json(out / "erros.json", result)
    report = MarkdownReport("Matriz de confusão e erros no teste interno")
    report.paragraph("Linhas: rótulo do corpus; colunas: decisão do modelo (semente escolhida pela validação).")
    report.table(["modelo", "n", "DD → DD", "DD → NDD (pedido perdido)", "NDD → DD (falsa ativação)", "NDD → NDD"],
                 [[m, v["n"], v["DD->DD"], v["DD->NDD"], v["NDD->DD"], v["NDD->NDD"]]
                  for m, v in result["matrizes"].items()])
    report.paragraph(f"Erros mais confiantes do {principal.label} (p(DD) do modelo). O teste mede a concordância "
                     "com o rótulo do LLM gerador: um erro confiante pode ser um rótulo errado no corpus.")
    for kind, title in (("falsos_positivos", "Rotulados NDD, decididos DD"),
                        ("falsos_negativos", "Rotulados DD, decididos NDD")):
        report.heading(title, 3)
        report.table(["frase", "p(DD)", "fonte", "faixa"],
                     [[e["text"], e["score_dd"], e["source"], e["bin"]] for e in result["erros"][kind]])
    report.save(out / "erros.md")
    log.info("relatório em %s", out / "erros.md")
    return 0


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    _setup()
    from ddsd.common.io import write_json
    from ddsd.common.paths import REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.evaluation.bootstrap import load_predictions, pairwise
    from ddsd.evaluation.runs import best_seed, discover
    from ddsd.training.config import TRAIN_ORDER

    runs = discover()
    chosen = {m: best_seed(runs, "principal", m) for m in TRAIN_ORDER}
    chosen |= {r.model: r for r in runs if r.condition == "baseline"}
    predictions = {name: load_predictions(run.path / "predictions_test.jsonl")
                   for name, run in chosen.items() if run is not None}
    result = pairwise(predictions, n_bootstrap=args.n)
    out = REPORTS_DIR / "experimento"
    write_json(out / "bootstrap.json", result)
    report = MarkdownReport(f"Bootstrap pareado ({args.n} reamostragens, teste principal)")
    report.paragraph("A - B; IC 95% por percentis; P(A melhor) = fração das reamostragens em que A supera B "
                     "(no EER, menor é melhor). Semente de cada Transformer escolhida pela validação.")
    rows = [[pair, key, v["diferenca_media"], v["ic95_inf"], v["ic95_sup"], v["p_a_melhor"]]
            for pair, metrics in result.items() for key, v in metrics.items()]
    report.table(["comparação", "métrica", "diferença", "IC95 inf", "IC95 sup", "P(A melhor)"], rows)
    report.save(out / "bootstrap.md")
    log.info("relatório em %s", out / "bootstrap.md")
    return 0


def _cmd_external(args: argparse.Namespace) -> int:
    """Taxa de FP em transcrições reais: todas as execuções, baselines inclusos, e bootstrap pareado."""
    _setup()
    import numpy as np

    from ddsd.classifiers.model import DddClassifier
    from ddsd.common.io import read_jsonl, write_json
    from ddsd.common.paths import DATA_DIR, REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.corpus.config import load_corpus_config
    from ddsd.evaluation.baselines import BaselineClassifier
    from ddsd.evaluation.external_text import false_positive_report, paired_fp_bootstrap
    from ddsd.evaluation.runs import best_seed, discover
    from ddsd.training.config import TRAIN_ORDER

    data = DATA_DIR / "eval" / "externa_texto"
    corpora = ("coraa", "nurcsp", "tagarela")
    rows = [r for corpus in corpora for r in read_jsonl(data / f"{corpus}.jsonl")]
    bins = [(b.low, b.high, b.name) for b in load_corpus_config().balance.length_bins]
    runs = discover()
    predictions = {}
    for run in runs:
        clf = (BaselineClassifier(run.path) if run.condition == "baseline"
               else DddClassifier(run.model, run.path, device=args.device))
        preds = clf.predict_batch([r["text"] for r in rows])
        predictions[run.label] = [(r, p.label, p.p_dd) for r, p in zip(rows, preds, strict=True)]
    result = false_positive_report(predictions, bins)
    for label, preds in predictions.items():
        result[label]["geral"] = {"n": len(preds), "fp_rate": sum(p == "DD" for _, p, _ in preds) / len(preds)}

    # Pares: cada Transformer (melhor semente pela validação) × cada baseline; principal × controle; e os
    # Transformers entre si (o BERTimbau, principal do TCC, primeiro).
    labels = {label: [p for _, p, _ in preds] for label, preds in predictions.items()}
    baselines = [r.label for r in runs if r.condition == "baseline"]
    pairs = []
    mains = {}
    for model in TRAIN_ORDER:
        main, control = best_seed(runs, "principal", model), best_seed(runs, "controle", model)
        if main is not None:
            mains[model] = main.label
            pairs += [(main.label, b) for b in baselines]
            if control is not None:
                pairs.append((main.label, control.label))
    ordered = [mains[m] for m in ("bertimbau", "albertina", "debertinha") if m in mains]
    pairs += [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]]
    bootstrap = {f"{a} vs {b}": paired_fp_bootstrap(labels[a], labels[b], n_bootstrap=args.n) for a, b in pairs}

    # Variação entre sementes: a semente escolhida pela validação não precisa ser a de menor FP externo.
    seeds = {}
    for model in TRAIN_ORDER:
        model_runs = sorted((r for r in runs if r.condition == "principal" and r.model == model),
                            key=lambda r: r.seed)
        if len(model_runs) > 1:
            fps = [result[r.label]["geral"]["fp_rate"] for r in model_runs]
            seeds[model] = {"sementes": {str(r.seed): fp for r, fp in zip(model_runs, fps, strict=True)},
                            "media": float(np.mean(fps)), "desvio": float(np.std(fps, ddof=1)),
                            "escolhida": mains.get(model)}

    out = REPORTS_DIR / "externa_texto"
    write_json(out / "resultado.json", {"fp": result, "bootstrap_fp": bootstrap, "fp_sementes": seeds})
    report = MarkdownReport("Validação externa em texto - taxa de falso positivo (NDD por suposição de domínio)")
    report.paragraph(f"{len(rows)} frases de conversa real ({', '.join(corpora)}), sem as iguais ou quase iguais "
                     "ao corpus. Toda ativação conta como falso positivo.")
    report.table(["execução", *corpora, "geral"],
                 [[label, *[res[c]["fp_rate"] for c in corpora], res["geral"]["fp_rate"]]
                  for label, res in result.items()])
    if seeds:
        report.heading("FP geral por semente (condição principal)", 3)
        report.paragraph("A semente usada no pipeline é a de maior F1 macro de validação; o teste e os dados "
                         "externos nunca orientam essa escolha.")
        seed_names = sorted({k for v in seeds.values() for k in v["sementes"]})
        report.table(["modelo", *[f"s{k}" for k in seed_names], "média", "desvio", "escolhida pela validação"],
                     [[m, *[v["sementes"].get(k) for k in seed_names], v["media"], v["desvio"], v["escolhida"]]
                      for m, v in seeds.items()])
    if bootstrap:
        report.heading("Bootstrap pareado da diferença de FP (A - B; negativo = A ativa menos)", 3)
        report.table(["comparação", "FP A", "FP B", "diferença", "IC95 inf", "IC95 sup", "P(A menos FP)"],
                     [[k, v["fp_a"], v["fp_b"], v["diferenca"], v["ic95_inf"], v["ic95_sup"], v["p_a_menos_fp"]]
                      for k, v in bootstrap.items()])
    report.save(out / "resultado.md")
    log.info("relatório em %s", out / "resultado.md")
    return 0


def _register_more(sub) -> None:
    p = sub.add_parser("internal", help="resumo do teste interno (sementes, faixas, controle no teste principal)")
    p.add_argument("--device", default="mps", choices=("cpu", "mps"))
    p.set_defaults(handler=_cmd_internal)
    p = sub.add_parser("bootstrap", help="bootstrap pareado entre modelos (teste principal)")
    p.add_argument("--n", type=int, default=2000)
    p.set_defaults(handler=_cmd_bootstrap)
    p = sub.add_parser("external", help="taxa de falso positivo nas transcrições reais preparadas")
    p.add_argument("--device", default="mps", choices=("cpu", "mps"))
    p.add_argument("--n", type=int, default=2000, help="reamostragens do bootstrap pareado")
    p.set_defaults(handler=_cmd_external)


def _cmd_asr(args: argparse.Namespace) -> int:
    """Impacto do ASR: TTS ou gravações reais."""
    _setup()
    from ddsd.asr.backends import create_backend
    from ddsd.asr.types import DecodingOptions
    from ddsd.classifiers.model import DddClassifier
    from ddsd.common.io import write_json
    from ddsd.common.paths import DATA_DIR, REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.corpus.config import load_corpus_config
    from ddsd.evaluation import asr_impact
    from ddsd.evaluation.baselines import BaselineClassifier
    from ddsd.evaluation.runs import best_seed, discover
    from ddsd.realtime.config import load_pipeline_config
    from ddsd.training.config import TRAIN_ORDER

    cfg = load_pipeline_config()
    a = cfg.asr
    backend = create_backend(args.backend or a.backend, mlx_model=a.mlx_model,
                             faster_whisper_model=a.faster_whisper_model, compute_type=a.faster_whisper_compute_type,
                             options=DecodingOptions(a.language, a.temperatures, a.compression_ratio_threshold,
                                                     a.logprob_threshold), mlx_cache_limit_gb=a.mlx_cache_limit_gb,
                             seed=a.seed)
    backend.warm_up()
    if args.dataset == "tts":
        test_split = load_corpus_config().paths.frozen / "splits" / "test.jsonl"
        items = asr_impact.tts_items(test_split, DATA_DIR / "eval" / "tts", n=args.n)
    elif args.dataset == "gravacoes":
        items = asr_impact.recording_items(DATA_DIR / "eval" / "gravacoes" / "manifest.jsonl", DATA_DIR)
    else:  # trechos do teste do CORAA e do NURC-SP, inteiros, como nos outros dois conjuntos
        from ddsd.evaluation.pipeline_eval import public_items

        items = public_items(DATA_DIR / "eval" / "audio_publico")
    runs = discover()
    classifiers = {}
    for model in TRAIN_ORDER:
        for condition in ("principal", "controle"):
            run = best_seed(runs, condition, model)
            if run is not None:
                classifiers[run.label] = DddClassifier(model, run.path, device="cpu")
    for run in (r for r in runs if r.condition == "baseline"):
        classifiers[run.label] = BaselineClassifier(run.path)
    records = asr_impact.transcribe_items(items, backend)
    asr_impact.classify_records(records, classifiers)
    out = REPORTS_DIR / f"asr_{args.dataset}"
    asr_impact.save_records(out / "registros.jsonl", records)
    _asr_report(args.dataset, records, list(classifiers), backend.name, out)
    return 0


def _cmd_asr_reuse(args: argparse.Namespace) -> int:
    """Refaz o relatório do impacto do ASR a partir dos registros já gravados, sem transcrever de novo."""
    from ddsd.common.io import read_jsonl
    from ddsd.common.paths import REPORTS_DIR
    from ddsd.evaluation import asr_impact

    out = REPORTS_DIR / f"asr_{args.dataset}"
    records = list(read_jsonl(out / "registros.jsonl"))
    models = list(dict.fromkeys(k.split(":")[0] for k in records[0] if k.endswith(":hypothesis")))
    _asr_report(args.dataset, records, models, "mlx", out)
    return 0


def _asr_report(dataset: str, records: list[dict], models: list[str], backend_name: str, out: Path) -> None:
    from ddsd.common.io import write_json
    from ddsd.common.report import MarkdownReport
    from ddsd.evaluation import asr_impact

    summary = asr_impact.summarize(records, models)
    summary["queda_por_wer"] = drops = asr_impact.drop_by_wer(records, models)
    write_json(out / "resumo.json", summary)
    report = MarkdownReport(f"Impacto do ASR - {dataset} ({len(records)} itens, backend {backend_name})")
    block = summary
    report.heading(f"WER médio {block['wer_medio']}", 3)
    if "wer_medio_sem_pausas" in block:
        report.paragraph(f"WER médio sem as pausas preenchidas ({', '.join(sorted(asr_impact.FILLED_PAUSES))}), "
                         f"removidas da referência e da transcrição: {block['wer_medio_sem_pausas']}")
    if "spearman_confiabilidade_wer" in block:
        s = block["spearman_confiabilidade_wer"]
        report.paragraph(f"Spearman(confiabilidade fuzzy, WER) = {s['rho']:.3f} (p = {s['p']:.2g})")
    free = block["n"] - block["n_com_referencia"]
    report.paragraph(f"Referência e ASR nos mesmos {block['n_com_referencia']} itens (pareado)."
                     + (f" Fala livre sem roteiro ({free} itens) à parte: só FP." if free else ""))
    has_dd = any(r["label"] == "DD" for r in records if r["wer"] is not None)  # o áudio público só tem NDD
    report.table(["modelo", "acc referência", "acc ASR", "FP referência", "FP ASR", *(["FN ASR"] if has_dd else []),
                  *(["FP fala livre"] if free else [])],
                 [[m, block[m]["referencia"]["acuracia"], block[m]["asr"]["acuracia"], block[m]["referencia"]["fp_rate"],
                   block[m]["asr"]["fp_rate"], *([block[m]["asr"]["fn_rate"]] if has_dd else []),
                   *([block[m]["asr_sem_referencia"]["fp_rate"]] if free else [])] for m in models])
    report.heading("Queda do acerto por faixa de WER (referência - transcrição, mesmos itens)", 3)
    report.paragraph("n por faixa: " + ", ".join(f"{k}: {v}" for k, v in drops["n_por_faixa"].items())
                     + ". Queda = 1 se o modelo acertou na referência e errou na transcrição, -1 no contrário.")
    report.table(["modelo", *drops["faixas"], "queda média", "Spearman ρ (WER × queda)", "p"],
                 [[m, *[drops[m]["queda_por_faixa"].get(b) for b in drops["faixas"]], drops[m]["queda_media"],
                   drops[m]["spearman_wer_queda"]["rho"], drops[m]["spearman_wer_queda"]["p"]] for m in models])
    report.save(out / "resumo.md")
    log.info("relatório em %s", out / "resumo.md")


def _register_asr(sub) -> None:
    p = sub.add_parser("asr", help="impacto do ASR (WER × classificador)")
    p.add_argument("dataset", choices=("tts", "gravacoes", "publico"))
    p.add_argument("--n", type=int, default=750, help="(tts) frases sintetizadas")
    p.add_argument("--backend", choices=("mlx", "faster-whisper"), default=None)
    p.set_defaults(handler=_cmd_asr)
    p = sub.add_parser("asr-relatorio", help="refaz o relatório do impacto do ASR com os registros já gravados")
    p.add_argument("dataset", choices=("tts", "gravacoes", "publico"))
    p.set_defaults(handler=_cmd_asr_reuse)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    _setup()
    from ddsd.common.io import read_json, write_json
    from ddsd.common.paths import REPORTS_DIR
    from ddsd.evaluation import benchmark
    from ddsd.realtime.config import load_pipeline_config

    out = REPORTS_DIR / "benchmark"
    if args.relatorio:  # refaz só o relatório de um JSON já gravado
        path = Path(args.relatorio)
        _benchmark_report(read_json(path), out, path.stem.removeprefix("benchmark-"))
        return 0
    env = benchmark.environment()
    if env["carga_1min"] > 3.0:
        log.warn("carga do sistema alta (%.1f): os números não representam a máquina ociosa", env["carga_1min"])
    result: dict = {"ambiente": env}
    if "whisper" in args.parts:
        result["whisper"] = benchmark.bench_whisper()
    if "frontend" in args.parts:
        result["frontend"] = benchmark.bench_frontend()
    if "classificadores" in args.parts:
        cfg = load_pipeline_config(args.config)
        from ddsd.classifiers.model import has_classifier

        runs = {k: v for k, v in cfg.classifiers.runs().items() if has_classifier(v)}
        if runs:
            result["classificadores"] = benchmark.bench_classifiers(runs)
        else:
            log.warn("nenhum classificador em %s (rode `ddsd download-models`); parte pulada", cfg.classifiers.runs())
    stamp = __import__("time").strftime("%Y%m%d-%H%M")
    write_json(out / f"benchmark-{stamp}.json", result)
    _benchmark_report(result, out, stamp)
    return 0


def _benchmark_report(result: dict, out: Path, stamp: str) -> None:
    from ddsd.common.report import MarkdownReport

    env = result["ambiente"]
    report = MarkdownReport(f"Benchmark - {env['chip']}, {env['memoria_gb']} GB, macOS {env['macos']}")
    code = f"Código: {env['codigo']}. " if "codigo" in env else ""  # os JSONs versionados não têm o commit
    report.paragraph(f"Carga do sistema no início: {env['carga_1min']}. Energia: {env['energia']}. "
                     f"{code}Versões: {env['versoes']}")
    if "whisper" in result:
        rows = [[backend, d, v["p50_s"], v["p95_s"], v["rtf_p50"]] for backend, b in result["whisper"].items()
                for d, v in b["por_duracao"].items()]
        report.table(["backend", "áudio", "p50 (s)", "p95 (s)", "RTF p50"], rows)
    if "classificadores" in result:
        names = sorted(next(iter(result["classificadores"].values()))["isolado"])
        rows = [[k, *[1000 * v["isolado"][m]["p50_s"] for m in names]] for k, v in result["classificadores"].items()]
        report.paragraph("Um classificador por vez, como no pipeline: latência p50 de uma frase.")
        report.table(["config", *[f"{n} p50 (ms)" for n in names]], rows)
    if "frontend" in result:
        report.table(["etapa", "ms por segundo de áudio"], [[k, 1000 * v] for k, v in result["frontend"].items()])
    report.save(out / f"benchmark-{stamp}.md")
    log.info("relatório em %s", out / f"benchmark-{stamp}.md")


def _register_benchmark(subparsers_root) -> None:
    p = subparsers_root.add_parser("benchmark", help="desempenho no Mac (Whisper, classificadores, frontend)")
    p.add_argument("--parts", nargs="+", default=["whisper", "frontend", "classificadores"],
                   choices=("whisper", "frontend", "classificadores"))
    p.add_argument("--config", type=Path, default=None, help="TOML do pipeline (classificadores a medir)")
    p.add_argument("--relatorio", default=None, metavar="JSON", help="refaz só o relatório de um benchmark gravado")
    p.set_defaults(handler=_cmd_benchmark)


def _cmd_parity(args: argparse.Namespace) -> int:
    """Paridade: inferência do pipeline (1 frase, CPU fp32) × predições gravadas pelo treino."""
    _setup()
    import numpy as np

    from ddsd.classifiers.model import DddClassifier
    from ddsd.common.io import read_jsonl, write_json
    from ddsd.common.paths import REPORTS_DIR

    results = {}
    for run_dir in args.runs:
        run_dir = Path(run_dir)
        model = run_dir.name.split("-")[0]
        clf = DddClassifier(model, run_dir, device="cpu")
        rows = list(read_jsonl(run_dir / "predictions_test.jsonl"))
        if args.limit:
            rows = rows[: args.limit]
        same, deltas = 0, []
        for row in rows:
            p = clf.predict(row["text"])
            same += p.label == row["pred"]
            deltas.append(abs(p.p_dd - row["score_dd"]))
        deltas_arr = np.array(deltas)
        results[str(run_dir)] = {"n": len(rows), "decisoes_identicas": same / len(rows),
                                 "max_delta_p": float(deltas_arr.max()), "media_delta_p": float(deltas_arr.mean())}
        log.info("%s: %d/%d decisões idênticas (%.4f%%), máx |Δp| %.2e", run_dir.name, same, len(rows),
                 100 * same / len(rows), deltas_arr.max())
    write_json(REPORTS_DIR / "experimento" / "paridade.json", results)
    ok = all(r["decisoes_identicas"] >= 0.999 for r in results.values())
    log.info("paridade %s (critério: ≥ 99,9%% de decisões idênticas)", "OK" if ok else "FALHOU")
    return 0 if ok else 1


def _register_errors(sub) -> None:
    p = sub.add_parser("erros", help="matriz de confusão e erros mais confiantes no teste interno")
    p.add_argument("--n", type=int, default=10, help="erros de cada tipo no relatório")
    p.set_defaults(handler=_cmd_errors)


def _register_parity(sub) -> None:
    p = sub.add_parser("parity", help="inferência do pipeline × predições do treino (≥ 99,9%% idênticas)")
    p.add_argument("runs", nargs="+", help="diretórios de execução (models/<modelo>-s<semente>)")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(handler=_cmd_parity)


def _cmd_audio_prepare(args: argparse.Namespace) -> int:
    _setup()
    from ddsd.common.paths import DATA_DIR
    from ddsd.evaluation.pipeline_eval import prepare_public_audio

    manifest = prepare_public_audio(DATA_DIR / "eval" / "audio_publico", per_corpus=args.n)
    log.info("%d áudios públicos preparados", len(manifest))
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Sistema completo em tempo real sobre um conjunto de áudio, com um classificador (pasta própria)."""
    _setup()
    from ddsd.common.io import read_jsonl, write_json, write_jsonl
    from ddsd.common.paths import DATA_DIR, REPORTS_DIR
    from ddsd.common.system import code_version
    from ddsd.evaluation import pipeline_eval
    from ddsd.realtime.config import load_pipeline_config

    if args.dataset == "gravacoes":
        items = pipeline_eval.recording_items(DATA_DIR / "eval" / "gravacoes" / "manifest.jsonl", DATA_DIR)
    elif args.dataset == "tts":
        items = pipeline_eval.tts_items(DATA_DIR / "eval" / "tts")
    else:
        items = pipeline_eval.public_items(DATA_DIR / "eval" / "audio_publico")
    if args.limit:
        items = items[: args.limit]
    overrides = {"classifiers__active": args.classificador}
    cfg = load_pipeline_config(args.config, **overrides)
    out = REPORTS_DIR / f"pipeline_{args.dataset}" / cfg.classifiers.active
    log.info("pipeline %s com %s: %d itens em tempo real (semente do Whisper %d) → %s", args.dataset,
             cfg.classifiers.active, len(items), cfg.asr.seed, out)
    records = pipeline_eval.run_items(args.config, items, overrides)
    write_jsonl(out / "registros.jsonl", records)
    summary = pipeline_eval.summarize(records)
    write_json(out / "resumo.json", {"classificador": cfg.classifiers.active, "semente_whisper": cfg.asr.seed,
                                     "codigo": code_version(), "resumo": summary})
    _pipeline_report(f"Sistema completo - {args.dataset} ({len(items)} itens), {cfg.classifiers.active}",
                     summary).save(out / "resumo.md")
    log.info("relatório em %s", out / "resumo.md")
    return 0


def _pipeline_report(title: str, summary: dict, note: str | None = None):
    """Tabelas do resumo do pipeline (por grupo × rótulo, mais a fala livre)."""
    from ddsd.common.report import MarkdownReport
    from ddsd.evaluation import pipeline_eval

    report = MarkdownReport(title)
    if note:
        report.paragraph(note)
    rows, free_rows = [], []
    for key, block in summary.items():
        if pipeline_eval.FREE_SPEECH in block:
            f = block[pipeline_eval.FREE_SPEECH]
            free_rows.append([key, f["n"], round(f["minutos"], 1), f["falas"], f["fp_por_fala"],
                              f["fp_por_fala_politica_baixa_ndd"], f["ativacoes_por_minuto"]])
        for label in pipeline_eval.LABELS:
            if label in block:
                b = block[label]
                rows.append([key, label, b["n"], b["ativacao"], b["ativacao_politica_baixa_ndd"],
                             b["sem_fala_detectada"], b["falas_por_item"],
                             block.get("latencia_s", {}).get("p50"), block.get("latencia_s", {}).get("p95")])
    report.paragraph("ativação = fração de itens em que o sistema decidiu DD (em DD é o acerto; em NDD e RUÍDO, "
                     "falso positivo). 'política baixa→NDD' = como ficaria se confiabilidade baixa forçasse NDD.")
    report.table(["grupo", "rótulo", "n", "ativação", "ativação (política)", "sem fala", "falas/item",
                  "latência p50 (s)", "latência p95 (s)"], rows)
    if free_rows:
        report.heading("Fala livre (NDD sem roteiro): falsos positivos por fala detectada", 3)
        report.paragraph("Cada arquivo tem várias falas, então a ativação por arquivo da tabela acima não se "
                         "compara à das frases. Aqui o FP é contado por fala e por minuto de áudio.")
        report.table(["grupo", "arquivos", "minutos", "falas", "FP por fala", "FP por fala (política)",
                      "ativações/min"], free_rows)
    return report


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compara os classificadores no sistema completo: os cinco ao vivo e os três sem correção."""
    _setup()
    import numpy as np
    import torch

    from ddsd.common.io import read_jsonl, write_json, write_jsonl
    from ddsd.common.paths import MODELS_DIR, REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.common.system import code_version
    from ddsd.evaluation import pipeline_eval, reclassify, resampling
    from ddsd.realtime.config import load_pipeline_config

    cfg = load_pipeline_config(args.config)
    base = REPORTS_DIR / f"pipeline_{args.dataset}"
    live = {name: list(read_jsonl(base / name / "registros.jsonl")) for name in reclassify.LIVE_ORDER
            if (base / name / "registros.jsonl").exists()}
    if reclassify.REFERENCE not in live:
        log.error("falta a execução de referência em %s", base / reclassify.REFERENCE)
        return 1
    missing = [n for n in reclassify.LIVE_ORDER if n not in live]
    if missing:
        log.warn("execuções ao vivo ausentes: %s", missing)
    reference = live[reclassify.REFERENCE]
    same = {name: reclassify.same_transcriptions(reference, recs) for name, recs in live.items()
            if name != reclassify.REFERENCE}
    for name, check in same.items():
        level = log.info if check["transcricao_identica"] == check["itens"] else log.error
        level("%s: %d de %d itens com os mesmos trechos e a mesma transcrição do BERTimbau", name,
              check["transcricao_identica"], check["itens"])
    torch.set_num_threads(cfg.classifiers.torch_threads)
    results: dict[str, list] = dict(live)
    timing = {name: reclassify.inference_ms(recs, name) for name, recs in live.items()}
    parity = None
    for name in (reclassify.REFERENCE, *reclassify.CONTROL_ORDER):
        classifier = reclassify.load_classifier(name, cfg.classifiers.runs(), MODELS_DIR, cfg.classifiers.device)
        classifier.warm_up()
        records, inference = reclassify.reclassify(reference, name, classifier)
        if name == reclassify.REFERENCE:
            try:
                parity = reclassify.check_parity(reference, records, name)
            except reclassify.ParityError as exc:
                log.error("paridade falhou (nada foi salvo): %s", exc)
                return 1
            log.info("paridade: %d de %d decisões idênticas à execução em tempo real (max |Δp| %.2g)",
                     parity["identicas"], parity["falas"], parity["max_delta_p"])
            continue
        results[name] = records  # o tempo de inferência daqui é fora do tempo real: não entra na tabela
        summary = pipeline_eval.summarize(records)
        target = base / name
        write_jsonl(target / "registros.jsonl", records)
        write_json(target / "resumo.json", {"classificador": name, "resumo": summary})
        note = ("Este modelo julgou as transcrições da execução em tempo real com o BERTimbau (as mesmas das "
                "outras execuções); a latência ponta a ponta é a daquela execução.")
        _pipeline_report(f"Sistema completo - {args.dataset}, {reclassify.DISPLAY[name]} (mesmas transcrições)",
                         summary, note=note).save(target / "resumo.md")
        log.info("%s salvo em %s", reclassify.DISPLAY[name], target)
    stats = {name: {"n": len(t), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}
             for name, t in timing.items() if t}
    group_field = resampling.group_field(args.dataset)
    comparison = reclassify.compare(results, group_field=group_field)
    unit = resampling.unit_name(group_field)
    origins = None
    asr_path = REPORTS_DIR / f"asr_{args.dataset}" / "registros.jsonl"
    if asr_path.exists():  # conjuntos com texto de referência: de onde vem cada erro
        from ddsd.evaluation.error_origin import error_origins

        asr_records = list(read_jsonl(asr_path))
        labels = {n: (f"controle/{reclassify.control_run(n.removeprefix(reclassify.CONTROL_PREFIX), MODELS_DIR).name}"
                      if n.startswith(reclassify.CONTROL_PREFIX) else cfg.classifiers.run_for(n).name)
                  for n in results}
        origins = {n: error_origins(recs, asr_records, labels[n]) for n, recs in results.items()}
    write_json(base / "comparacao.json", {"codigo": code_version(), "transcricoes_iguais": same, "paridade": parity,
                                          "inferencia_ms": stats, "unidade_do_bootstrap": unit,
                                          "comparacao": comparison, "origem_dos_erros": origins})
    names = [n for n in reclassify.CLASSIFIER_ORDER if n in results]
    report = MarkdownReport(f"Comparação dos classificadores no sistema completo - {args.dataset}")
    report.paragraph("Os modelos com correção rodaram ao vivo, cada um na sua execução, com a semente fixa do "
                     "Whisper; os sem correção julgaram as transcrições da execução do BERTimbau. Ativação = "
                     "fração de itens decididos DD: em DD são os pedidos atendidos; nos demais, falsas ativações.")
    report.table(["execução", "itens", "mesmos trechos", "mesma transcrição"],
                 [[reclassify.DISPLAY[n], c["itens"], c["segmentacao_identica"], c["transcricao_identica"]]
                  for n, c in same.items()])
    report.paragraph(f"Paridade dos modelos sem correção: o BERTimbau, julgando de novo, reproduziu "
                     f"{parity['identicas']} de {parity['falas']} decisões da execução em tempo real.")
    rows = []
    groups = sorted({g for per in comparison["por_grupo"].values() for g in per})
    for g in groups:
        for lab in ("DD", "NDD", "NDD_LIVRE", "RUIDO"):
            present = [n for n in names if lab in comparison["por_grupo"][n][g]]
            if present:
                rows.append([g, lab, comparison["por_grupo"][present[0]][g][lab]["n"],
                             *[comparison["por_grupo"][n][g].get(lab, {}).get("ativacao") for n in names]])
    report.table(["grupo", "rótulo", "n", *[reclassify.DISPLAY[n] for n in names]], rows)
    pair_rows = []
    for key, by_label in comparison["pareado"].items():
        a, b = key.split(" vs ")
        pair_rows += [[f"{reclassify.DISPLAY[a]} - {reclassify.DISPLAY[b]}", lab, v["fp_a"], v["fp_b"],
                       v["diferenca"], v["ic95_inf"], v["ic95_sup"]] for lab, v in by_label.items()]
    if pair_rows:
        report.heading("Diferenças pareadas (A - B; todos os grupos juntos; bootstrap de 2.000)", 3)
        report.paragraph("Cada modelo contra o BERTimbau e cada Transformer contra a própria versão sem correção."
                         + (" O bootstrap sorteia as frases: as leituras da mesma frase nos 4 cenários entram "
                            "juntas, porque os erros de um modelo se repetem na mesma frase."
                            if unit == "frase" else ""))
        report.table(["comparação", "rótulo", "ativação A", "ativação B", "diferença", "IC95 inf", "IC95 sup"],
                     pair_rows)
    if origins:
        report.heading("De onde vêm os erros", 3)
        report.paragraph("Segmentação: o detector de voz não achou fala ou dividiu o item. Classificador: o modelo "
                         "também erra no texto lido. Transcrição: acerta no texto lido e erra na transcrição.")
        report.table(["modelo", "rótulo", "segmentação", "classificador", "transcrição"],
                     [[reclassify.DISPLAY[n], lab, *[c[o] for o in ("segmentacao", "classificador", "transcricao")]]
                      for n in names for lab, c in origins[n]["por_rotulo"].items()])
    report.heading("Tempo de inferência por fala no sistema ao vivo (CPU, mesmas threads)", 3)
    report.paragraph("Cada modelo medido na sua execução ao vivo. Os modelos sem correção têm a mesma arquitetura "
                     "dos com correção e, portanto, o mesmo custo.")
    report.table(["classificador", "falas", "p50 (ms)", "p95 (ms)"],
                 [[reclassify.DISPLAY[n], t["n"], t["p50_ms"], t["p95_ms"]] for n, t in stats.items()])
    report.save(base / "comparacao.md")
    log.info("comparação em %s", base / "comparacao.md")
    return 0


def _register_pipeline(sub) -> None:
    from ddsd.realtime.config import PIPELINE_CHOICES

    p = sub.add_parser("audio-prepare", help="baixa e recorta áudio de teste do CORAA e do NURC-SP")
    p.add_argument("--n", type=int, default=300)
    p.set_defaults(handler=_cmd_audio_prepare)
    p = sub.add_parser("pipeline", help="sistema completo em tempo real sobre um conjunto de áudio")
    p.add_argument("dataset", choices=("gravacoes", "tts", "publico"))
    p.add_argument("--classificador", choices=PIPELINE_CHOICES, default=None,
                   help="classificador da execução (padrão: [classifiers] active); saída em "
                        "reports/pipeline_<conjunto>/<classificador>/")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(handler=_cmd_pipeline)
    r = sub.add_parser("comparar", help="compara as execuções ao vivo e julga as mesmas transcrições com os "
                                        "modelos sem correção")
    r.add_argument("dataset", choices=("gravacoes", "tts", "publico"))
    r.add_argument("--config", type=Path, default=None)
    r.set_defaults(handler=_cmd_compare)


def _cmd_asr_hypothesis(args: argparse.Namespace) -> int:
    """A hipótese nas transcrições do Whisper: acurácia, F1 macro e EER, com bootstrap pareado entre modelos."""
    _setup()
    from ddsd.common.io import read_jsonl, write_json
    from ddsd.common.paths import REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.evaluation import resampling
    from ddsd.evaluation.asr_impact import drop_by_wer, scored_rows, text_predictions
    from ddsd.evaluation.bootstrap import pairwise
    from ddsd.evaluation.runs import best_seed, discover
    from ddsd.training.config import TRAIN_ORDER
    from ddsd.training.metrics import classification_metrics

    base = REPORTS_DIR / f"asr_{args.dataset}"
    records = list(read_jsonl(base / "registros.jsonl"))
    runs = discover()
    models = [best_seed(runs, "principal", m).label for m in TRAIN_ORDER if best_seed(runs, "principal", m)]
    models += [r.label for r in runs if r.condition == "baseline"]
    keys = ("accuracy", "precision_dd", "recall_dd", "f1_macro", "eer")
    metrics = {}
    for field in ("reference_norm", "hypothesis"):
        preds = text_predictions(records, models, field)
        metrics[field] = {m: {k: classification_metrics(v["y_true"], v["y_pred"], v["y_score"])[k] for k in keys}
                          for m, v in preds.items()}
    hyp = text_predictions(records, models, "hypothesis")
    drops = drop_by_wer(records, models)
    group_by = resampling.group_field(args.dataset)  # o laço abaixo usa `field` para o tipo de texto
    groups = None if group_by is None else [r[group_by] for r in scored_rows(records)]
    result = {"n": int(len(next(iter(hyp.values()))["y_true"])), "metricas": metrics, "queda_por_wer": drops,
              "unidade_do_bootstrap": resampling.unit_name(group_by),
              "pareado_transcricao": pairwise(hyp, n_bootstrap=args.n, groups=groups)}
    write_json(base / "hipotese.json", result)
    report = MarkdownReport(f"Hipótese nas transcrições do Whisper - {args.dataset} ({result['n']} itens)")
    report.paragraph("Os mesmos itens para todos os modelos: o texto de referência (o que foi lido) e a transcrição "
                     "do Whisper. Bootstrap pareado (2.000 reamostras) nas transcrições, como no teste interno: "
                     "A - B, IC 95% e P(A melhor); no EER, menor é melhor."
                     + (" O bootstrap sorteia as frases: as leituras da mesma frase nos 4 cenários entram juntas."
                        if group_by else ""))
    for field, title in (("hypothesis", "Na transcrição do Whisper"), ("reference_norm", "No texto de referência")):
        report.heading(title, 3)
        report.table(["modelo", "acurácia", "precisão DD", "sensibilidade DD", "F1 macro", "EER"],
                     [[m, *[v[k] for k in keys]] for m, v in metrics[field].items()])
    report.heading("Queda do acerto por faixa de WER (texto de referência - transcrição, mesmos itens)", 3)
    report.paragraph("n por faixa: " + ", ".join(f"{k}: {v}" for k, v in drops["n_por_faixa"].items())
                     + ". Spearman entre o WER e a queda, item a item (queda = 1 se o modelo acertou no texto "
                     "e errou na transcrição, -1 no contrário, 0 se não mudou)."
                     + (" O p trata as leituras da mesma frase como independentes, então é otimista." if group_by
                        else ""))
    report.table(["modelo", *drops["faixas"], "queda média", "Spearman ρ", "p"],
                 [[m, *[drops[m]["queda_por_faixa"].get(b) for b in drops["faixas"]], drops[m]["queda_media"],
                   drops[m]["spearman_wer_queda"]["rho"], drops[m]["spearman_wer_queda"]["p"]] for m in models])
    report.heading("Diferenças pareadas na transcrição", 3)
    report.table(["comparação", "métrica", "diferença", "IC95 inf", "IC95 sup", "P(A melhor)"],
                 [[pair, k, v["diferenca_media"], v["ic95_inf"], v["ic95_sup"], v["p_a_melhor"]]
                  for pair, by in result["pareado_transcricao"].items() for k, v in by.items()])
    report.save(base / "hipotese.md")
    log.info("hipótese nas transcrições em %s", base / "hipotese.md")
    return 0


def _cmd_asr_robustness(args: argparse.Namespace) -> int:
    """Robustez ao ASR: a queda de acerto (texto → transcrição) de cada modelo, comparada em pares."""
    _setup()
    from ddsd.common.io import read_jsonl, write_json
    from ddsd.common.paths import REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.evaluation import resampling
    from ddsd.evaluation.asr_impact import paired_drop_bootstrap
    from ddsd.evaluation.runs import best_seed, discover

    base = REPORTS_DIR / f"asr_{args.dataset}"
    records = list(read_jsonl(base / "registros.jsonl"))
    runs = discover()
    mains = {m: best_seed(runs, "principal", m).label for m in ("bertimbau", "albertina", "debertinha")
             if best_seed(runs, "principal", m) is not None}
    baselines = [r.label for r in runs if r.condition == "baseline"]
    pairs = [(mains["bertimbau"], other) for other in [mains[m] for m in ("albertina", "debertinha") if m in mains]]
    pairs += [(model, b) for model in mains.values() for b in baselines]
    field = resampling.group_field(args.dataset)
    result = {f"{a} vs {b}": paired_drop_bootstrap(records, a, b, n_bootstrap=args.n, group_field=field)
              for a, b in pairs}
    write_json(base / "robustez.json", {"unidade_do_bootstrap": resampling.unit_name(field), "pares": result})
    report = MarkdownReport(f"Robustez ao ASR - {args.dataset}: queda de acerto do texto de referência para a "
                            "transcrição do Whisper")
    report.paragraph("Queda = acerto no texto de referência - acerto na transcrição, nos mesmos itens. Diferença "
                     "A - B com IC 95% por bootstrap pareado (2.000 reamostras); negativo = A perde menos."
                     + (" O bootstrap sorteia as frases: as leituras da mesma frase nos 4 cenários entram juntas."
                        if field else ""))
    report.table(["comparação", "n", "queda A", "queda B", "diferença", "IC95 inf", "IC95 sup", "P(A perde menos)"],
                 [[k, v["n"], v["queda_a"], v["queda_b"], v["diferenca"], v["ic95_inf"], v["ic95_sup"],
                   v["p_a_perde_menos"]] for k, v in result.items()])
    report.save(base / "robustez.md")
    log.info("robustez ao ASR em %s", base / "robustez.md")
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    """Custo dos classificadores, a partir dos artefatos e relatórios já gravados."""
    _setup()
    from ddsd.common.io import read_json, write_json
    from ddsd.common.paths import MODELS_DIR, REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.evaluation import cost
    from ddsd.evaluation.runs import best_seed, discover

    runs = discover()
    transformers = {}
    for model in ("bertimbau", "albertina", "debertinha"):
        chosen = best_seed(runs, "principal", model)
        if chosen is not None:
            transformers[model] = cost.transformer_cost(
                [r for r in runs if r.condition == "principal" and r.model == model], chosen)
    baselines = {r.label: cost.baseline_cost(r.path) for r in runs if r.condition == "baseline"}
    # o benchmark mais recente que mediu os classificadores (um benchmark só do Whisper não serve aqui)
    benchmarks = [p for p in sorted((REPORTS_DIR / "benchmark").glob("benchmark-*.json"))
                  if "classificadores" in read_json(p)]
    inference = cost.inference_from_reports(REPORTS_DIR / "pipeline_gravacoes" / "comparacao.json",
                                            benchmarks[-1] if benchmarks else None)
    out = REPORTS_DIR / "custo"
    write_json(out / "custo.json", {"transformers": transformers, "baselines": baselines, "inferencia": inference,
                                    "benchmark": benchmarks[-1].name if benchmarks else None})
    report = MarkdownReport("Custo dos classificadores")
    report.paragraph("Parâmetros contados nos pesos do melhor checkpoint; treino = média ± desvio das sementes "
                     "(Mac M5, MPS); inferência por fala na CPU com as threads do pipeline, medida no sistema ao "
                     "vivo na voz do autor (cada modelo na sua execução).")

    def ms(name: str) -> float | None:
        return inference.get(name, {}).get("pipeline_p50_ms")

    def num(value: float | None, digits: int = 1) -> str:
        return "-" if value is None else f"{value:.{digits}f}".replace(".", ",")

    rows = [[m, c["checkpoint_base"], num(c["parametros"] / 1e6), num(c["tamanho_mb"], 0), c["precisao_treino"],
             f"{c['treino_min']['media']:.0f} ± {c['treino_min']['desvio']:.0f}",
             num(c["pico_memoria_gb"]["media"]), num(ms(m))] for m, c in transformers.items()]
    rows += [[n, "TF-IDF de 1 a 3-gramas, " + f"{c['vocabulario']:,}".replace(",", ".") + " termos", "-",
              num(c["tamanho_mb"]), "-", num(c["treino_s"] / 60, 2), "-", num(ms(n), 2)] for n, c in baselines.items()]
    report.table(["modelo", "base", "parâmetros (mi)", "tamanho (MB)", "precisão do treino", "treino (min)",
                  "pico de memória no treino (GB)", "inferência p50 (ms)"], rows)
    bench_rows = [[n, *[num(v.get(f"benchmark_{k}_p50_ms")) for k in ("cpu-1t", "cpu-2t", "cpu-4t", "mps-2t")]]
                  for n, v in inference.items() if any(k.startswith("benchmark_") for k in v)]
    if bench_rows:
        report.heading(f"Benchmark dos Transformers ({benchmarks[-1].name}), p50 em ms por frase", 3)
        report.table(["modelo", "CPU 1 thread", "CPU 2 threads", "CPU 4 threads", "MPS"], bench_rows)
    report.save(out / "custo.md")
    log.info("custo em %s", out / "custo.md")
    return 0


def _register_comparisons(sub) -> None:
    p = sub.add_parser("asr-robustez", help="queda de acerto causada pelo ASR, comparada entre os modelos")
    p.add_argument("dataset", choices=("tts", "gravacoes", "publico"))
    p.add_argument("--n", type=int, default=2000, help="reamostras do bootstrap")
    p.set_defaults(handler=_cmd_asr_robustness)
    p = sub.add_parser("asr-hipotese", help="a hipótese (Transformers × TF-IDF) nas transcrições do Whisper")
    p.add_argument("dataset", choices=("tts", "gravacoes"))
    p.add_argument("--n", type=int, default=2000, help="reamostras do bootstrap")
    p.set_defaults(handler=_cmd_asr_hypothesis)
    p = sub.add_parser("custo", help="parâmetros, tamanho, treino, memória e inferência de cada classificador")
    p.set_defaults(handler=_cmd_cost)


def _cmd_session(args: argparse.Namespace) -> int:
    """Pontua uma sessão ao vivo gravada (`ddsd run --gravar`) contra o roteiro."""
    _setup()
    import soundfile as sf
    import torch

    from ddsd.classifiers.model import DddClassifier
    from ddsd.common.io import file_sha256, read_json, read_jsonl, write_json, write_jsonl
    from ddsd.common.paths import PROJECT_ROOT, REPORTS_DIR, resolve_path
    from ddsd.common.system import code_version
    from ddsd.evaluation import live_session, reclassify
    from ddsd.text.normalization import normalize_text

    session_dir, script_path = resolve_path(args.sessao), resolve_path(args.roteiro)
    session = read_json(session_dir / "sessao.json")
    evidences = list(read_jsonl(session_dir / "evidencias.jsonl"))
    script = live_session.parse_script(script_path)
    speakers = [name.strip() for name in args.falantes.split(",") if name.strip()]
    spoken = script.spoken_by(speakers)
    if not spoken:
        log.error("nenhuma linha do roteiro é de %s", speakers)
        return 1
    clf_cfg, asr_cfg = (session["metadados"]["config"][k] for k in ("classifiers", "asr"))
    active = clf_cfg["active"]
    torch.set_num_threads(clf_cfg["torch_threads"])
    classifier = DddClassifier(active, Path(clf_cfg[active]), device=clf_cfg["device"])
    used = {e["votes"][active]["threshold"] for e in evidences if active in e["votes"]}
    if used != {classifier.threshold}:
        log.error("limiar da sessão %s diferente do carregado (%.3f)", sorted(used), classifier.threshold)
        return 1
    classifier.warm_up()
    # Paridade: julgar de novo as transcrições da sessão precisa repetir todas as decisões.
    pseudo = [{"id": session_dir.name, "group": "sessao", "label": None, "reference": None, "audio_s": None,
               "decisions": evidences}]
    rejudged, _ = reclassify.reclassify(pseudo, active, classifier)
    try:
        parity = reclassify.check_parity(pseudo, rejudged, active)
    except reclassify.ParityError as exc:
        log.error("paridade falhou (nada foi salvo): %s", exc)
        return 1
    log.info("paridade: %d de %d decisões idênticas (max |Δp| %.2g)", parity["identicas"], parity["falas"],
             parity["max_delta_p"])

    def judge(text: str) -> tuple[str, float]:
        prediction = classifier.predict(text)
        return prediction.label, prediction.p_dd

    groups = live_session.align([normalize_text(line.text) for line in spoken], [e["normalized"] for e in evidences])
    records = live_session.score(groups, spoken, evidences, script, judge)
    summary = live_session.summarize(records, spoken, script)
    audio_s = {path.name: sf.info(path).duration for path in sorted(session_dir.glob("U*.wav"))}

    def shown(path: Path) -> str:
        return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)

    payload = {
        "codigo": code_version(), "sessao": session_dir.name, "diretorio": shown(session_dir),
        "evidencias_sha256": file_sha256(session_dir / "evidencias.jsonl"),
        "roteiro": shown(script_path), "roteiro_sha256": file_sha256(script_path), "falantes": speakers,
        "inicio": session["inicio"], "fim": session["resumo"]["fim"],
        "classificador": {"nome": active, "execucao": Path(clf_cfg[active]).name, "limiar": classifier.threshold},
        "asr": {"backend": asr_cfg["backend"], "semente": asr_cfg.get("seed"),
                "modelo": asr_cfg["mlx_model"] if asr_cfg["backend"] == "mlx" else asr_cfg["faster_whisper_model"]},
        "paridade": parity, "resumo": summary, "latencia": live_session.latency_summary(evidences),
        "verificacoes": live_session.session_checks(evidences, session, audio_s),
    }
    out = resolve_path(args.saida) if args.saida else REPORTS_DIR / "sessao_ao_vivo"
    write_json(out / "resumo.json", payload)
    write_jsonl(out / "grupos.jsonl", records)
    live_session.build_report(payload, records).save(out / "resumo.md")
    dd, ndd = summary["pedidos_atendidos"], summary["falsas_ativacoes"]
    log.info("pedidos atendidos %d de %d; falsas ativações %d de %d; relatório em %s", dd["ativados"], dd["n"],
             ndd["ativados"], ndd["n"], out / "resumo.md")
    return 0


def _register_session(sub) -> None:
    p = sub.add_parser("sessao", help="pontua uma sessão ao vivo gravada contra o roteiro")
    p.add_argument("sessao", type=Path, help="diretório da sessão (data/sessoes/<data-hora>)")
    p.add_argument("--roteiro", type=Path, default=Path("docs/roteiro_sessao_ao_vivo.md"))
    p.add_argument("--falantes", default="você",
                   help="quem leu as próprias falas na sessão, separado por vírgula (padrão: você)")
    p.add_argument("--saida", type=Path, default=None, help="padrão: reports/sessao_ao_vivo")
    p.set_defaults(handler=_cmd_session)

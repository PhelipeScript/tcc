"""Subcomandos `ddsd corpus ...` e `ddsd generate ...`."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddsd.observability.logger import configure_logging, get_logger

log = get_logger("corpus")


def _config(args: argparse.Namespace):
    from ddsd.corpus.config import load_corpus_config

    configure_logging(level="debug" if args.debug else "info")
    return load_corpus_config(args.config)


def _cmd_import_legacy(args: argparse.Namespace) -> int:
    from ddsd.corpus.sources import import_legacy

    import_legacy(_config(args), force=args.force)
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from ddsd.corpus.build import build_corpus

    cfg = _config(args)
    summary = build_corpus(cfg, args.output)
    final = summary["diagnostics"]["final"]
    log.info("relatório: %s", (args.output or cfg.paths.build) / "report.md")
    log.info("AUC(comprimento→DD) final = %.3f | partições = %s", final["auc_comprimento"], summary["split"])
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    from ddsd.corpus.freeze import SourcesChanged, freeze_corpus

    cfg = _config(args)
    try:
        archive = freeze_corpus(cfg.paths.build, cfg.paths.frozen, cfg.paths.sources, force=args.force,
                                reports_dir=cfg.paths.reports)
    except (FileNotFoundError, FileExistsError, SourcesChanged) as exc:
        log.error("%s", exc)
        return 1
    log.info("corpus congelado em %s, com as fontes do build em %s (use este diretório no treino); "
             "relatório copiado para %s", cfg.paths.frozen, archive.name, cfg.paths.reports)
    return 0


def _cmd_generate_plan(args: argparse.Namespace) -> int:
    from ddsd.common.report import MarkdownReport
    from ddsd.corpus.generation.generator import CorpusState, target_bins

    cfg = _config(args)
    state = CorpusState.load(cfg)
    bins = target_bins(cfg)
    sources = sorted({s for (origin, s, _, _) in state.counts if origin == "mitigation"})
    report = MarkdownReport("Mitigação: frases aceitas por fonte (DD/NDD) em cada faixa-alvo")
    if not sources:
        report.paragraph("Nenhuma frase de mitigação ainda.")
    rows = []
    for source in sources:
        cells = [(state.mitigation_count(source, b.name, "DD"), state.mitigation_count(source, b.name, "NDD"))
                 for b in bins]
        rows.append([source, *[f"{dd}/{ndd}" for dd, ndd in cells], sum(min(c) for c in cells)])
    if rows:
        report.table(["fonte", *[b.name for b in bins], "pares"], rows)
    print(report.render())
    return 0


def _cmd_generate_run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    from ddsd.common.paths import PROJECT_ROOT
    from ddsd.corpus.generation.generator import run_generation

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = _config(args)
    run_generation(cfg, args.provider, per_cell=args.per_cell, source_name=args.source, sleep=args.sleep,
                   batch_size=args.batch_size, strategy=args.strategy)
    return 0


def _cmd_generate_ingest(args: argparse.Namespace) -> int:
    from ddsd.corpus.generation.generator import ingest_file

    cfg = _config(args)
    # Sem juiz: frases ingeridas não passam pela autoverificação por LLM (ver docs/protocolo_experimental.md).
    provenance = {"source": args.source, "model": args.model, "provider": "subagente", "strategy": args.strategy,
                  "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"), "file": args.file.name}
    ingest_file(cfg, args.source, args.label, args.file, provenance)
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    from ddsd.corpus.annotation import annotate

    cfg = _config(args)
    return annotate(args.annotator, cfg.paths.frozen / "splits", n=args.n)


def _cmd_agreement(args: argparse.Namespace) -> int:
    from ddsd.corpus.annotation import agreement_report

    _config(args)
    path = agreement_report(args.annotator)
    log.info("relatório: %s", path)
    print(path.read_text(encoding="utf-8")[:1500])
    return 0


def _cmd_self_check(args: argparse.Namespace) -> int:
    """Taxa da autoverificação do rótulo, somada dos logs da geração (`data/logs/geracao_*.log`)."""
    from ddsd.common.io import read_jsonl, write_json
    from ddsd.common.paths import DATA_DIR, REPORTS_DIR
    from ddsd.common.report import MarkdownReport
    from ddsd.corpus.self_check import self_check_stats

    cfg = _config(args)
    stats = self_check_stats(list((DATA_DIR / "logs").glob("geracao_*.log")))
    # Cobertura: frases verificadas nas fontes (com `verified_by`) e no corpus congelado.
    verified = sum(1 for f in (cfg.paths.sources / "mitigation").glob("*/*.jsonl")
                   for r in read_jsonl(f) if r.get("verified_by"))
    corpus = [r for s in ("train", "val", "test") for r in read_jsonl(cfg.paths.frozen / "splits" / f"{s}.jsonl")]
    mitigation = [r for r in corpus if r["origin"] == "mitigation"]
    claude = sum(r["source"].startswith("claude") for r in mitigation)
    logged = stats["total"]["DD"]["aceitas"] + stats["total"]["NDD"]["aceitas"]
    stats["cobertura"] = {"aceitas_nos_logs": logged, "verificadas_nas_fontes": verified,
                          "corpus_final": len(corpus), "mitigacao_no_corpus": len(mitigation),
                          "mitigacao_verificada_no_corpus": len(mitigation) - claude, "claude_sem_verificacao": claude}
    out = REPORTS_DIR / "corpus"
    write_json(out / "autoverificacao.json", stats)
    report = MarkdownReport("Autoverificação do rótulo na geração")
    c = stats["cobertura"]
    report.paragraph(
        f"O mesmo modelo que gerou o lote o reclassifica em temperatura 0, sem ver o rótulo pretendido; só ficam "
        f"as frases em que ele confirma o rótulo. No corpus final, {c['mitigacao_verificada_no_corpus']} das "
        f"{c['corpus_final']} frases passaram por ela; o corpus inicial e as "
        f"{c['claude_sem_verificacao']} frases do Claude, não. Os logs cobrem {c['aceitas_nos_logs']} das "
        f"{c['verificadas_nas_fontes']} frases verificadas nas fontes; a taxa é parcial.")
    rows = []
    for scope, blocks in (("total", {"total": stats["total"]}), ("estratégia", stats["por_estrategia"]),
                          ("log", stats["por_log"])):
        for name, by_label in blocks.items():
            rows += [[scope, name, label, v["julgadas"], v["juiz_discorda"], v["taxa_discorda"], v["juiz_invalido"]]
                     for label, v in by_label.items()]
    report.table(["escopo", "nome", "classe", "julgadas", "discordâncias", "taxa de discordância",
                  "descartadas (resposta inválida)"], rows)
    report.save(out / "autoverificacao.md")
    log.info("relatório: %s", out / "autoverificacao.md")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="arquivo TOML (padrão: config/corpus.toml)")
    common.add_argument("--debug", action="store_true", help="habilita logs de debug")

    corpus = subparsers.add_parser("corpus", help="preparação do corpus DD/NDD")
    sub = corpus.add_subparsers(dest="corpus_command", required=True, metavar="<ação>")

    p = sub.add_parser("import-legacy", parents=[common], help="copia o corpus inicial (congelado)")
    p.add_argument("--force", action="store_true", help="reimporta, substituindo a cópia existente")
    p.set_defaults(handler=_cmd_import_legacy)

    p = sub.add_parser("build", parents=[common], help="constrói splits, artefatos e relatório")
    p.add_argument("--output", type=Path, default=None, help="diretório de saída (padrão: paths.build)")
    p.set_defaults(handler=_cmd_build)

    p = sub.add_parser("freeze", parents=[common], help="congela o último build para o treino")
    p.add_argument("--force", action="store_true", help="substitui um corpus já congelado")
    p.set_defaults(handler=_cmd_freeze)

    p = sub.add_parser("annotate", parents=[common], help="anotação humana cega (no seu terminal)")
    p.add_argument("--annotator", required=True, help="nome curto do anotador (ex.: phelipe)")
    p.add_argument("--n", type=int, default=300, help="tamanho da amostra (fixado na primeira execução)")
    p.set_defaults(handler=_cmd_annotate)
    p = sub.add_parser("agreement", parents=[common], help="kappa de Cohen entre anotador e rótulos do corpus")
    p.add_argument("--annotator", required=True)
    p.set_defaults(handler=_cmd_agreement)
    p = sub.add_parser("autoverificacao", parents=[common], help="taxa da autoverificação do rótulo (logs da geração)")
    p.set_defaults(handler=_cmd_self_check)

    from ddsd.corpus.generation.providers import PROVIDERS

    generate = subparsers.add_parser("generate", help="geração da mitigação de comprimento com LLMs")
    gsub = generate.add_subparsers(dest="generate_command", required=True, metavar="<ação>")
    p = gsub.add_parser("plan", parents=[common], help="mostra o progresso da mitigação por fonte e faixa")
    p.set_defaults(handler=_cmd_generate_plan)
    p = gsub.add_parser("run", parents=[common], help="gera frases com um provedor (chaves no .env)")
    p.add_argument("provider", choices=sorted(PROVIDERS), help="provedor HTTP")
    p.add_argument("--per-cell", type=int, default=250,
                   help="meta de frases por (classe, faixa-alvo); são 8 faixas × 2 classes (padrão: 250)")
    p.add_argument("--source", default=None, help="nome da fonte (padrão: o do provedor)")
    p.add_argument("--sleep", type=float, default=None, help="pausa entre chamadas em segundos")
    p.add_argument("--batch-size", type=int, default=None, help="frases pedidas por chamada (padrão do TOML)")
    p.add_argument("--strategy", choices=("comprimento", "negativos_dificeis"), default="comprimento",
                   help="comprimento: pares curtos e longos; negativos_dificeis: pedidos a pessoas × software (2-6 palavras)")
    p.set_defaults(handler=_cmd_generate_run)
    p = gsub.add_parser("ingest", parents=[common], help="valida e anexa frases geradas fora dos provedores")
    p.add_argument("--source", required=True, help="nome da fonte (ex.: claude_subagentes)")
    p.add_argument("--label", required=True, choices=("DD", "NDD"))
    p.add_argument("--model", required=True, help="modelo que gerou as frases")
    p.add_argument("--file", type=Path, required=True, help="JSONL com {'text': ...} ou arrays JSON por linha")
    p.add_argument("--strategy", choices=("comprimento", "negativos_dificeis"), default="comprimento",
                   help="estratégia de geração registrada na proveniência (padrão: comprimento)")
    p.set_defaults(handler=_cmd_generate_ingest)

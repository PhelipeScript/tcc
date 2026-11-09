"""Subcomandos do tempo real: `devices`, `download-models`, `run` (microfone) e `replay` (arquivos)."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import threading
import time
from pathlib import Path

from ddsd.observability.logger import BufferSink, ConsoleSink, configure_logging, get_logger
from ddsd.realtime.config import PIPELINE_CHOICES

log = get_logger("cli")


def _quiet_libraries() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")


def _load_config(args: argparse.Namespace):
    from ddsd.realtime.config import load_pipeline_config

    overrides = {"asr__backend": args.backend, "logging__level": "debug" if args.debug else None,
                 "classifiers__active": getattr(args, "classificador", None)}
    return load_pipeline_config(args.config, **overrides)


def _cmd_devices(args: argparse.Namespace) -> int:
    configure_logging(level="info")
    from ddsd.audio.capture import list_input_devices

    for d in list_input_devices():
        mark = "*" if d.is_default else " "
        print(f"{mark} [{d.index}] {d.name} - {d.samplerate} Hz, {d.channels} canal(is)")
    print("\n* = entrada padrão do sistema. Use --device com o índice ou parte do nome.")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    configure_logging(level="info")
    from huggingface_hub import snapshot_download

    from ddsd.classifiers.hub import download
    from ddsd.realtime.config import load_pipeline_config

    cfg = load_pipeline_config(args.config)
    for repo in (cfg.asr.mlx_model, cfg.asr.faster_whisper_model):
        started = time.perf_counter()
        path = snapshot_download(repo)
        log.info("%s em cache (%s) em %.0f s", repo, path, time.perf_counter() - started)
    from silero_vad import load_silero_vad

    load_silero_vad(onnx=True)
    log.info("Silero VAD disponível")
    for repo in cfg.hub.repos:  # classificadores publicados no Hugging Face
        download(repo)
    return 0


def _run_with_dashboard(pipeline, cfg, stop: threading.Event, sink: BufferSink | None) -> None:
    """Painel em tela cheia, sem rolagem: o quadro inteiro é refeito no tamanho atual do terminal."""
    from rich.console import Console
    from rich.live import Live

    from ddsd.observability.dashboard import Dashboard

    dashboard = Dashboard(pipeline.status, sink)
    with Live(get_renderable=dashboard.render, refresh_per_second=cfg.dashboard.refresh_per_second,
              screen=True, console=Console(), redirect_stdout=False, redirect_stderr=False):
        while not stop.wait(0.2):
            pass


def _make_recorder(args: argparse.Namespace, cfg, command: str):
    """`--gravar [DIR]`: abre a sessão (padrão `data/sessoes/<data-hora>/`) com os metadados.

    Devolve um gerenciador de contexto: o `SessionRecorder` ou, sem a flag, um contexto vazio (`None`).
    Levanta `FileExistsError` se o diretório já tiver uma sessão gravada.
    """
    if getattr(args, "gravar", None) is None:
        return contextlib.nullcontext()
    import dataclasses

    from ddsd.common.paths import DATA_DIR
    from ddsd.common.system import code_version
    from ddsd.realtime.session import SessionRecorder

    out_dir = Path(args.gravar) if args.gravar else DATA_DIR / "sessoes" / time.strftime("%Y%m%d-%H%M%S")
    metadata = {"comando": command, "dispositivo": getattr(args, "device", None),
                "arquivos": getattr(args, "files", None), "config": dataclasses.asdict(cfg),
                "codigo": code_version()}
    return SessionRecorder(out_dir, metadata)


def _cmd_run(args: argparse.Namespace) -> int:
    _quiet_libraries()
    cfg = _load_config(args)
    from ddsd.observability.dashboard import classifier_label

    dashboard = cfg.dashboard.enabled and not args.no_dashboard
    sink: BufferSink | None = BufferSink(cfg.dashboard.log_lines) if dashboard else None
    # O carregamento dos modelos aparece em linhas normais; o painel assume depois.
    configure_logging(level=cfg.logging.level, sink=ConsoleSink())
    from ddsd.realtime.pipeline import Pipeline, PipelineError

    try:
        session = _make_recorder(args, cfg, "run")
    except FileExistsError as exc:
        log.error("%s", exc)
        return 1
    with session as recorder:  # a gravação é fechada (com resumo) em qualquer saída
        pipeline = Pipeline(cfg, recorder=recorder)
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        log.info("carregando Whisper e o classificador %s (%s)...", classifier_label(cfg.classifiers.active),
                 cfg.classifiers.active_run.name)
        try:
            pipeline.start(input_device=args.device)
        except PipelineError as exc:
            log.error("%s", exc)
            return 1
        from ddsd.realtime.keys import LEGEND, KeyReader, handle_key

        def on_key(key: str) -> None:
            if handle_key(pipeline, key) == "sair":
                stop.set()

        keys = KeyReader(on_key)
        try:
            if keys.start():
                log.info("teclas: %s", LEGEND)
            if dashboard:
                configure_logging(level=cfg.logging.level, sink=sink)
                _run_with_dashboard(pipeline, cfg, stop, sink)
                configure_logging(level=cfg.logging.level, sink=ConsoleSink())
            else:
                while not stop.wait(0.2):
                    pass
        finally:
            keys.stop()  # devolve o terminal ao modo normal antes de imprimir o resumo
            pipeline.stop()
            _print_summary(pipeline)
    return 0


def _print_summary(pipeline) -> None:
    from rich.console import Console

    s = pipeline.status.snapshot()
    console = Console()
    c = s.counters
    console.print(f"[bold]Resumo[/bold]: {c.get('falas', 0)} falas · DD {c.get('decisoes.DD', 0)} · "
                  f"NDD {c.get('decisoes.NDD', 0)}")
    for name, (n, p50, p95, _) in sorted(s.latencies.items()):
        console.print(f"  {name:<16} n={n:<5} p50={1000 * p50:7.0f} ms  p95={1000 * p95:7.0f} ms")


def _cmd_replay(args: argparse.Namespace) -> int:
    """Roda o pipeline completo sobre gravações (mesmo caminho do microfone, só a fonte muda)."""
    _quiet_libraries()
    cfg = _load_config(args)
    configure_logging(level=cfg.logging.level, sink=ConsoleSink())
    from ddsd.common.io import write_jsonl
    from ddsd.realtime.pipeline import Pipeline, load_audio, utterances_for_file

    files = [Path(p) for p in args.files]
    signals = [load_audio(p) for p in files]
    rates = {rate for _, rate in signals}
    if len(rates) != 1:
        log.error("todos os arquivos precisam ter a mesma taxa; encontradas %s", sorted(rates))
        return 1
    try:
        session = _make_recorder(args, cfg, "replay")
    except FileExistsError as exc:
        log.error("%s", exc)
        return 1
    rows = []
    with session as recorder:  # a gravação é fechada (com resumo) em qualquer saída
        pipeline = Pipeline(cfg, recorder=recorder)
        pipeline.start(file_rate=rates.pop())
        try:
            pipeline.play(signals[0][0][:0], trailing_silence_s=0.5)  # aquece o frontend com silêncio
            for path, (audio, _) in zip(files, signals, strict=True):
                start16, end16 = pipeline.play(audio, trailing_silence_s=args.silence)
                pipeline.wait_idle(timeout_s=60)
                decided = utterances_for_file(pipeline.decisions, start16, end16)
                rows.append({"file": str(path), "decisions": [d.to_dict() for d in decided]})
                labels = ", ".join(f"{d.utterance_label}={d.label}" for d in decided) or "nenhuma fala detectada"
                log.info("%s → %s", path.name, labels)
        finally:
            pipeline.stop()
            _print_summary(pipeline)
    if args.output:
        write_jsonl(Path(args.output), rows)
        log.info("decisões gravadas em %s", args.output)
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    configure_logging(level="warn")
    from ddsd.corpus.config import load_corpus_config
    from ddsd.evaluation.recording import run_session

    cfg = load_corpus_config()
    test_split = cfg.paths.frozen / "splits" / "test.jsonl"
    if not test_split.exists():
        log.error("corpus ainda não congelado (%s)", test_split)
        return 1
    return run_session(test_split, args.scenario)


def _add_record_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gravar", nargs="?", const="", default=None, metavar="DIR",
                        help="grava o áudio de cada fala e as evidências da decisão (grava voz; "
                             "padrão: data/sessoes/<data-hora>/)")


def register(subparsers: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="TOML do pipeline (padrão: config/pipeline.toml)")
    common.add_argument("--backend", choices=("mlx", "faster-whisper"), default=None)
    common.add_argument("--classificador", choices=PIPELINE_CHOICES, default=None,
                        help="classificador do pipeline (padrão: [classifiers] active, BERTimbau)")
    common.add_argument("--debug", action="store_true")

    p = subparsers.add_parser("devices", help="lista os dispositivos de entrada de áudio")
    p.set_defaults(handler=_cmd_devices)
    p = subparsers.add_parser("download-models", parents=[common],
                              help="baixa Whisper, Silero e os classificadores do Hugging Face")
    p.set_defaults(handler=_cmd_download)
    p = subparsers.add_parser("run", parents=[common], help="pipeline em tempo real com o microfone")
    p.add_argument("--device", default=None, help="índice ou parte do nome do dispositivo de entrada")
    p.add_argument("--no-dashboard", action="store_true", help="logs em linhas em vez do painel")
    _add_record_flag(p)
    p.set_defaults(handler=_cmd_run)
    p = subparsers.add_parser("record-eval-set", help="gravação guiada do conjunto de avaliação (no seu terminal)")
    p.add_argument("--scenario", nargs="*", default=None, help="cenários a gravar (padrão: todos)")
    p.set_defaults(handler=_cmd_record)
    p = subparsers.add_parser("replay", parents=[common], help="pipeline completo sobre gravações (teste/avaliação)")
    p.add_argument("files", nargs="+", help="arquivos de áudio (mesma taxa)")
    p.add_argument("--silence", type=float, default=1.5, help="silêncio após cada arquivo (s)")
    p.add_argument("--output", default=None, help="JSONL com as decisões por arquivo")
    _add_record_flag(p)
    p.set_defaults(handler=_cmd_replay)

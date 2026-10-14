"""
Descrição:
    Curvas de loss (treino vs. validação) por época, extraídas do
    `trainer_state.json` de cada modelo — o dado que mostra o overfitting de
    calibração identificado: o `eval_loss` atinge o mínimo na época 2-3 e
    volta a subir, enquanto o `train_loss` cai para perto de zero.

    Lê sempre o checkpoint de **maior step** (o último salvo), não o "melhor"
    (`best_model_checkpoint`): o melhor checkpoint é uma cópia parcial do
    treino, mas o `trainer_state.json` do checkpoint final é o único que
    contém o `log_history` completo até a última época.

Uso:
    from src.analysis.loss_curves import load_log_history, plot_loss_curve
    history = load_log_history(run_dir)
    plot_loss_curve("BERTimbau", history, output_path)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

#: Casa o nome de um diretório de checkpoint do `Trainer`, ex. `checkpoint-23748`.
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(slots=True)
class LogHistory:
    """
    Descrição:
        Histórico de treino de uma execução, separado em série de treino
        (por step) e série de validação (por época).

    Atributos:
        train_epoch: Época de cada log de treino (fracionária).
        train_loss: Loss de treino correspondente.
        eval_epoch: Época de cada avaliação (inteira).
        eval_loss: Loss de validação por época.
        eval_f1_macro: F1 macro de validação por época, para contrastar com
            o `eval_loss` no mesmo gráfico.
    """

    train_epoch: list[float]
    train_loss: list[float]
    eval_epoch: list[float]
    eval_loss: list[float]
    eval_f1_macro: list[float]


def find_latest_checkpoint(run_dir: Path) -> Path:
    """
    Descrição:
        Encontra o checkpoint de maior `step` dentro de `run_dir/checkpoints`.

    Args:
        run_dir: Diretório da execução (ex.: `outputs/bertimbau`).

    Retorna:
        Caminho do checkpoint com o maior número de step.

    Levanta:
        FileNotFoundError: Se não houver nenhum diretório `checkpoint-<n>`.
    """
    checkpoints_dir = run_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoints_dir.exists():
        for entry in checkpoints_dir.iterdir():
            match = _CHECKPOINT_RE.match(entry.name)
            if entry.is_dir() and match:
                candidates.append((int(match.group(1)), entry))

    if not candidates:
        raise FileNotFoundError(f"nenhum checkpoint encontrado em {checkpoints_dir}")

    return max(candidates, key=lambda pair: pair[0])[1]


def load_log_history(run_dir: Path) -> LogHistory:
    """
    Descrição:
        Carrega e separa o `log_history` do checkpoint mais recente de uma
        execução.

    Args:
        run_dir: Diretório da execução (ex.: `outputs/bertimbau`).

    Retorna:
        `LogHistory` com as séries de treino e validação.
    """
    checkpoint = find_latest_checkpoint(run_dir)
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))

    train_epoch: list[float] = []
    train_loss: list[float] = []
    eval_epoch: list[float] = []
    eval_loss: list[float] = []
    eval_f1_macro: list[float] = []

    for entry in state["log_history"]:
        if "eval_loss" in entry:
            eval_epoch.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])
            eval_f1_macro.append(entry["eval_f1_macro"])
        elif "loss" in entry:
            train_epoch.append(entry["epoch"])
            train_loss.append(entry["loss"])

    return LogHistory(
        train_epoch=train_epoch,
        train_loss=train_loss,
        eval_epoch=eval_epoch,
        eval_loss=eval_loss,
        eval_f1_macro=eval_f1_macro,
    )


def plot_loss_curve(display_name: str, history: LogHistory, output_path: Path) -> Path:
    """
    Descrição:
        Plota train_loss (por step, em época fracionária) contra eval_loss e
        eval_f1_macro (por época), em eixos y separados. É o gráfico que
        ilustra o overfitting: train_loss caindo a quase zero enquanto
        eval_loss sobe e f1_macro se mantém estável.

    Args:
        display_name: Nome do modelo, para o título.
        history: Histórico já carregado por `load_log_history`.
        output_path: Caminho do PNG a gravar.

    Retorna:
        `output_path`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, loss_axis = plt.subplots(figsize=(6.4, 4.2))
    loss_axis.plot(
        history.train_epoch, history.train_loss, color="tab:blue", alpha=0.5, label="train_loss"
    )
    loss_axis.plot(
        history.eval_epoch,
        history.eval_loss,
        color="tab:red",
        marker="o",
        label="eval_loss",
    )
    best_epoch_index = min(range(len(history.eval_loss)), key=history.eval_loss.__getitem__)
    loss_axis.axvline(
        history.eval_epoch[best_epoch_index], color="gray", linestyle="--", linewidth=0.8
    )
    loss_axis.set_xlabel("época")
    loss_axis.set_ylabel("loss")
    loss_axis.legend(loc="upper left")

    metric_axis = loss_axis.twinx()
    metric_axis.plot(
        history.eval_epoch,
        history.eval_f1_macro,
        color="tab:green",
        marker="s",
        linestyle=":",
        label="eval_f1_macro",
    )
    metric_axis.set_ylabel("f1_macro (validação)")
    metric_axis.legend(loc="upper right")

    loss_axis.set_title(f"Curva de loss por época — {display_name}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_overlay(histories: dict[str, LogHistory], output_path: Path) -> Path:
    """
    Descrição:
        Sobrepõe o `eval_loss` por época dos modelos informados, para mostrar
        que o padrão de overfitting (mínimo na época 2-3, depois sobe) se
        repete de forma consistente entre arquiteturas independentes.

    Args:
        histories: Mapeamento `nome de exibição -> LogHistory`.
        output_path: Caminho do PNG a gravar.

    Retorna:
        `output_path`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for name, history in histories.items():
        axis.plot(history.eval_epoch, history.eval_loss, marker="o", label=name)
    axis.set_xlabel("época")
    axis.set_ylabel("eval_loss")
    axis.set_title("Eval loss por época — comparação entre modelos")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path

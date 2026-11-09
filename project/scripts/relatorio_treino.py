"""Resumo do treino dos classificadores (`uv run python scripts/relatorio_treino.py`).

Lê `models/` (fora do git): o `run_manifest.json`, o `metrics_val.json` e o histórico do Trainer
(`trainer_state.json` do último checkpoint) de cada execução. Grava `reports/treino/treino.json` e `treino.md`:
hiperparâmetros, F1 macro e perda de validação por época, época do melhor checkpoint, F1 macro de validação do
modelo final (limiar 0,5) e o limiar escolhido na validação.

Tempo, memória e precisão por modelo ficam em `reports/custo/`; as métricas do teste, em `reports/experimento/`.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "treino"
TRANSFORMERS = {"bertimbau": "BERTimbau", "albertina": "Albertina", "debertinha": "DeBERTinha"}
TFIDF = {"baseline_logreg": "TF-IDF + Regressão Logística", "baseline_svm": "TF-IDF + SVM"}
SEEDS = (42, 43, 44)
CONDITIONS = {"principal": "com correção", "controle": "sem correção"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def decimal(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def trainer_history(run_dir: Path) -> list[dict]:
    """`log_history` do último checkpoint, que guarda o histórico inteiro do treino."""
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return read_json(checkpoints[-1] / "trainer_state.json")["log_history"]


def transformer_run(run_dir: Path, family: str, condition: str) -> dict:
    manifest = read_json(run_dir / "run_manifest.json")
    model, training, result = manifest["model"], manifest["training"], manifest["result"]
    steps_per_epoch = manifest["timing"]["steps_per_epoch"]
    best_step = int(PureWindowsPath(result["best_checkpoint"]).name.split("-")[1])
    history = trainer_history(run_dir)
    epochs = []
    for entry in (e for e in history if "eval_loss" in e):
        epoch = round(entry["epoch"])
        train_losses = [e["loss"] for e in history if "loss" in e and epoch - 1 < e["epoch"] <= epoch]
        epochs.append({"epoca": epoch, "f1_macro_validacao": entry["eval_f1_macro"],
                       "perda_validacao": entry["eval_loss"], "perda_treino_media": float(np.mean(train_losses))})
    at_best = next(e for e in epochs if e["epoca"] == round(best_step / steps_per_epoch))
    lowest = min(epochs, key=lambda e: e["perda_validacao"])
    return {
        "execucao": run_dir.name if condition == "principal" else f"controle/{run_dir.name}",
        "modelo": family, "condicao": condition, "semente": training["seed"],
        "hiperparametros": {"checkpoint_base": model["checkpoint"], "taxa_aprendizado": model["learning_rate"],
                            "lote": model["batch_size"] * model["grad_accum_steps"], "epocas": model["epochs"],
                            "max_tokens": model["max_length"], "aquecimento": model["warmup_ratio"],
                            "dropout": model["dropout"], "weight_decay": training["weight_decay"],
                            "paciencia": training["early_stopping_patience"],
                            "metrica_do_checkpoint": training["metric_for_best_model"],
                            "agendador": training["lr_scheduler_type"], "precisao": training["precision_effective"]},
        "passos_por_epoca": steps_per_epoch,
        "melhor_checkpoint": {"passo": best_step, "epoca": best_step / steps_per_epoch},
        "por_epoca": epochs,
        "epoca_menor_perda_validacao": lowest["epoca"],
        # a menor perda de validação vem antes do melhor checkpoint, e a F1 de validação ainda melhora depois dela
        "menor_perda_antes_do_melhor": (lowest["epoca"] < at_best["epoca"]
                                        and at_best["perda_validacao"] > lowest["perda_validacao"]
                                        and at_best["f1_macro_validacao"] > lowest["f1_macro_validacao"]),
        "f1_macro_validacao_limiar_05": read_json(run_dir / "metrics_val.json")["f1_macro"],
        "limiar_escolhido": result["decision_threshold"],
        "no_sistema": False,  # marcado em `build`
    }


def tfidf_run(run_dir: Path) -> dict:
    result = read_json(run_dir / "run_manifest.json")["result"]
    return {"execucao": run_dir.name, "c": result["best_c"], "f1_macro_validacao_cruzada": result["cv_f1_macro"],
            "f1_macro_validacao": result["val_metrics"]["f1_macro"], "limiar_escolhido": result["decision_threshold"],
            "vocabulario": result["vocabulary"]}


def build() -> dict:
    runs = [transformer_run(MODELS / f"{family}-s{seed}", family, "principal")
            for family in TRANSFORMERS for seed in SEEDS]
    runs += [transformer_run(MODELS / "controle" / f"{family}-s42", family, "controle") for family in TRANSFORMERS]
    for family in TRANSFORMERS:  # a execução do sistema segue a regra: maior F1 de validação com limiar 0,5
        seeds = [r for r in runs if r["modelo"] == family and r["condicao"] == "principal"]
        best = max(seeds, key=lambda r: r["f1_macro_validacao_limiar_05"])
        best["no_sistema"] = True
        published = MODELS / f"ddsd-ptbr-{family}" / "treinamento.json"  # o modelo do sistema, baixado do HF
        if published.exists() and read_json(published)["semente"] != best["semente"]:
            raise SystemExit(f"{family}: o modelo publicado não é a execução de maior F1 de validação "
                             f"({best['execucao']})")
        hyper = [json.dumps(r["hiperparametros"], sort_keys=True) for r in runs if r["modelo"] == family]
        if len(set(hyper)) != 1:
            raise SystemExit(f"{family}: hiperparâmetros diferentes entre as execuções")
    return {"transformers": runs, "tfidf": [tfidf_run(MODELS / name) for name in TFIDF]}


def markdown(report: dict) -> str:
    runs = report["transformers"]
    first = {family: next(r for r in runs if r["modelo"] == family)["hiperparametros"] for family in TRANSFORMERS}
    rows = [("Checkpoint base", lambda h: f"`{h['checkpoint_base']}`"),
            ("Taxa de aprendizado", lambda h: f"{h['taxa_aprendizado']:.0e}".replace("e-0", "e-")),
            ("Lote", lambda h: str(h["lote"])), ("Épocas (máximo)", lambda h: str(h["epocas"])),
            ("Tokens por frase (máximo)", lambda h: str(h["max_tokens"])),
            ("Aquecimento", lambda h: f"{100 * h['aquecimento']:.0f}% dos passos"),
            ("Agendador", lambda h: h["agendador"]), ("Dropout", lambda h: decimal(h["dropout"], 1)),
            ("Weight decay", lambda h: decimal(h["weight_decay"], 2)),
            ("Parada antecipada", lambda h: f"paciência {h['paciencia']}, {h['metrica_do_checkpoint']}"),
            ("Precisão", lambda h: h["precisao"])]
    lines = ["# Treino dos classificadores", "",
             "Gerado por `scripts/relatorio_treino.py` a "
             "partir de `models/` (fora do git). O mesmo treino vale para as 3 sementes e para o modelo sem "
             "correção de cada família. Tempo, memória e precisão: `reports/custo/`; teste: `reports/experimento/`.",
             "", "## Hiperparâmetros", "", "| Parâmetro | " + " | ".join(TRANSFORMERS.values()) + " |",
             "|---|---|---|---|"]
    lines += [f"| {name} | " + " | ".join(fmt(first[f]) for f in TRANSFORMERS) + " |" for name, fmt in rows]
    lines += ["", "## Por execução", "",
              "| Execução | Corpus | Melhor checkpoint | F1 macro de validação (limiar 0,5) | Limiar escolhido | "
              "No sistema |", "|---|---|---|---|---|---|"]
    for r in runs:
        best = r["melhor_checkpoint"]
        lines.append(f"| {r['execucao']} | {CONDITIONS[r['condicao']]} | época {best['epoca']:.0f} de "
                     f"{r['hiperparametros']['epocas']} | {decimal(r['f1_macro_validacao_limiar_05'], 4)} | "
                     f"{decimal(r['limiar_escolhido'], 3)} | {'sim' if r['no_sistema'] else ''} |")
    max_epochs = max(len(r["por_epoca"]) for r in runs)
    lines += ["", "## F1 macro de validação por época, durante o treino", "",
              "| Execução | " + " | ".join(f"Época {k}" for k in range(1, max_epochs + 1)) + " |",
              "|---|" + "---|" * max_epochs]
    for r in runs:
        cells = [decimal(e["f1_macro_validacao"], 4) for e in r["por_epoca"]]
        lines.append(f"| {r['execucao']} | " + " | ".join(cells + [""] * (max_epochs - len(cells))) + " |")
    lines += ["", "## Perda de validação por época, durante o treino", "",
              "| Execução | " + " | ".join(f"Época {k}" for k in range(1, max_epochs + 1)) + " |",
              "|---|" + "---|" * max_epochs]
    for r in runs:
        cells = [decimal(e["perda_validacao"], 4) for e in r["por_epoca"]]
        lines.append(f"| {r['execucao']} | " + " | ".join(cells + [""] * (max_epochs - len(cells))) + " |")
    lines += ["", "## TF-IDF", "",
              "| Modelo | C (validação cruzada de 5 partes) | F1 macro na validação cruzada | F1 macro de validação | "
              "Limiar escolhido |", "|---|---|---|---|---|"]
    for t in report["tfidf"]:
        lines.append(f"| {TFIDF[t['execucao']]} | {decimal(t['c'], 1)} | {decimal(t['f1_macro_validacao_cruzada'], 4)} | "
                     f"{decimal(t['f1_macro_validacao'], 4)} | {decimal(t['limiar_escolhido'], 3)} |")
    earlier = sum(r["menor_perda_antes_do_melhor"] for r in runs)
    lines += ["", "Notas:",
              f"- Em {earlier} de {len(runs)} execuções, a menor perda de validação vem numa época anterior à do "
              "melhor checkpoint: depois dela, a perda de validação sobe, mas a F1 de validação ainda melhora. O "
              "checkpoint é escolhido pela F1.",
              "- A validação dos modelos sem correção é a do corpus deles, que ainda tem o atalho do tamanho.",
              "- O melhor checkpoint é o de maior F1 macro de validação; o limiar maximiza a mesma F1 na validação.",
              "- \"No sistema\": a semente de maior F1 macro de validação (limiar 0,5), usada no sistema em tempo real.",
              ""]
    return "\n".join(lines)


def main() -> None:
    report = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "treino.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "treino.md").write_text(markdown(report), encoding="utf-8")
    print(f"treino em {OUT / 'treino.md'}")


if __name__ == "__main__":
    main()

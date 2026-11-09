"""Comparação dos classificadores no sistema completo (`ddsd eval comparar`).

Os cinco modelos com correção do viés de comprimento rodam o sistema completo em tempo real, cada um na
sua execução (`ddsd eval pipeline <conjunto> --classificador X`). Com a semente fixa do Whisper, as cinco
execuções recebem o mesmo áudio e as mesmas transcrições, o que `same_transcriptions` confere.

Os três Transformers sem correção não rodam ao vivo: eles julgam as transcrições da execução do BERTimbau,
pela mesma regra do decisor (`TaskEvidence.label_by`).
- O resultado equivale a rodar o sistema com aquele classificador, porque as transcrições são as mesmas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ddsd.classifiers.model import Prediction
from ddsd.decision.decider import TaskEvidence
from ddsd.evaluation.external_text import paired_fp_bootstrap
from ddsd.realtime.domain import AudioSpan, TaskId

#: Modelos sem correção do viés de comprimento (treinados no corpus de controle, `models/controle/`).
CONTROL_PREFIX = "sem_correcao_"
#: Ordem do TCC: o principal, os baselines e os outros dois Transformers, todos ao vivo.
LIVE_ORDER = ("bertimbau", "baseline_logreg", "baseline_svm", "debertinha", "albertina")
#: Os mesmos três Transformers sem a correção, julgando as transcrições da execução do BERTimbau.
CONTROL_ORDER = ("sem_correcao_bertimbau", "sem_correcao_debertinha", "sem_correcao_albertina")
CLASSIFIER_ORDER = LIVE_ORDER + CONTROL_ORDER
DISPLAY = {"bertimbau": "BERTimbau", "baseline_logreg": "TF-IDF + Regressão Logística",
           "baseline_svm": "TF-IDF + SVM", "debertinha": "DeBERTinha", "albertina": "Albertina",
           "sem_correcao_bertimbau": "BERTimbau sem correção", "sem_correcao_debertinha": "DeBERTinha sem correção",
           "sem_correcao_albertina": "Albertina sem correção"}
#: Execução cujas transcrições os modelos sem correção julgam (e referência das comparações).
REFERENCE = "bertimbau"


class ParityError(RuntimeError):
    """A reclassificação com o classificador da execução não reproduziu as decisões dela."""


class TextClassifier(Protocol):
    def predict(self, normalized_text: str) -> Prediction: ...


def control_run(model: str, models_dir: Path) -> Path:
    """Execução sem correção do modelo (`models/controle/`): a semente escolhida pela validação."""
    from ddsd.evaluation.runs import best_seed, discover

    run = best_seed(discover(models_dir), "controle", model)
    if run is None:
        raise FileNotFoundError(f"nenhuma execução de controle de {model} em {models_dir / 'controle'}")
    return run.path


def load_classifier(name: str, runs: dict[str, Path], models_dir: Path, device: str = "cpu") -> TextClassifier:
    """Transformer pela execução do pipeline (`[classifiers]`), sem correção ou baseline TF-IDF de `models/`."""
    if name.startswith("baseline_"):
        from ddsd.evaluation.baselines import BaselineClassifier

        return BaselineClassifier(models_dir / name)
    from ddsd.classifiers.model import DddClassifier

    if name.startswith(CONTROL_PREFIX):
        model = name.removeprefix(CONTROL_PREFIX)
        return DddClassifier(model, control_run(model, models_dir), device=device)
    return DddClassifier(name, runs[name], device=device)


def _final_task(decision: dict[str, Any]) -> dict[str, Any]:
    """A tarefa da fala completa (`U0007-F`)."""
    return next(t for t in decision["tasks"] if t["task"].endswith("-F"))


def reclassify(records: list[dict[str, Any]], name: str, classifier: TextClassifier
               ) -> tuple[list[dict[str, Any]], list[float]]:
    """Registros no formato de `pipeline_eval`, com a decisão de `name` em cada fala.

    Devolve também o tempo de inferência (ms) de cada transcrição classificada.
    """
    out: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    for record in records:
        decisions = []
        for d in record["decisions"]:
            final = _final_task(d)
            evidence = TaskEvidence(TaskId(int(d["utterance"][1:])), AudioSpan(0, 0))
            evidence.asr_error, evidence.normalized = final["asr_error"], final["normalized"]
            if evidence.asr_error is None and evidence.normalized:
                prediction = classifier.predict(evidence.normalized)
                evidence.predictions[name] = prediction  # indexada pelo nome pedido (ex.: sem_correcao_bertimbau)
                inference_ms.append(prediction.latency_ms)
            label, reason = evidence.label_by(name)
            prediction = evidence.predictions.get(name)
            decisions.append({
                "utterance": d["utterance"], "label": label, "reason": reason, "classifier": name,
                "p_dd": None if prediction is None else round(prediction.p_dd, 6),
                "threshold": None if prediction is None else prediction.threshold,
                "reliability": d["reliability"], "text_raw": d["text_raw"], "normalized": d["normalized"],
                "span_s": d["span_s"], "close_reason": d["close_reason"],
                "decisao_execucao": {"classifier": d["classifier"], "label": d["label"]},
            })
        out.append({
            "id": record["id"], "group": record["group"], "label": record["label"],
            "reference": record["reference"], "classifier": name, "audio_s": record["audio_s"],
            "utterances": len(decisions), "activated": any(x["label"] == "DD" for x in decisions),
            "activated_low_policy": any(x["label"] == "DD" and not x["reliability"]["alert"] for x in decisions),
            "decisions": decisions,
            # A latência ponta a ponta é a da execução em tempo real (com o classificador dela): não se
            # atribui a outro classificador. O tempo de inferência de cada um vai à parte.
            "latencies_s": [],
        })
    return out, inference_ms


def _utterances(record: dict[str, Any]) -> list[tuple[tuple[float, float], str]]:
    return [(tuple(d["span_s"]), _final_task(d)["text_raw"]) for d in record["decisions"]]


def same_transcriptions(reference: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    """Confere, item a item, se duas execuções tiveram os mesmos trechos de fala e as mesmas transcrições."""
    if [r["id"] for r in reference] != [r["id"] for r in other]:
        raise ValueError("as execuções não têm os mesmos itens, na mesma ordem")
    segmentation = text = 0
    different: list[str] = []
    for a, b in zip(reference, other, strict=True):
        ua, ub = _utterances(a), _utterances(b)
        segmentation += [s for s, _ in ua] == [s for s, _ in ub]
        if ua == ub:
            text += 1
        else:
            different.append(a["id"])
    return {"itens": len(reference), "segmentacao_identica": segmentation, "transcricao_identica": text,
            "itens_diferentes": different}


def inference_ms(records: list[dict[str, Any]], name: str) -> list[float]:
    """Tempo de inferência do classificador em cada fala de uma execução em tempo real."""
    return [p["latency_ms"] for r in records for d in r["decisions"] for t in d["tasks"]
            for n, p in (t.get("predictions") or {}).items() if n == name and p.get("latency_ms") is not None]


def check_parity(original: list[dict[str, Any]], reclassified: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Compara rótulo e p(DD) fala a fala; `ParityError` se algum rótulo diferir."""
    total = mismatched = 0
    max_delta = 0.0
    for a, b in zip(original, reclassified, strict=True):
        if a["id"] != b["id"] or len(a["decisions"]) != len(b["decisions"]):
            raise ParityError(f"registros desalinhados em {a['id']} / {b['id']}")
        for da, db in zip(a["decisions"], b["decisions"], strict=True):
            total += 1
            mismatched += da["label"] != db["label"]
            vote = da.get("votes", {}).get(name)
            if vote is not None and db["p_dd"] is not None:
                max_delta = max(max_delta, abs(vote["p_dd"] - db["p_dd"]))
    result = {"falas": total, "identicas": total - mismatched, "max_delta_p": max_delta}
    if mismatched:
        raise ParityError(f"{name}: {mismatched} de {total} decisões diferentes da execução em tempo real")
    return result


def _rate(records: list[dict[str, Any]], label: str, key: str = "activated") -> float | None:
    sel = [r[key] for r in records if r["label"] == label]
    return float(np.mean(sel)) if sel else None


def default_pairs(names: list[str], reference: str = REFERENCE) -> list[tuple[str, str]]:
    """Cada classificador contra o de referência e cada Transformer contra a própria versão sem correção."""
    pairs = [(n, reference) for n in names if n != reference and not n.startswith(CONTROL_PREFIX)]
    pairs += [(n, n.removeprefix(CONTROL_PREFIX)) for n in names
              if n.startswith(CONTROL_PREFIX) and n.removeprefix(CONTROL_PREFIX) in names]
    return pairs


def compare(by_classifier: dict[str, list[dict[str, Any]]], reference: str = REFERENCE,
            n_bootstrap: int = 2000, pairs: list[tuple[str, str]] | None = None,
            group_field: str | None = None) -> dict[str, Any]:
    """Taxas por classificador × grupo e diferenças pareadas (todos os grupos juntos).

    Em DD, a taxa de ativação é o acerto; em NDD e na fala livre, é o falso positivo. `pairs` (A, B) dá a
    diferença A - B; o padrão é cada um contra o de referência e cada modelo sem correção contra o com. Com
    `group_field`, o bootstrap sorteia as frases (as leituras com o mesmo valor desse campo entram juntas).
    """
    groups = sorted({r["group"] for recs in by_classifier.values() for r in recs})
    labels = [lab for lab in ("DD", "NDD", "NDD_LIVRE", "RUIDO")
              if any(r["label"] == lab for recs in by_classifier.values() for r in recs)]
    out: dict[str, Any] = {"por_grupo": {}, "pareado": {}}
    for name, recs in by_classifier.items():
        out["por_grupo"][name] = {
            g: {lab: {"n": sum(r["label"] == lab for r in recs if r["group"] == g),
                      "ativacao": _rate([r for r in recs if r["group"] == g], lab),
                      "ativacao_politica_baixa_ndd": _rate([r for r in recs if r["group"] == g], lab,
                                                           "activated_low_policy")}
                for lab in labels if any(r["label"] == lab and r["group"] == g for r in recs)}
            for g in groups}
    for name_a, name_b in pairs if pairs is not None else default_pairs(list(by_classifier), reference):
        key = f"{name_a} vs {name_b}"
        out["pareado"][key] = {}
        for lab in ("DD", "NDD"):
            a = [r for r in by_classifier[name_a] if r["label"] == lab]
            b = [r for r in by_classifier[name_b] if r["label"] == lab]
            if not a:
                continue
            if [r["id"] for r in a] != [r["id"] for r in b]:
                raise ValueError(f"{name_a}: itens desalinhados com {name_b}")
            out["pareado"][key][lab] = paired_fp_bootstrap(
                ["DD" if r["activated"] else "NDD" for r in a],
                ["DD" if r["activated"] else "NDD" for r in b], n_bootstrap=n_bootstrap,
                groups=None if group_field is None else [r[group_field] for r in a])
    return out

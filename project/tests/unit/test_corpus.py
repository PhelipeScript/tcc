"""Testes do pipeline do corpus: dedup aproximado, balanceamento, split e diagnósticos."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from ddsd.corpus import diagnostics as diag
from ddsd.corpus.balance import balance_by_length, largest_remainder
from ddsd.corpus.config import LengthBin, load_corpus_config
from ddsd.corpus.generation.generator import parse_json_array, screening_reason
from ddsd.corpus.near_dedup import connected_groups, find_near_duplicates, tfidf_matrix
from ddsd.corpus.split import SPLIT_NAMES, stratified_split, stratum_keys, verify_disjoint

WORDS = ("abre fecha toca pausa navegador aba janela musica volume arquivo planilha pasta "
         "ontem hoje amanha voce ele ela gente nossa tipo entao agora depois rapido devagar "
         "o a de do da pra com sem que e um uma meu minha esse essa").split()


def _random_corpus(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    texts = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 9))) for _ in range(n)]
    # Quase-duplicatas plantadas: mesma frase com uma palavra repetida no fim.
    texts += [t + " " + t.split()[-1] for t in texts[: n // 10]]
    return texts


@pytest.mark.parametrize("threshold", [0.95, 0.9, 0.8])
def test_near_duplicates_match_brute_force(threshold: float) -> None:
    """A filtragem por prefixo é exata: encontra os mesmos pares que o produto completo."""
    texts = _random_corpus(600, seed=3)
    matrix = tfidf_matrix(texts, (1, 2))
    found = find_near_duplicates(matrix, threshold)
    dense = (matrix @ matrix.T).toarray()
    expected = {(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts)) if dense[i, j] > threshold}
    got = set(zip(found.left.tolist(), found.right.tolist(), strict=True))
    assert got == expected
    for a, b, cos in zip(found.left, found.right, found.cosine, strict=True):
        assert cos == pytest.approx(dense[a, b])


def test_connected_groups_root_is_smallest_index() -> None:
    texts = ["abre o navegador agora", "abre o navegador agora agora", "fecha a janela", "abre o navegador agora ja"]
    matrix = tfidf_matrix(texts, (1, 2))
    pairs = find_near_duplicates(matrix, 0.5)
    roots = connected_groups(len(texts), pairs)
    assert roots[0] == 0 and roots[2] == 2
    assert all(r <= i for i, r in enumerate(roots))


def test_largest_remainder_respects_caps_and_total() -> None:
    quotas = largest_remainder({"a": 5, "b": 3, "c": 0}, 10, {"a": 4, "b": 10, "c": 10})
    assert sum(quotas.values()) == 10
    assert quotas["a"] == 4  # bateu no teto
    assert quotas["b"] == 6  # recebeu o excedente
    assert quotas["c"] == 0  # peso zero só entra quando os outros se esgotam
    spill = largest_remainder({"a": 1}, 5, {"a": 2, "b": 10})
    assert spill == {"a": 2, "b": 3}
    with pytest.raises(ValueError):
        largest_remainder({"a": 1}, 50, {"a": 2})


BINS = (LengthBin(2, 2), LengthBin(3, 4), LengthBin(5, 1000))


def _synthetic_items(seed: int = 1) -> tuple[list[str], list[str], list[int]]:
    rng = random.Random(seed)
    labels, sources, words = [], [], []
    for _ in range(3000):
        label = rng.choice(["DD", "NDD"])
        # Atalho plantado: DD curto, NDD longo; fonte "x" gera quase só DD.
        w = rng.randint(2, 4) if (label == "DD") ^ (rng.random() < 0.2) else rng.randint(5, 12)
        src = "x" if (label == "DD" and rng.random() < 0.6) else rng.choice(["y", "z"])
        labels.append(label)
        sources.append(src)
        words.append(w)
    return labels, sources, words


@pytest.mark.parametrize("mode", ["faixa", "faixa_fonte", "faixa_fonte_origem"])
def test_balance_equalizes_each_bin(mode: str) -> None:
    labels, sources, words = _synthetic_items()
    outcome = balance_by_length(labels, sources, words, BINS, mode=mode, seed=8)
    kept = outcome.kept
    assert sorted(kept) == kept and len(set(kept)) == len(kept)
    assert set(kept) | set(outcome.discarded) == set(range(len(labels)))
    bin_names = [next(b.name for b in BINS if b.contains(w)) for w in words]
    counts = Counter((bin_names[i], labels[i]) for i in kept)
    for b in BINS:
        assert counts[(b.name, "DD")] == counts[(b.name, "NDD")]
    auc_before = diag.length_auc(labels, words)
    auc_after = diag.length_auc([labels[i] for i in kept], [words[i] for i in kept])
    assert auc_before > 0.7
    assert abs(auc_after - 0.5) < abs(auc_before - 0.5)


def test_cell_balance_removes_source_shortcut() -> None:
    labels, sources, words = _synthetic_items()
    bin_names = [next(b.name for b in BINS if b.contains(w)) for w in words]
    before, _ = diag.categorical_auc_within_bins(labels, sources, bin_names)
    outcome = balance_by_length(labels, sources, words, BINS, mode="faixa_fonte", seed=8)
    k = outcome.kept
    after, _ = diag.categorical_auc_within_bins([labels[i] for i in k], [sources[i] for i in k],
                                                [bin_names[i] for i in k])
    assert before > 0.6
    assert after == pytest.approx(0.5)


def test_balance_is_deterministic() -> None:
    labels, sources, words = _synthetic_items()
    a = balance_by_length(labels, sources, words, BINS, mode="faixa", seed=8).kept
    b = balance_by_length(labels, sources, words, BINS, mode="faixa", seed=8).kept
    c = balance_by_length(labels, sources, words, BINS, mode="faixa", seed=9).kept
    assert a == b and a != c


def test_stratified_split_proportions_and_disjointness() -> None:
    labels, sources, words = _synthetic_items()
    bins = [next(b.name for b in BINS if b.contains(w)) for w in words]
    keys = stratum_keys(labels, sources, bins, min_stratum=10)
    parts = stratified_split(keys, (0.8, 0.1, 0.1), seed=8)
    sizes = {name: len(parts[name]) for name in SPLIT_NAMES}
    assert sum(sizes.values()) == len(labels)
    assert abs(sizes["train"] / len(labels) - 0.8) < 0.01
    texts = [f"t{i}" for i in range(len(labels))]
    verify_disjoint(texts, parts)
    with pytest.raises(AssertionError):
        verify_disjoint(["igual"] * len(labels), parts)
    assert stratified_split(keys, (0.8, 0.1, 0.1), seed=8) == parts


def test_stratum_keys_fall_back_when_small() -> None:
    keys = stratum_keys(["DD"] * 12 + ["DD"], ["a"] * 12 + ["raro"], ["2"] * 13, min_stratum=10)
    assert keys[0] == "DD|a|2"
    assert keys[-1] == "DD|*|*"  # 'raro' sozinho cai até a classe


def test_auc_matches_known_values() -> None:
    assert diag.auc_from_scores([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert diag.auc_from_scores([1, 1, 1, 1], [False, True, False, True]) == 0.5
    assert diag.auc_from_scores([4, 3, 2, 1], [False, False, True, True]) == 0.0
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 4000), rng.normal(0, 1, 4000)
    auc = diag.auc_from_scores(np.r_[pos, neg], [True] * 4000 + [False] * 4000)
    assert auc == pytest.approx(0.7602, abs=0.01)  # Φ(1/√2) para d' = 1
    assert np.isnan(diag.auc_from_scores([1, 2], [True, True]))


def test_parse_json_array_variants() -> None:
    assert parse_json_array('["a", "b"]') == ["a", "b"]
    assert parse_json_array('```json\n["x"]\n```') == ["x"]
    assert parse_json_array('Claro! Aqui: ["um", "dois"] espero ajudar') == ["um", "dois"]
    assert parse_json_array('{"frases": ["k"]}') == ["k"]
    with pytest.raises(ValueError):
        parse_json_array("sem json aqui")


def test_screening_reason() -> None:
    assert screening_reason("alexa abre o chrome") == "palavra_ativacao"
    assert screening_reason("liga o ventilador ai") == "casa_inteligente"
    assert screening_reason("abre o chrome") is None
    assert screening_reason("") == "vazio"


def test_report_dir_defaults_by_config_name() -> None:
    from ddsd.common.paths import PROJECT_ROOT

    assert load_corpus_config().paths.reports == PROJECT_ROOT / "reports" / "corpus" / "principal"
    control = load_corpus_config(PROJECT_ROOT / "config" / "corpus_controle.toml")
    assert control.paths.reports == PROJECT_ROOT / "reports" / "corpus" / "controle"


def test_project_config_is_valid() -> None:
    cfg = load_corpus_config()
    assert cfg.balance.length_bins[0].low == cfg.filters.min_words
    assert cfg.balance.mode == "faixa_fonte_origem"


def test_cohen_kappa_known_values() -> None:
    from ddsd.corpus.annotation import cohen_kappa

    assert cohen_kappa([("DD", "DD"), ("NDD", "NDD")] * 5) == 1.0
    # Exemplo clássico: 20 itens, concordância observada 0,7, esperada 0,5 → κ = 0,4.
    pairs = [("DD", "DD")] * 7 + [("NDD", "NDD")] * 7 + [("DD", "NDD")] * 3 + [("NDD", "DD")] * 3
    assert cohen_kappa(pairs) == pytest.approx(0.4)
    assert np.isnan(cohen_kappa([]))


def test_stratified_annotation_sample_covers_strata() -> None:
    from ddsd.corpus.annotation import stratified_sample

    rows = [{"source": s, "label": l, "bin": b, "text_raw": f"{s}{l}{b}{i}"}
            for s in ("a", "b") for l in ("DD", "NDD") for b in ("2", "9") for i in range(50)]
    sample = stratified_sample(rows, 40, seed=1)
    assert len(sample) == 40
    assert len({(r["source"], r["label"], r["bin"]) for r in sample}) == 8
    assert stratified_sample(rows, 40, seed=1) == sample


def test_rejects_serialized_lists_and_overlong_texts() -> None:
    from ddsd.corpus.build import rejection_reason

    cfg = load_corpus_config()
    raw_list = "['Por gentileza, poderia capturar a tela?', 'Seria possível tirar um foto da tela?']"
    assert rejection_reason("por gentileza poderia capturar a tela", cfg.filters, raw_list) == "lista_serializada"
    tail_list = "o arquivo tá lá no desktop','essa planilha vai ser nossa salvação','você viu a última versão"
    assert rejection_reason("o arquivo", cfg.filters, tail_list) == "lista_serializada"
    quoted = "Organize as pastas 'Projetos', 'Notas' e 'Relatórios'"
    assert rejection_reason("organize as pastas projetos notas e relatorios", cfg.filters, quoted) is None
    assert rejection_reason(" ".join(["palavra"] * 51), cfg.filters) == "longo_demais"


def test_agreement_report_groups_by_strategy(tmp_path, monkeypatch) -> None:
    import json

    from ddsd.corpus import annotation

    monkeypatch.setattr(annotation, "ANNOTATION_DIR", tmp_path / "annotation")
    monkeypatch.setattr(annotation, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "annotation").mkdir()
    sample = [
        {"id": "A0000", "text_raw": "passa o copo", "text": "passa o copo", "label": "NDD", "source": "x_negdif",
         "origin": "mitigation", "strategy": "negativos_dificeis", "bin": "3"},
        {"id": "A0001", "text_raw": "fecha a aba", "text": "fecha a aba", "label": "DD", "source": "x_negdif",
         "origin": "mitigation", "strategy": "negativos_dificeis", "bin": "3"},
        {"id": "A0002", "text_raw": "abre o editor", "text": "abre o editor", "label": "DD", "source": "legado_a",
         "origin": "legacy", "strategy": "legado", "bin": "3"},
    ]
    (tmp_path / "annotation" / "amostra.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample), encoding="utf-8")
    answers = [{"id": "A0000", "answer": "DD"}, {"id": "A0000", "answer": "NDD"},  # correção: a última vale
               {"id": "A0001", "answer": "NDD"}, {"id": "A0002", "answer": "DD"}]
    (tmp_path / "annotation" / "respostas_t.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in answers), encoding="utf-8")
    path = annotation.agreement_report("t")
    summary = json.loads((tmp_path / "reports" / "corpus" / "anotacao_t.json").read_text(encoding="utf-8"))
    assert summary["anotadas"] == 3 and summary["concordancia"] == pytest.approx(2 / 3)
    assert summary["por_estrategia"]["negativos_dificeis"] == {"n": 2, "concordancia": 0.5, "kappa": 0.0}
    assert summary["por_estrategia"]["legado"]["concordancia"] == 1.0
    assert [d["id"] for d in summary["divergencias"]] == ["A0001"]
    assert "Por estratégia" in path.read_text(encoding="utf-8")


def test_freeze_archives_exact_sources_and_refuses_changed_ones(tmp_path) -> None:
    import json
    import tarfile

    from ddsd.common.io import file_sha256
    from ddsd.corpus.freeze import SOURCES_ARCHIVE, SourcesChanged, freeze_corpus

    sources = tmp_path / "sources"
    (sources / "mitigation" / "x").mkdir(parents=True)
    src = sources / "mitigation" / "x" / "dd.jsonl"
    src.write_text('{"text": "abre o editor", "label": "DD"}\n', encoding="utf-8")
    build = tmp_path / "build"
    (build / "splits").mkdir(parents=True)
    (build / "splits" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = {"sources": [{"file": "mitigation/x/dd.jsonl", "sha256": file_sha256(src)}]}
    (build / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    frozen = tmp_path / "frozen"

    (build / "report.md").write_text("# relatório\n", encoding="utf-8")
    archive = freeze_corpus(build, frozen, sources, reports_dir=tmp_path / "reports" / "principal")
    assert (frozen / "splits" / "train.jsonl").is_file() and archive == frozen / SOURCES_ARCHIVE
    assert sorted(p.name for p in (tmp_path / "reports" / "principal").iterdir()) == ["manifest.json", "report.md"]
    with tarfile.open(archive) as tar:
        assert tar.getnames() == ["mitigation/x/dd.jsonl"]
        assert tar.extractfile("mitigation/x/dd.jsonl").read() == src.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_corpus(build, frozen, sources)

    with src.open("a", encoding="utf-8") as handle:  # a geração anexou depois do build
        handle.write('{"text": "fecha a aba", "label": "DD"}\n')
    with pytest.raises(SourcesChanged):
        freeze_corpus(build, frozen, sources, force=True)
    assert (frozen / SOURCES_ARCHIVE).is_file()  # a recusa não destrói o congelado anterior
    assert not (tmp_path / "frozen.tmp").exists()


def test_self_check_rates_come_from_the_generation_batch_lines(tmp_path) -> None:
    from ddsd.corpus.self_check import self_check_stats, strategy_of

    log = tmp_path / "geracao_fonte_negdif.log"
    log.write_text(
        "10:00 INF [corpus.geracao] lote 1 NDD|5p: +6 (faltam 10) rejeitadas {'duplicata': 2, 'juiz_discorda': 3}\n"
        "10:01 INF [corpus.geracao] lote 2 DD|5p: +9 (faltam 1) rejeitadas {'juiz_discorda': 0, 'celula_cheia': 1}\n"
        "10:02 INF [corpus.geracao] lote 3 NDD|5p: +0 (faltam 10) rejeitadas {'juiz_invalido': 10}\n"
        "10:03 AVI [corpus.geracao] lote 4 falhou (NDD|2 palavras): array JSON não encontrado\n")
    (tmp_path / "geracao_vazio.log").write_text("nada aqui\n")
    stats = self_check_stats(sorted(tmp_path.glob("geracao_*.log")))
    ndd, dd = stats["total"]["NDD"], stats["total"]["DD"]
    assert (ndd["julgadas"], ndd["juiz_discorda"], ndd["juiz_invalido"], ndd["aceitas"]) == (19, 3, 10, 6)
    assert ndd["taxa_discorda"] == 3 / 19 and dd["julgadas"] == 10 and dd["taxa_discorda"] == 0.0
    assert list(stats["por_log"]) == ["fonte_negdif"] and list(stats["por_estrategia"]) == ["negativos_dificeis"]
    assert strategy_of("qwen38") == "comprimento"

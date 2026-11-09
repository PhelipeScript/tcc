"""Pontuação da sessão ao vivo contra o roteiro (`ddsd eval sessao`)."""

from __future__ import annotations

from ddsd.common.paths import PROJECT_ROOT
from ddsd.evaluation import live_session as ls
from ddsd.text.normalization import normalize_text

ROTEIRO = PROJECT_ROOT / "docs" / "roteiro_sessao_ao_vivo.md"

MINI = """## Cena 1 - Teste

| # | Quem → para quem | Fala |
|---|---|---|
| 01 | você → (pensando alto) | Agora sim, dá pra concentrar. |
| 02 | você → **computador** | Abre a agenda e mostra o que eu tenho pra hoje. |
| 03 | mãe → você | Abre essa janela um pouquinho. |
| 04 | você → mãe | Me manda o arquivo por e-mail. |
| 05 | você → **computador** | Bloqueia a tela que eu vou almoçar. |
| 06 | você → **computador** | Imprime de novo, só que frente e verso. |

## Resumo

- Casos difíceis de propósito:
  - pedidos a pessoas com vocabulário de computador (03,
    04);
- Nenhuma fala é igual.
"""


def _evidence(uid: str, text: str, label: str, p_dd: float) -> dict:
    return {"utterance": uid, "normalized": normalize_text(text), "text_raw": f" {text}", "label": label,
            "reason": "classificador", "classifier": "bertimbau",
            "votes": {"bertimbau": {"label": label, "p_dd": p_dd, "threshold": 0.5}},
            "reliability": {"label": "alta"}, "audio": f"{uid}.wav", "span_s": [0.0, 1.0], "latency_s": 1.15,
            "close_reason": "fim_da_fala", "timed_out": False, "contexto": {"classificador": "bertimbau"},
            "tasks": [{"task": f"{uid}-F", "asr_latency_s": 0.4,
                       "predictions": {"bertimbau": {"latency_ms": 30.0}}}]}


def _mini_session(tmp_path):
    roteiro = tmp_path / "roteiro.md"
    roteiro.write_text(MINI, encoding="utf-8")
    script = ls.parse_script(roteiro)
    spoken = script.spoken_by({"você"})
    evidences = [
        _evidence("U0001", "Agora sim, dá pra concentrar. Abre a agenda e mostra o que eu tenho pra hoje.", "DD", 0.99),
        _evidence("U0002", "Uhum.", "NDD", 0.01),
        _evidence("U0003", "Me manda o arquivo por e-mail.", "DD", 0.97),
        _evidence("U0004", "Bloqueei a tela que eu vou almoçar.", "NDD", 0.02),
        _evidence("U0005", "Imprime de novo, só que frente e verso.", "NDD", 0.03),
    ]
    # No texto do roteiro, o classificador acerta os pedidos 02 e 05, erra a 06 e ativa na 04.
    on_script = {"me manda o arquivo por e mail": ("DD", 0.97), "imprime de novo so que frente e verso": ("NDD", 0.03)}

    def judge(text: str) -> tuple[str, float]:
        return on_script.get(text, ("DD", 0.99))

    groups = ls.align([normalize_text(line.text) for line in spoken], [e["normalized"] for e in evidences])
    records = ls.score(groups, spoken, evidences, script, judge)
    return script, spoken, evidences, records


def test_parse_script_reads_lines_speakers_and_hard_cases():
    script = ls.parse_script(ROTEIRO)
    assert len(script.lines) == 113
    assert sum(line.label == "DD" for line in script.lines) == 42
    spoken = script.spoken_by({"você"})
    assert (len(spoken), sum(line.label == "DD" for line in spoken)) == (88, 41)
    by_number = {line.number: line for line in script.lines}
    assert (by_number[55].speaker, by_number[55].label) == ("mãe", "DD")
    assert by_number[2].addressee_type == "pensando alto"
    assert by_number[97].addressee_type == "pessoa na chamada"
    assert by_number[84].addressee_type == "pessoa no quarto"
    cases = script.categories
    assert len(cases["pedidos a pessoas com vocabulário de computador"]) == 11
    assert cases["o mesmo pedido feito primeiro a uma pessoa e depois ao computador"] == (28, 29, 36, 41, 53, 54)
    assert len(cases["falas de até 4 palavras nas duas classes"]) == 14


def test_align_handles_merge_split_improvisation_and_missing_line():
    script = [normalize_text(t) for t in (
        "Abre o navegador e entra no meu e-mail.", "Agora sim, dá pra concentrar.",
        "Fecha todas as abas do navegador.", "Abro sim.", "Salva o documento.",
        "Depois do almoço eu volto pra terminar a conclusão.")]
    spoken = [normalize_text(t) for t in (
        "Abra o navegador e entra no meu e-mail.", "Uhum.",
        "Agora sim, dá pra concentrar. Fecha todas as abas do navegador.", "Abro sim, calma aí.",
        "Depois do almoço eu volto", "pra terminar a conclusão.")]
    groups = ls.align(script, spoken)
    assert [(g.lines, g.utterances, g.kind) for g in groups] == [
        ((0,), (0,), "1:1"), ((), (1,), "fora_do_roteiro"), ((1, 2), (2,), "juncao"), ((3,), (3,), "1:1"),
        ((4,), (), "sem_fala"), ((5,), (4, 5), "divisao")]


def test_score_uses_the_group_label_and_finds_the_error_origin(tmp_path):
    script, spoken, _, records = _mini_session(tmp_path)
    by_lines = {tuple(r["linhas"]): r for r in records}
    merged = by_lines[(1, 2)]
    assert (merged["tipo"], merged["label"], merged["acerto"], merged["destinatario"]) == ("juncao", "DD", True,
                                                                                            "computador")
    extra = by_lines[()]
    assert (extra["tipo"], extra["label"], extra["acerto"]) == ("fora_do_roteiro", None, None)
    assert (by_lines[(4,)]["origem"], by_lines[(5,)]["origem"], by_lines[(6,)]["origem"]) == (
        "classificador", "transcricao", "classificador")
    assert by_lines[(4,)]["categorias"] == ["pedidos a pessoas com vocabulário de computador"]

    s = ls.summarize(records, spoken, script)
    assert (s["pedidos_atendidos"]["ativados"], s["pedidos_atendidos"]["n"]) == (1, 3)
    assert (s["falsas_ativacoes"]["ativados"], s["falsas_ativacoes"]["n"]) == (1, 1)
    assert (s["no_roteiro"]["pedidos_atendidos"]["ativados"], s["no_roteiro"]["pedidos_perdidos"]) == (2, [[6]])
    assert s["por_linha"] == {"linhas_ndd": 2, "linhas_ndd_em_falas_ativadas": [1, 4], "absorvidas_em_pedidos": [1]}
    assert (s["casamento"]["juncao"], s["casamento"]["juncoes"], s["casamento"]["sem_fala"]) == (1, [[1, 2]], 0)
    assert s["erros"]["por_rotulo"]["DD"] == {"segmentacao": 0, "classificador": 1, "transcricao": 1}
    assert s["fora_do_roteiro"] == {"falas": 1, "ativadas": 0}
    case = s["por_caso_dificil"]["pedidos a pessoas com vocabulário de computador"]
    assert (case["linhas_lidas"], case["nao_lidas"], case["NDD"]["ativados"]) == (1, [3], 1)
    assert s["linhas_nao_lidas"] == {"n": 1, "falantes": ["mãe"], "DD": []}


def test_build_report_renders_counts_in_words(tmp_path):
    script, spoken, evidences, records = _mini_session(tmp_path)
    session = {"metadados": {"config": {"classifiers": {"active": "bertimbau"}},
                             "codigo": {"commit": "abc1234def", "alteracoes_nao_commitadas": False}},
               "resumo": {"erros": 0, "pendentes_no_fim": 0, "thread_encerrada": True}}
    payload = {"codigo": {"commit": "fedcba98", "alteracoes_nao_commitadas": False}, "sessao": "20261106-101231",
               "roteiro": "docs/roteiro.md", "roteiro_sha256": "0" * 64, "evidencias_sha256": "1" * 64,
               "falantes": ["você"], "inicio": "2026-11-06T10:12:31-0300", "fim": "2026-11-06T10:22:03-0300",
               "classificador": {"nome": "bertimbau", "execucao": "bertimbau-s44", "limiar": 0.685},
               "asr": {"backend": "mlx", "modelo": "mlx-community/whisper-large-v3-turbo", "semente": 0},
               "paridade": {"falas": 5, "identicas": 5, "max_delta_p": 0.0},
               "resumo": ls.summarize(records, spoken, script), "latencia": ls.latency_summary(evidences),
               "verificacoes": ls.session_checks(evidences, session, {f"U000{i}.wav": 1.0 for i in range(1, 6)})}
    text = ls.build_report(payload, records).render()
    assert "1 junção (01 + 02), nenhuma divisão, nenhuma linha sem fala e 1 fala fora do roteiro" in text
    assert "o classificador perderia o pedido 06 e ativaria na linha NDD 04" in text
    assert "porque a linha 01 foi dita na mesma fala que um pedido ao computador" in text
    assert "9,5 min" in text and "COM ALTERAÇÕES" not in text

"""
Descrição:
    Roda os classificadores já treinados sobre as diferentes condições de
    entrada do experimento (texto original, ASR 1-best, ASR n-best),
    reaproveitando `ModelSession` de `src.training.predict_interactive` —
    o mesmo carregamento de checkpoint e limiar congelado usados na predição
    interativa e em `podcast_eval.py`.

Uso:
    from src.asr_eval.classify import load_sessions, build_nbest_input, predict_all
"""

from __future__ import annotations

from src.data_prep.normalizer import normalize_text
from src.training.predict_interactive import ModelSession


def load_sessions(model_keys: tuple[str, ...]) -> dict[str, ModelSession]:
    """
    Descrição:
        Carrega o melhor checkpoint e o limiar congelado de cada modelo.

    Args:
        model_keys: Chaves dos modelos em `src.training.config.MODELS`.

    Retorna:
        Mapeamento `chave -> ModelSession`.
    """
    return {key: ModelSession.load(key) for key in model_keys}


def build_nbest_input(hypotheses: list[str], session: ModelSession) -> str:
    """
    Descrição:
        Concatena as hipóteses n-best em uma única entrada, separadas pelo
        token de separação real do tokenizador do modelo (não um literal
        `"[SEP]"` fixo: modelos diferentes usam tokens de separação
        diferentes, ex. DeBERTa usa `</s>`).

        Cada hipótese é normalizada individualmente (mesma normalização do
        treino) antes de ser concatenada; hipóteses que ficam vazias após a
        normalização são descartadas.

    Args:
        hypotheses: Lista de transcrições candidatas, na ordem de score do
            beam search (a primeira é a 1-best).
        session: Sessão do modelo — fornece o tokenizador e, com ele, o
            token de separação correto.

    Retorna:
        Texto único pronto para `session.predict`.
    """
    normalized = [normalize_text(text) for text in hypotheses]
    normalized = [text for text in normalized if text]
    separator = session.tokenizer.sep_token or "[SEP]"
    return f" {separator} ".join(normalized)


def predict_all(
    sessions: dict[str, ModelSession], text: str
) -> dict[str, tuple[str, float]]:
    """
    Descrição:
        Roda todos os modelos carregados sobre o mesmo texto de entrada.

    Args:
        sessions: Mapeamento `chave -> ModelSession`.
        text: Texto já normalizado (ou já concatenado por
            `build_nbest_input`, que normaliza internamente).

    Retorna:
        Mapeamento `chave -> (classe_predita, probabilidade_DD)`.
    """
    return {key: session.predict(text) for key, session in sessions.items()}

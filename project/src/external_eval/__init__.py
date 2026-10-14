"""
Descrição:
    Validação externa dos 3 classificadores contra fala real (não sintética):
    CORAA, NURC-SP e TAGARELA, três corpora públicos de fala espontânea em
    português brasileiro, disponíveis no Hugging Face Hub.

    O corpus de treino é 100% sintético (gerado por LLM); o risco reconhecido
    no README do projeto é que o desempenho quase perfeito nos splits
    internos reflita separabilidade estilística artificial, não generalização
    real. Este módulo testa a classe NDD nesses corpora reais — não há corpus
    público de comandos DD reais para computador, então a validação cobre
    apenas a taxa de falso positivo em fala espontânea genuína (rótulo NDD
    assumido por suposição de domínio, não anotação humana; é uma limitação
    documentada, não um problema a "resolver" sintetizando DD real).

    Não retreina nada: só lê texto (nunca áudio, que não é decodificado) e
    roda os 3 `ModelSession` já treinados.

Uso:
    python -m src.external_eval.run_external_eval
"""

from __future__ import annotations

__all__ = ["config", "fetch", "normalize", "evaluate", "engine"]

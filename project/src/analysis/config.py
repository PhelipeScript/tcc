"""
Descrição:
    Configuração da análise: quais execuções existem em `outputs/` e onde
    gravar os artefatos gerados.

Uso:
    from src.analysis.config import AnalysisConfig
    config = AnalysisConfig()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.training.config import DEFAULT_OUTPUT_DIR

#: Nome de exibição de cada execução, na ordem usada pelos gráficos e tabelas.
DISPLAY_NAMES: dict[str, str] = {
    "bertimbau": "BERTimbau",
    "albertina": "Albertina",
    "debertinha": "DeBERTinha",
    "baseline_svm": "TF-IDF + SVM",
    "baseline_logreg": "TF-IDF + LogReg",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Descrição:
        Configuração de uma rodada de análise.

    Atributos:
        output_dir: Raiz de `outputs/`, onde as execuções já treinadas moram.
        analysis_dir: Diretório onde os artefatos da análise são gravados.
        transformer_models: Chaves das execuções de Transformer (têm
            `checkpoints/` com `trainer_state.json`; baselines não têm).
        all_models: Todas as execuções a incluir nas comparações que não
            dependem de `trainer_state.json` (per-fonte, calibração,
            bootstrap) — preenchido em `__post_init__` incluindo os
            baselines, se já tiverem sido treinados.
        n_bootstrap: Número de reamostragens do bootstrap pareado.
        seed: Semente do bootstrap.
    """

    output_dir: Path = DEFAULT_OUTPUT_DIR
    analysis_dir: Path = field(init=False)
    transformer_models: tuple[str, ...] = ("bertimbau", "albertina", "debertinha")
    all_models: tuple[str, ...] = field(init=False)
    n_bootstrap: int = 2000
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_dir", self.output_dir / "analysis")
        candidates = (*self.transformer_models, "baseline_svm", "baseline_logreg")
        available = tuple(
            key for key in candidates if (self.output_dir / key / "metrics_test.json").exists()
        )
        object.__setattr__(self, "all_models", available)

    def run_dir(self, model_key: str) -> Path:
        """
        Descrição:
            Caminho da execução de um modelo em `outputs/`.

        Args:
            model_key: Chave da execução (ex.: `"bertimbau"`).

        Retorna:
            `output_dir/model_key`.
        """
        return self.output_dir / model_key

    def display_name(self, model_key: str) -> str:
        """
        Descrição:
            Nome de exibição de um modelo, para títulos e legendas de gráfico.

        Args:
            model_key: Chave da execução.

        Retorna:
            Nome legível, ou a própria chave se não estiver em `DISPLAY_NAMES`.
        """
        return DISPLAY_NAMES.get(model_key, model_key)

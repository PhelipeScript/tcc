"""
Descrição:
    Configuração dos baselines: caminhos, hiperparâmetros do vetorizador e
    grade de busca do `C` para SVM/Regressão Logística.

    Reaproveita `DEFAULT_DATA_DIR`/`DEFAULT_OUTPUT_DIR` de `src.training.config`
    para apontar para os mesmos splits e a mesma raiz de saída usados pelos
    três modelos Transformer, garantindo comparabilidade direta.

Uso:
    from src.baseline.config import BaselineConfig
    config = BaselineConfig()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.training.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

#: Chave de cada baseline, usada no nome do diretório de saída (`baseline_<key>`).
MODEL_KEYS: Final[tuple[str, ...]] = ("svm", "logreg")


@dataclass(frozen=True)
class BaselineConfig:
    """
    Descrição:
        Configuração de uma execução de baseline clássico.

    Atributos:
        data_dir: Diretório com `train.jsonl`, `val.jsonl` e `test.jsonl`.
        output_dir: Diretório raiz das execuções (mesmo usado pelo treino).
        ngram_range: Faixa de n-gramas de palavra do TF-IDF, conforme a
            metodologia (`1` a `3`).
        max_features: Teto de vocabulário do TF-IDF. Necessário para não
            estourar memória com n-gramas 1-3 sobre ~190 mil frases curtas.
        min_df: Frequência mínima de documento para um termo entrar no
            vocabulário — descarta ruído de baixa frequência (erros de
            digitação, vazamento de script estrangeiro).
        c_grid: Valores de `C` testados no `GridSearchCV`.
        cv: Número de folds da validação cruzada usada para escolher `C`.
        calibration_cv: Número de folds usados por `CalibratedClassifierCV`
            para calibrar o LinearSVC escolhido (só se aplica ao SVM).
        seed: Semente usada nos estimadores e no shuffle do `GridSearchCV`.
        svm_max_iter: Máximo de iterações do `LinearSVC`.
        logreg_max_iter: Máximo de iterações da `LogisticRegression`.
        max_train_samples: Limita o treino a N exemplos (teste de fumaça).
    """

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    ngram_range: tuple[int, int] = (1, 3)
    max_features: int | None = 50_000
    min_df: int = 2
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    cv: int = 5
    calibration_cv: int = 5
    seed: int = 42
    svm_max_iter: int = 5_000
    logreg_max_iter: int = 1_000
    max_train_samples: int | None = None

    def run_dir(self, model_key: str) -> Path:
        """
        Descrição:
            Monta o diretório de saída de um baseline.

        Args:
            model_key: Chave do baseline (`"svm"` ou `"logreg"`).

        Retorna:
            `output_dir/baseline_<model_key>`.
        """
        return self.output_dir / f"baseline_{model_key}"

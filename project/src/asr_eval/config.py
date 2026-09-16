"""
Descrição:
    Configuração do experimento de ASR simulado.

Uso:
    from src.asr_eval.config import AsrEvalConfig
    config = AsrEvalConfig()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.training.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class AsrEvalConfig:
    """
    Descrição:
        Configuração de uma execução do experimento ASR simulado (TTS →
        Whisper → classificador).

    Atributos:
        data_dir: Diretório com `test.jsonl` (mesma partição usada pelos
            modelos treinados — nunca amostrar de treino/validação).
        output_dir: Raiz de saída do experimento.
        audio_dir: Onde os `.mp3` sintetizados são cacheados.
        sample_size: Tamanho da amostra do conjunto de teste.
        n_values: Valores de N testados na concatenação de hipóteses n-best.
        whisper_size: Tamanho do modelo Whisper (`faster-whisper`).
        beam_size: Tamanho do beam search — deve ser >= `max(n_values)`.
        language: Código de idioma forçado no Whisper.
        model_keys: Modelos de classificação avaliados.
        seed: Semente da amostragem estratificada e da atribuição de vozes.
        device: Dispositivo do Whisper (`"auto"`, `"cpu"` ou `"cuda"`).
        compute_type: Precisão do `ctranslate2` (`"default"`, `"int8"`,
            `"float16"`).
    """

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR / "asr_eval"
    audio_dir: Path = DEFAULT_OUTPUT_DIR / "asr_eval" / "audio"
    sample_size: int = 750
    n_values: tuple[int, ...] = (1, 3, 5)
    whisper_size: str = "medium"
    beam_size: int = 5
    language: str = "pt"
    model_keys: tuple[str, ...] = ("bertimbau", "albertina", "debertinha")
    seed: int = 42
    device: str = "auto"
    compute_type: str = "default"

    @property
    def max_n(self) -> int:
        """
        Descrição:
            Maior valor de N testado — é quantas hipóteses pedir ao Whisper
            de uma vez (as condições menores usam um prefixo da mesma lista).

        Retorna:
            `max(n_values)`.
        """
        return max(self.n_values)

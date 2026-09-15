"""
Descrição:
    Fine-tuning do DeBERTinha para classificação DD vs NDD.

    Encoder DeBERTaV3 compacto para português brasileiro.

    Este script é intencionalmente fino: toda a lógica de treino, avaliação e
    geração de artefatos vive nos módulos compartilhados do pacote
    (`engine`, `data`, `metrics`, `modeling`, `reporting`). O que é
    específico deste modelo — checkpoint, família da arquitetura, atributo de
    dropout, lote, taxa de aprendizado e épocas — está declarado em
    `config.MODELS["debertinha"]`.

    Portanto: para ajustar hiperparâmetros, edite `config.py` ou passe as
    opções na linha de comando; não é necessário (nem desejável) alterar este
    arquivo.

Uso:
    python -m src.training.train_debertinha
    python -m src.training.train_debertinha --epochs 5 --batch-size 16
    python -m src.training.train_debertinha --text-field text_raw   # ablação com acento
    python -m src.training.train_debertinha --help

Saída:
    outputs/debertinha/
        metrics_val.json, metrics_test.json
        confusion_matrix_test.{csv,png}
        roc_curve_test.png, det_curve_test.png
        predictions_test.jsonl, metrics_by_source_test.json, errors_test.json
        run_manifest.json
        checkpoints/
"""

from __future__ import annotations

from src.training.cli import main_for_model

#: Chave deste modelo em `src.training.config.MODELS`.
MODEL_KEY = "debertinha"


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Executa o treino do DeBERTinha com a configuração declarada em
        `config.MODELS[MODEL_KEY]`, aplicando as sobreposições da linha de
        comando.

    Args:
        argv: Argumentos de linha de comando. `None` usa `sys.argv`.

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    return main_for_model(MODEL_KEY, argv)


if __name__ == "__main__":
    raise SystemExit(main())

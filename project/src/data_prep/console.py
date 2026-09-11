"""
Descrição:
    Utilidades mínimas de saída no terminal, com códigos ANSI de cor.

Uso:
    from src.data_prep import console
    console.header("Stage 1 - snapshot")
    console.ok("18 arquivos copiados")
"""

from __future__ import annotations

from typing import Final, Sequence

_RESET: Final[str] = "\033[0m"
_BOLD: Final[str] = "\033[1m"
_DIM: Final[str] = "\033[2m"
_RED: Final[str] = "\033[31m"
_GREEN: Final[str] = "\033[32m"
_YELLOW: Final[str] = "\033[33m"
_CYAN: Final[str] = "\033[36m"


def header(text: str) -> None:
    """
    Descrição:
        Imprime um título de seção destacado, precedido de uma linha em branco.

    Args:
        text: Título a exibir.
    """
    print(f"\n{_BOLD}{_CYAN}=== {text} ==={_RESET}", flush=True)


def info(text: str) -> None:
    """
    Descrição:
        Imprime uma mensagem informativa neutra.

    Args:
        text: Mensagem a exibir.
    """
    print(f"  {text}", flush=True)


def detail(text: str) -> None:
    """
    Descrição:
        Imprime uma mensagem secundária, em tom esmaecido.

    Args:
        text: Mensagem a exibir.
    """
    print(f"  {_DIM}{text}{_RESET}", flush=True)


def ok(text: str) -> None:
    """
    Descrição:
        Imprime uma mensagem de sucesso.

    Args:
        text: Mensagem a exibir.
    """
    print(f"  {_GREEN}OK{_RESET} {text}", flush=True)


def warn(text: str) -> None:
    """
    Descrição:
        Imprime um aviso — algo inesperado, mas que não interrompe o pipeline.

    Args:
        text: Mensagem a exibir.
    """
    print(f"  {_YELLOW}AVISO{_RESET} {text}", flush=True)


def error(text: str) -> None:
    """
    Descrição:
        Imprime uma mensagem de erro.

    Args:
        text: Mensagem a exibir.
    """
    print(f"  {_RED}ERRO{_RESET} {text}", flush=True)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """
    Descrição:
        Imprime uma tabela de largura fixa, alinhando colunas numéricas à direita
        e textuais à esquerda. Usada pelos estágios para exibir contagens.

    Args:
        headers: Rótulos das colunas.
        rows: Linhas da tabela; cada célula é convertida com `str()`.
    """
    if not rows:
        detail("(sem linhas)")
        return

    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in text_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        """
        Descrição:
            Formata uma linha da tabela. A primeira coluna é tratada como rótulo
            e alinhada à esquerda; as demais, como números, à direita.

        Args:
            cells: Células já convertidas em string.

        Retorna:
            A linha formatada, com o recuo padrão de duas colunas.
        """
        parts = [cells[0].ljust(widths[0])]
        parts += [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
        return "  " + "  ".join(parts)

    print(_BOLD + fmt(list(headers)) + _RESET, flush=True)
    for row in text_rows:
        print(fmt(row), flush=True)

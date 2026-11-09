"""Montagem de relatórios em Markdown."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


def format_cell(value: object) -> str:
    """Formata números no padrão do relatório (inteiros com milhar, floats com 4 casas)."""
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


@dataclass
class MarkdownReport:
    """Relatório incremental: título, seções, tabelas e notas."""

    title: str
    _blocks: list[str] = field(default_factory=list)

    def heading(self, text: str, level: int = 2) -> None:
        self._blocks.append(f"{'#' * level} {text}")

    def paragraph(self, text: str) -> None:
        self._blocks.append(text)

    def bullets(self, items: Iterable[str]) -> None:
        self._blocks.append("\n".join(f"- {item}" for item in items))

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        lines += ["| " + " | ".join(format_cell(cell) for cell in row) + " |" for row in rows]
        self._blocks.append("\n".join(lines))

    def code(self, text: str, language: str = "") -> None:
        self._blocks.append(f"```{language}\n{text}\n```")

    def render(self) -> str:
        return "\n\n".join([f"# {self.title}", *self._blocks]) + "\n"

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path

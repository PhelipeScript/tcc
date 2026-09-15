"""
Descrição:
    Coleta e consolidação das estatísticas do pipeline.

    Cada estágio grava um JSON próprio em `data/reports/stages/stageN.json` com
    suas contagens. Este módulo agrega esses arquivos em um relatório único
    (`pipeline_report.json` e `pipeline_report.md`), cujas tabelas são feitas para
    serem copiadas direto no capítulo de metodologia do TCC.

    O formato das estatísticas é genérico de propósito: um estágio descreve o que
    quer exibir na forma de tabelas (`tables`) e observações (`notes`), e este
    módulo apenas renderiza. Assim, adicionar um estágio novo não exige alterar
    o gerador de relatório.

Uso:
    from src.data_prep import report
    report.save_stage_stats(report.StageStats(stage=1, name="snapshot"))
    report.write_report()

Saída:
    data/reports/stages/stageN.json
    data/reports/pipeline_report.json
    data/reports/pipeline_report.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.data_prep import console
from src.data_prep.config import REPORTS_DIR

#: Subdiretório onde as estatísticas por estágio são persistidas.
STAGE_STATS_DIR: Path = REPORTS_DIR / "stages"

#: Relatório consolidado, em JSON (para consumo programático).
REPORT_JSON_PATH: Path = REPORTS_DIR / "pipeline_report.json"

#: Relatório consolidado, em Markdown (para leitura e para o texto do TCC).
REPORT_MD_PATH: Path = REPORTS_DIR / "pipeline_report.md"


@dataclass
class Table:
    """
    Descrição:
        Uma tabela de estatísticas, renderizável no terminal e em Markdown.

    Atributos:
        title: Legenda da tabela.
        headers: Rótulos das colunas.
        rows: Linhas; cada célula é convertida com `str()` na renderização.
    """

    title: str
    headers: list[str]
    rows: list[list[Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Descrição:
            Serializa a tabela em dicionário.

        Retorna:
            Dicionário com `title`, `headers` e `rows`.
        """
        return {"title": self.title, "headers": self.headers, "rows": self.rows}

    def render_console(self) -> None:
        """
        Descrição:
            Imprime a tabela no terminal, com a legenda em tom esmaecido.
        """
        console.detail(self.title)
        console.table(self.headers, self.rows)

    def render_markdown(self) -> list[str]:
        """
        Descrição:
            Renderiza a tabela como Markdown (GitHub-flavored).

        Retorna:
            Lista de linhas Markdown, incluindo a legenda em negrito.
        """
        lines = [f"**{self.title}**", ""]
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        lines.append("")
        return lines


@dataclass
class StageStats:
    """
    Descrição:
        Estatísticas produzidas por um estágio do pipeline.

    Atributos:
        stage: Número do estágio (1 a 7).
        name: Nome curto do estágio, ex.: `"snapshot"`.
        totals: Contagem final por classe ao término do estágio.
        tables: Tabelas detalhadas a exibir no relatório.
        notes: Observações em texto livre (avisos, decisões, valores derivados).
        extra: Campos adicionais arbitrários, preservados no JSON.
        timestamp: Instante de gravação, em ISO 8601 UTC. Preenchido
            automaticamente por `save_stage_stats`.
    """

    stage: int
    name: str
    totals: dict[str, int] = field(default_factory=dict)
    tables: list[Table] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def add_table(self, title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> Table:
        """
        Descrição:
            Cria e anexa uma tabela às estatísticas do estágio.

        Args:
            title: Legenda da tabela.
            headers: Rótulos das colunas.
            rows: Linhas da tabela.

        Retorna:
            A `Table` criada, já anexada a `self.tables`.
        """
        table = Table(title=title, headers=list(headers), rows=[list(r) for r in rows])
        self.tables.append(table)
        return table

    def note(self, text: str) -> None:
        """
        Descrição:
            Anexa uma observação em texto livre.

        Args:
            text: Observação a registrar.
        """
        self.notes.append(text)

    def to_dict(self) -> dict[str, Any]:
        """
        Descrição:
            Serializa as estatísticas do estágio em dicionário.

        Retorna:
            Dicionário pronto para `json.dump`.
        """
        return {
            "stage": self.stage,
            "name": self.name,
            "timestamp": self.timestamp,
            "totals": self.totals,
            "tables": [t.to_dict() for t in self.tables],
            "notes": self.notes,
            "extra": self.extra,
        }

    def render_console(self) -> None:
        """
        Descrição:
            Imprime no terminal as tabelas, observações e totais do estágio.
        """
        for table in self.tables:
            table.render_console()
        for note in self.notes:
            console.info(note)
        if self.totals:
            total = sum(self.totals.values())
            partes = "  ".join(f"{k}={v}" for k, v in self.totals.items())
            console.ok(f"totais: {partes}  (geral={total})")


def stage_stats_path(stage: int) -> Path:
    """
    Descrição:
        Retorna o caminho do JSON de estatísticas de um estágio.

    Args:
        stage: Número do estágio.

    Retorna:
        Caminho para `data/reports/stages/stageN.json`.
    """
    return STAGE_STATS_DIR / f"stage{stage}.json"


def save_stage_stats(stats: StageStats) -> Path:
    """
    Descrição:
        Persiste as estatísticas de um estágio, preenchendo o timestamp.

    Args:
        stats: Estatísticas a gravar.

    Retorna:
        Caminho do arquivo gravado.
    """
    stats.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = stage_stats_path(stats.stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_all_stage_stats() -> list[dict[str, Any]]:
    """
    Descrição:
        Carrega as estatísticas de todos os estágios já executados, em ordem.

    Retorna:
        Lista de dicionários de estatísticas, ordenada pelo número do estágio.
        Estágios ainda não executados simplesmente não aparecem.
    """
    if not STAGE_STATS_DIR.exists():
        return []
    loaded: list[dict[str, Any]] = []
    for path in sorted(STAGE_STATS_DIR.glob("stage*.json")):
        try:
            loaded.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            console.warn(f"estatísticas ilegíveis em {path.name}: {exc}")
    loaded.sort(key=lambda s: s.get("stage", 0))
    return loaded


def build_markdown(all_stats: list[dict[str, Any]]) -> str:
    """
    Descrição:
        Monta o relatório consolidado em Markdown a partir das estatísticas dos
        estágios, incluindo uma tabela-resumo da evolução das contagens.

    Args:
        all_stats: Estatísticas carregadas por `load_all_stage_stats`.

    Retorna:
        Conteúdo Markdown completo do relatório.
    """
    lines: list[str] = [
        "# Relatório do pipeline de preparação do dataset DDSD",
        "",
        f"Gerado em {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
    ]

    if not all_stats:
        lines += ["Nenhum estágio foi executado ainda.", ""]
        return "\n".join(lines)

    # Resumo: evolução das contagens por classe ao longo dos estágios.
    labels: list[str] = []
    for stats in all_stats:
        for label in stats.get("totals", {}):
            if label not in labels:
                labels.append(label)

    lines += ["## Resumo — evolução das contagens", ""]
    lines.append("| Estágio | " + " | ".join(labels) + " | Total |")
    lines.append("|" + "|".join(["---"] * (len(labels) + 2)) + "|")
    for stats in all_stats:
        totals = stats.get("totals", {})
        cells = [str(totals.get(label, "-")) for label in labels]
        soma = sum(v for v in totals.values() if isinstance(v, int))
        lines.append(
            f"| {stats['stage']}. {stats['name']} | " + " | ".join(cells) + f" | {soma} |"
        )
    lines.append("")

    # Detalhe por estágio.
    for stats in all_stats:
        lines += [f"## Stage {stats['stage']} — {stats['name']}", ""]
        if stats.get("timestamp"):
            lines += [f"_Executado em {stats['timestamp']}._", ""]
        for note in stats.get("notes", []):
            lines.append(f"- {note}")
        if stats.get("notes"):
            lines.append("")
        for table in stats.get("tables", []):
            lines += Table(
                title=table["title"], headers=table["headers"], rows=table["rows"]
            ).render_markdown()

    return "\n".join(lines)


def write_report() -> tuple[Path, Path]:
    """
    Descrição:
        Consolida as estatísticas de todos os estágios e grava o relatório final
        em JSON e em Markdown.

    Retorna:
        Tupla `(caminho_json, caminho_markdown)`.
    """
    all_stats = load_all_stage_stats()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(
        json.dumps({"stages": all_stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_MD_PATH.write_text(build_markdown(all_stats), encoding="utf-8")
    return REPORT_JSON_PATH, REPORT_MD_PATH


def main(argv: list[str] | None = None) -> int:
    """
    Descrição:
        Ponto de entrada de linha de comando: regera o relatório consolidado a
        partir das estatísticas já gravadas, sem reexecutar nenhum estágio.

    Args:
        argv: Argumentos de linha de comando (não utilizados; mantido para
            uniformidade com os módulos de estágio).

    Retorna:
        Código de saída do processo (0 em sucesso).
    """
    console.header("Relatório consolidado")
    json_path, md_path = write_report()
    console.ok(f"{json_path.relative_to(json_path.parents[2])}")
    console.ok(f"{md_path.relative_to(md_path.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

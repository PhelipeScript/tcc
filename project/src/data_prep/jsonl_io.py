"""
Descrição:
    Leitura e escrita de JSONL para o pipeline de preparação do dataset.

    Define o registro canônico (`Record`) que trafega entre todos os estágios e
    as funções de I/O correspondentes.

Uso:
    from src.data_prep.jsonl_io import Record, read_jsonl, write_jsonl

Saída:
    Uma linha JSON por registro, UTF-8, `ensure_ascii=False`:
    {"text": "abre a pasta downloads", "text_raw": "Abre a pasta Downloads.",
     "label": "DD", "source": "glm"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.data_prep import console

#: Chaves obrigatórias de um registro serializado.
RECORD_KEYS: tuple[str, ...] = ("text", "text_raw", "label", "source")


@dataclass(slots=True)
class Record:
    """
    Descrição:
        Registro canônico do dataset, usado do Stage 1 ao Stage 7.

    Atributos:
        text: Texto de trabalho. No Stage 1 é idêntico a `text_raw`; a partir do
            Stage 2 contém a forma normalizada (sem acento, sem pontuação,
            minúscula, espaços colapsados). É a chave usada em toda deduplicação.
        text_raw: Texto original do provedor, preservado sem alteração. Permite a
            ablação "com acento vs sem acento" no treino sem reprocessar o
            pipeline.
        label: Rótulo da classe, `"DD"` ou `"NDD"`.
        source: Provedor de origem (ex.: `"glm"`). Necessário para estratificar o
            split por fonte.
    """

    text: str
    text_raw: str
    label: str
    source: str

    def to_dict(self) -> dict[str, str]:
        """
        Descrição:
            Converte o registro em dicionário, com as chaves em ordem canônica.

        Retorna:
            Dicionário com as chaves de `RECORD_KEYS`.
        """
        return {
            "text": self.text,
            "text_raw": self.text_raw,
            "label": self.label,
            "source": self.source,
        }

    def to_json(self) -> str:
        """
        Descrição:
            Serializa o registro como uma única linha JSON (sem `indent`).

        Retorna:
            String JSON compacta, com caracteres não-ASCII literais.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        """
        Descrição:
            Constrói um `Record` a partir de um dicionário desserializado.

        Args:
            data: Dicionário contendo, no mínimo, as chaves de `RECORD_KEYS`.

        Retorna:
            Instância de `Record`.

        Levanta:
            KeyError: Se alguma chave obrigatória estiver ausente.
        """
        missing = [k for k in RECORD_KEYS if k not in data]
        if missing:
            raise KeyError(f"registro sem as chaves obrigatórias: {missing}")
        return cls(
            text=data["text"],
            text_raw=data["text_raw"],
            label=data["label"],
            source=data["source"],
        )

    def replace_text(self, new_text: str) -> "Record":
        """
        Descrição:
            Retorna uma cópia do registro com `text` substituído, preservando
            `text_raw`, `label` e `source`.

        Args:
            new_text: Novo valor para o campo `text`.

        Retorna:
            Novo `Record` (o original não é modificado).
        """
        return Record(
            text=new_text,
            text_raw=self.text_raw,
            label=self.label,
            source=self.source,
        )


def read_jsonl(path: Path, *, strict: bool = False) -> Iterator[Record]:
    """
    Descrição:
        Lê um arquivo JSONL no formato canônico e produz `Record`s.

        Linhas em branco são ignoradas silenciosamente. Linhas inválidas (JSON
        malformado ou chave faltando) geram aviso e são puladas, ou interrompem a
        leitura se `strict=True`.

    Args:
        path: Caminho do arquivo JSONL.
        strict: Se `True`, levanta exceção na primeira linha inválida.

    Produz:
        Um `Record` por linha válida, na ordem do arquivo.

    Levanta:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se `strict=True` e alguma linha for inválida.
    """
    if not path.exists():
        raise FileNotFoundError(f"arquivo não encontrado: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Record.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError) as exc:
                message = f"{path.name}:{lineno} linha inválida ignorada ({exc})"
                if strict:
                    raise ValueError(message) from exc
                console.warn(message)


def read_flat_jsonl(path: Path, source: str, *, strict: bool = False) -> Iterator[Record]:
    """
    Descrição:
        Lê um arquivo JSONL no formato legado dos provedores — exatamente duas
        chaves, `{"text": ..., "label": ...}` — e o promove ao formato canônico,
        injetando `source` e duplicando `text` em `text_raw`.

        Usada apenas pelo Stage 1, que é o único ponto de contato com os
        diretórios de origem.

    Args:
        path: Caminho do arquivo `dd.jsonl` ou `ndd.jsonl` do provedor.
        source: Nome do provedor, gravado no campo `source`.
        strict: Se `True`, levanta exceção na primeira linha inválida.

    Produz:
        Um `Record` por linha válida, na ordem do arquivo.

    Levanta:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se `strict=True` e alguma linha for inválida.
    """
    if not path.exists():
        raise FileNotFoundError(f"arquivo não encontrado: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data["text"]
                label = data["label"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                message = f"{path.name}:{lineno} linha inválida ignorada ({exc})"
                if strict:
                    raise ValueError(message) from exc
                console.warn(message)
                continue

            if not isinstance(text, str) or not isinstance(label, str):
                console.warn(f"{path.name}:{lineno} tipos inesperados, linha ignorada")
                continue

            yield Record(text=text, text_raw=text, label=label, source=source)


def write_jsonl(records: Iterable[Record], path: Path) -> int:
    """
    Descrição:
        Escreve registros em um arquivo JSONL, sobrescrevendo o conteúdo anterior.
        O diretório pai é criado se necessário.

    Args:
        records: Iterável de registros a gravar.
        path: Caminho de destino.

    Retorna:
        Quantidade de registros gravados.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
            written += 1
    return written


def write_dicts_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    """
    Descrição:
        Escreve dicionários arbitrários em JSONL. Usada para artefatos auxiliares
        que não seguem o schema de `Record` — por exemplo o relatório de
        conflitos de rótulo do Stage 5.

    Args:
        rows: Iterável de dicionários serializáveis em JSON.
        path: Caminho de destino.

    Retorna:
        Quantidade de linhas gravadas.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def count_lines(path: Path) -> int:
    """
    Descrição:
        Conta as linhas não vazias de um arquivo de texto, sem carregá-lo na
        memória. Usada pelo manifest do snapshot e pelas verificações de sanidade.

    Args:
        path: Caminho do arquivo.

    Retorna:
        Número de linhas não vazias, ou 0 se o arquivo não existir.
    """
    if not path.exists():
        return 0
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total

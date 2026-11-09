"""E/S de JSONL e JSON, hashes de arquivo e escrita atômica."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Lê um JSONL linha a linha, ignorando linhas vazias.

    Uma última linha truncada (arquivo ainda sendo escrito por outro processo)
    é ignorada em vez de interromper a leitura; qualquer outra linha inválida
    levanta `ValueError` com o número da linha.
    """
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines):
                return
            raise ValueError(f"{path}:{number}: JSON inválido ({exc})") from exc


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Grava um JSONL de forma atômica (arquivo temporário + rename). Devolve o nº de linhas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return count


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Acrescenta linhas a um JSONL (cada linha numa única chamada de escrita)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def write_json(path: Path, payload: Any) -> None:
    """Grava JSON indentado de forma atômica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    """Lê um arquivo JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 do conteúdo do arquivo, em hexadecimal."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

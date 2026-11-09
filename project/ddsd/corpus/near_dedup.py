"""Deduplicação aproximada exata por similaridade de cosseno TF-IDF.

Problema: achar todos os pares com cosseno > t (t = 0,95) entre ~300 mil
frases. O produto completo X*X^t é inviável, porque palavras comuns ("o", "de",
"pra") aparecem em dezenas de milhares de frases e tornam o resultado quase denso.

Solução: filtragem por prefixo do algoritmo All-Pairs (Bayardo, Ma e Srikant,
2007), que é exata, sem falsos negativos:

1. Os termos de cada vetor (TF-IDF, norma L2) são ordenados do mais frequente
   para o mais raro.
2. Para cada vetor y, o maior prefixo de termos frequentes com norma
   ‖y_U‖ < t fica fora do índice; só o sufixo S_y (termos raros) é indexado.
3. Se cos(x, y) ≥ t, então x e y compartilham algum termo de S_y, porque a
   parte não indexada contribui no máximo ‖x‖*‖y_U‖ = ‖y_U‖ < t (os pesos
   TF-IDF são não negativos e ‖x‖ = 1).
4. Por isso, as entradas não nulas de X*S_yᵀ contêm todos os pares candidatos.
   O cosseno exato de cada candidato é recalculado e filtrado por > t.

Os pares aceitos formam grupos por união-busca (ligação simples). Um grupo com
uma só classe mantém um representante (o primeiro na ordem canônica do corpus).
Um grupo com as duas classes é conflito, e todos os seus membros saem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

#: Linhas de X processadas por bloco no produto X·Sᵀ (equilíbrio memória × velocidade).
_BLOCK_ROWS = 4096


@dataclass(frozen=True, slots=True)
class NearDuplicatePairs:
    """Pares (i < j) com cosseno acima do limiar e o respectivo cosseno."""

    left: np.ndarray
    right: np.ndarray
    cosine: np.ndarray


def tfidf_matrix(texts: Sequence[str], ngram_range: tuple[int, int]) -> sp.csr_matrix:
    """TF-IDF de palavras (tf sublinear, norma L2), incluindo palavras de uma letra."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,  # o texto já chega normalizado
        dtype=np.float64,
    )
    return vectorizer.fit_transform(texts).tocsr()


def _indexed_suffix(matrix: sp.csr_matrix, threshold: float) -> sp.csr_matrix:
    """Mantém de cada linha só os termos raros necessários para não perder pares ≥ limiar."""
    document_frequency = np.bincount(matrix.indices, minlength=matrix.shape[1])
    data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
    keep = np.zeros(data.shape[0], dtype=bool)
    limit = threshold * threshold
    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        if start == end:
            continue
        cols = indices[start:end]
        # Termos mais frequentes primeiro: são os candidatos a ficar fora do índice.
        order = np.argsort(-document_frequency[cols], kind="stable")
        squared = data[start:end][order] ** 2
        prefix_norm2 = np.cumsum(squared)
        # Fica fora o maior prefixo com norma² < t²; o resto (inclusive o termo que
        # cruza o limite) é indexado.
        unindexed = int(np.searchsorted(prefix_norm2, limit, side="left"))
        keep[start + order[unindexed:]] = True
    suffix = sp.csr_matrix(
        (np.where(keep, data, 0.0), indices.copy(), indptr.copy()), shape=matrix.shape
    )
    suffix.eliminate_zeros()
    return suffix


def find_near_duplicates(matrix: sp.csr_matrix, threshold: float) -> NearDuplicatePairs:
    """Todos os pares (i < j) com cosseno estritamente maior que `threshold`."""
    matrix = matrix.tocsr()
    suffix_t = _indexed_suffix(matrix, threshold).T.tocsc()
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    cosines: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], _BLOCK_ROWS):
        block = matrix[start : start + _BLOCK_ROWS]
        candidates = (block @ suffix_t).tocoo()
        rows = candidates.row.astype(np.int64) + start
        cols = candidates.col.astype(np.int64)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        if rows.size == 0:
            continue
        # Cada par pode surgir nas duas ordens; normaliza para (menor, maior) e deduplica.
        pairs = np.unique(np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], axis=1), axis=0)
        exact = np.asarray(matrix[pairs[:, 0]].multiply(matrix[pairs[:, 1]]).sum(axis=1)).ravel()
        accepted = exact > threshold
        lefts.append(pairs[accepted, 0])
        rights.append(pairs[accepted, 1])
        cosines.append(exact[accepted])
    if not lefts:
        empty = np.empty(0, dtype=np.int64)
        return NearDuplicatePairs(empty, empty, np.empty(0))
    left = np.concatenate(lefts)
    right = np.concatenate(rights)
    cosine = np.concatenate(cosines)
    # Blocos diferentes podem achar o mesmo par: mantém uma ocorrência.
    _, unique_index = np.unique(np.stack([left, right], axis=1), axis=0, return_index=True)
    return NearDuplicatePairs(left[unique_index], right[unique_index], cosine[unique_index])


def connected_groups(n_items: int, pairs: NearDuplicatePairs) -> np.ndarray:
    """Rótulo de grupo (raiz da união-busca) de cada item; isolados formam o próprio grupo."""
    parent = np.arange(n_items)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # compressão de caminho
            parent[i], i = root, parent[i]
        return root

    for a, b in zip(pairs.left.tolist(), pairs.right.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            # A raiz é sempre o menor índice: representante = primeiro na ordem canônica.
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n_items)])

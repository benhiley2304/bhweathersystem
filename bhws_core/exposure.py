"""Correlation-aware exposure: how many independent bets the book really is.

Issue 10 used the participation ratio of the signed-position correlation
matrix ("effective bets 4.2 of 9"). For position signs s_i and return
correlation matrix C, the signed matrix is S C S with S = diag(s). With
eigenvalues l_k of that matrix,

    effective_bets = (sum l_k)^2 / sum(l_k^2)

which is N for uncorrelated legs and 1 for legs that are one bet.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def effective_bets(corr: np.ndarray, signs: Sequence[int] | None = None) -> float:
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError("corr must be a non-empty square matrix")
    if signs is not None:
        s = np.diag(np.asarray(signs, dtype=float))
        c = s @ c @ s
    lam = np.clip(np.linalg.eigvalsh((c + c.T) / 2.0), 0.0, None)
    tot = lam.sum()
    return float(tot * tot / (lam ** 2).sum()) if tot > 0 else 0.0


def first_factor_share(corr: np.ndarray) -> float:
    lam = np.clip(np.linalg.eigvalsh(np.asarray(corr, dtype=float)), 0.0, None)
    return float(lam.max() / lam.sum()) if lam.sum() > 0 else 0.0


def clusters(corr: np.ndarray, names: Sequence[str], signs: Sequence[int],
             threshold: float = 0.5) -> List[List[str]]:
    """Group legs whose signed correlation exceeds `threshold` (single link).
    "Three dollar legs, one bet" falls out as a single cluster."""
    c = np.asarray(corr, dtype=float) * np.outer(signs, signs)
    n = len(names)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if c[i, j] >= threshold:
                parent[find(i)] = find(j)
    groups: Dict[int, List[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return sorted(groups.values(), key=len, reverse=True)

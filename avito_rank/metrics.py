"""NDCG по правилам соревнования (бинарный gain, дисконт 1/log2(i+1), ничьи — пессимистично,
лента без позитивов -> 1). Ленты должны лежать в массивах подряд, границы — offsets."""

import numpy as np

from .config import NDCG_CUTS


def ndcg_per_feed(y: np.ndarray, score: np.ndarray, offsets: np.ndarray, cut: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("score содержит NaN/inf")
    lens = np.diff(offsets)
    G = len(lens)
    gid = np.repeat(np.arange(G), lens)
    rank = np.arange(len(y)) - np.repeat(offsets[:-1], lens)
    disc = 1.0 / np.log2(rank + 2.0)
    if cut is not None:
        disc = np.where(rank < cut, disc, 0.0)
    # пессимистичные ничьи: при равных score сначала отрицательные (y=0), потом положительные
    order = np.lexsort((y, -score, gid))
    dcg = np.add.reduceat(y[order] * disc, offsets[:-1])
    ideal = np.lexsort((-y, gid))
    idcg = np.add.reduceat(y[ideal] * disc, offsets[:-1])
    out = np.ones(G)
    nz = idcg > 0
    out[nz] = dcg[nz] / idcg[nz]
    return out


def ndcg_report(y, score, offsets) -> dict[str, float]:
    res = {}
    for cut in NDCG_CUTS:
        name = "ndcg" if cut is None else f"ndcg@{cut}"
        res[name] = float(ndcg_per_feed(y, score, offsets, cut).mean())
    return res

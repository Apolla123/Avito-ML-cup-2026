"""Ансамбль: веса подбираются по OOF-предсказаниям на val, затем применяются к test.

Предсказания каждой модели переводятся в z-score внутри ленты (шкалы моделей разные,
а ранжирование идёт внутри ленты), затем берётся взвешенная сумма.
Для test предсказания фолдовых моделей усредняются (тоже после z-score).
"""

import json
from pathlib import Path

import numpy as np
from .log import logger

from .metrics import ndcg_report
from .scores_io import load_scores


def group_zscore(score: np.ndarray, feed_id: np.ndarray) -> np.ndarray:
    _, inv = np.unique(feed_id, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    s = score.astype(np.float64)
    mu = np.bincount(inv, weights=s) / n
    var = np.bincount(inv, weights=(s - mu[inv]) ** 2) / n
    return (s - mu[inv]) / (np.sqrt(var)[inv] + 1e-9)


def _keys(feed_id, item_id):
    return (feed_id.astype(np.int64) << 32) | item_id.astype(np.int64)


def _align(ref_keys: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(keys)
    pos = np.searchsorted(keys[order], ref_keys)
    pos = np.clip(pos, 0, len(keys) - 1)
    if not (keys[order][pos] == ref_keys).all():
        raise ValueError("Наборы (feed_id, item_id) у моделей не совпадают")
    return values[order][pos]


def collect_oof(model_dir: Path) -> dict:
    """Склеивает val_scores всех фолдов модели."""
    parts = [load_scores(p) for p in sorted(Path(model_dir).glob("fold*/val_scores.npz"))]
    if not parts:
        raise FileNotFoundError(f"В {model_dir} нет fold*/val_scores.npz")
    return {k: np.concatenate([p[k] for p in parts]) for k in ("feed_id", "item_id", "score", "y")}


def collect_test(model_dir: Path) -> dict:
    """Среднее по фолдам (после z-score внутри ленты) предсказаний test."""
    parts = [load_scores(p) for p in sorted(Path(model_dir).glob("fold*/test_scores.npz"))]
    if not parts:
        raise FileNotFoundError(f"В {model_dir} нет fold*/test_scores.npz")
    z = np.mean([group_zscore(p["score"], p["feed_id"]) for p in parts], axis=0)
    return {"feed_id": parts[0]["feed_id"], "item_id": parts[0]["item_id"], "score": z}


def search_weights(zs: list[np.ndarray], y: np.ndarray, offsets: np.ndarray, step: float = 0.05, rounds: int = 3):
    """Покоординатный поиск весов на симплексе по NDCG@30."""
    n = len(zs)
    w = np.full(n, 1.0 / n)

    def score(w):
        return ndcg_report(y, sum(wi * z for wi, z in zip(w, zs)), offsets)["ndcg@30"]

    best = score(w)
    for _ in range(rounds):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = w.copy()
                d = min(step, cand[j])
                cand[i] += d
                cand[j] -= d
                s = score(cand)
                if s > best + 1e-9:
                    best, w, improved = s, cand, True
        if not improved:
            break
    return w / w.sum(), best


def blend(run_dirs: list[str], out_json: Path):
    oofs = [collect_oof(Path(d)) for d in run_dirs]
    ref = oofs[0]
    keys = _keys(ref["feed_id"], ref["item_id"])
    order = np.lexsort((ref["item_id"], ref["feed_id"]))
    ref_keys = keys[order]
    feed = ref["feed_id"][order]
    y = ref["y"][order]
    uniq, start = np.unique(feed, return_index=True)
    offsets = np.append(start, len(feed))
    zs = []
    for d, o in zip(run_dirs, oofs):
        s = _align(ref_keys, _keys(o["feed_id"], o["item_id"]), o["score"])
        z = group_zscore(s, feed)
        zs.append(z)
        logger.info("{}: OOF {}", d, ndcg_report(y, z, offsets))
    w, best = search_weights(zs, y, offsets)
    logger.info("веса {} -> OOF ndcg@30 {:.5f}", np.round(w, 3).tolist(), best)
    res = {"runs": run_dirs, "weights": w.tolist(),
           "oof": ndcg_report(y, sum(wi * z for wi, z in zip(w, zs)), offsets)}
    Path(out_json).write_text(json.dumps(res, indent=2))
    return w


def blend_test(run_dirs: list[str], weights: list[float]) -> dict:
    tests = [collect_test(Path(d)) for d in run_dirs]
    ref = tests[0]
    rk = _keys(ref["feed_id"], ref["item_id"])
    total = np.zeros(len(rk))
    for wi, t in zip(weights, tests):
        total += wi * _align(rk, _keys(t["feed_id"], t["item_id"]), t["score"])
    return {"feed_id": ref["feed_id"], "item_id": ref["item_id"], "score": total.astype(np.float32)}

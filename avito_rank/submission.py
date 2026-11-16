"""Сборка и проверка сабмита по sample_submission.parquet."""

import numpy as np


def _keys(feed_id, item_id):
    return (np.asarray(feed_id, np.int64) << 32) | np.asarray(item_id, np.int64)


def align_scores(sample_feed, sample_item, pred: dict) -> np.ndarray:
    """Оценка для каждой пары шаблона; ошибка при пропусках/дубликатах."""
    pk = _keys(pred["feed_id"], pred["item_id"])
    if len(np.unique(pk)) != len(pk):
        raise ValueError("В предсказаниях есть повторяющиеся пары (feed_id, item_id)")
    sk = _keys(sample_feed, sample_item)
    if len(np.unique(sk)) != len(sk):
        raise ValueError("В sample_submission есть повторяющиеся пары")
    order = np.argsort(pk)
    pos = np.clip(np.searchsorted(pk[order], sk), 0, len(pk) - 1)
    found = pk[order][pos] == sk
    if not found.all():
        raise ValueError(f"Нет предсказаний для {int((~found).sum())} пар из sample_submission")
    score = np.asarray(pred["score"])[order][pos].astype(np.float32)
    if not np.isfinite(score).all():
        raise ValueError("В score есть NaN/inf")
    return score


def write_submission(sample_path, pred: dict, out_path) -> None:
    import polars as pl

    s = pl.read_parquet(sample_path)
    score = align_scores(s["feed_id"].to_numpy(), s["item_id"].to_numpy(), pred)
    out = s.select(
        pl.col("feed_id").cast(pl.Int32), pl.col("item_id").cast(pl.Int32),
    ).with_columns(pl.Series("score", score, dtype=pl.Float32))
    out.write_parquet(out_path, compression="zstd")
    validate_submission(sample_path, out_path)


def validate_submission(sample_path, sub_path) -> None:
    import polars as pl

    s = pl.read_parquet(sample_path)
    d = pl.read_parquet(sub_path)
    assert d.columns == ["feed_id", "item_id", "score"], d.columns
    assert d.schema["feed_id"] == pl.Int32 and d.schema["item_id"] == pl.Int32
    assert d.schema["score"] == pl.Float32
    assert len(d) == len(s), (len(d), len(s))
    assert d["score"].null_count() == 0 and d["score"].is_finite().all()
    a = _keys(s["feed_id"].to_numpy(), s["item_id"].to_numpy())
    b = _keys(d["feed_id"].to_numpy(), d["item_id"].to_numpy())
    assert len(np.unique(b)) == len(b), "дубликаты пар"
    assert np.array_equal(np.sort(a), np.sort(b)), "наборы пар не совпадают с sample_submission"

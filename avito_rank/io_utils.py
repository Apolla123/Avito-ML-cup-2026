"""Чтение parquet-лент и memmap-эмбеддингов (формат соревнования)."""

import json
from pathlib import Path

import numpy as np

WANTED_COLS = [
    "feed_id", "timestamp", "user_id", "category_id", "f1", "f2", "f3", "f4", "f5",
    "pos", "target", "item_id__embedding_idx", "item_id", "user_embedding",
    "history__item_id", "history__category_id", "history__event_type",
    "history__event_source", "history__item_id__embedding_idx",
]


def list_parts(split_dir) -> list[Path]:
    return sorted((Path(split_dir) / "impressions").glob("part-*.parquet"))


def open_embeddings(split_dir) -> np.memmap:
    d = Path(split_dir) / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(
        d / "data.memmap", dtype=meta["dtype"], mode="r",
        shape=tuple(meta["shape"]), order=meta["order"],
    )


def read_part_rows(path):
    """Возвращает итератор по лентам (dict со списками) одного parquet-файла."""
    import polars as pl

    lf = pl.scan_parquet(str(path))
    names = lf.collect_schema().names()
    cols = [c for c in WANTED_COLS if c in names]
    return lf.select(cols).collect().iter_rows(named=True)

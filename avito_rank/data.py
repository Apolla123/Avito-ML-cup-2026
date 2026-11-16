"""Кэш признаков на диске и контейнер SplitData (ленты лежат подряд, границы в feed_off)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import feature_names

ARRAY_KEYS = ["X", "E", "cat", "y", "item_id", "pos"]
FEED_KEYS = ["feed_id_u", "user_id_u", "ts_u", "U"]


@dataclass
class SplitData:
    X: np.ndarray              # (N, F) float32
    E: np.ndarray              # (N, 128) int8, эмбеддинги кандидатов (0 если нет)
    cat: np.ndarray            # (N,) int16
    y: np.ndarray              # (N,) int8, -1 если меток нет
    item_id: np.ndarray        # (N,) int32
    pos: np.ndarray            # (N,) int16, только train
    feed_off: np.ndarray       # (G+1,) int64
    feed_id_u: np.ndarray      # (G,)
    user_id_u: np.ndarray
    ts_u: np.ndarray
    U: np.ndarray              # (G, 128) int8
    feed_w: np.ndarray = field(default=None)  # (G,) вес ленты при обучении

    def __post_init__(self):
        if self.feed_w is None:
            self.feed_w = np.ones(len(self.feed_id_u), np.float32)

    @property
    def n_feeds(self) -> int:
        return len(self.feed_id_u)

    @property
    def n_rows(self) -> int:
        return len(self.X)

    @property
    def lens(self) -> np.ndarray:
        return np.diff(self.feed_off)

    @property
    def row_feed_id(self) -> np.ndarray:
        return np.repeat(self.feed_id_u, self.lens)

    @property
    def row_weight(self) -> np.ndarray:
        return np.repeat(self.feed_w, self.lens).astype(np.float32)

    @property
    def has_target(self) -> bool:
        return bool(len(self.y)) and bool((self.y >= 0).all())

    def take(self, feed_idx) -> "SplitData":
        """Подвыборка лент (порядок сохраняется как в feed_idx)."""
        idx = np.asarray(feed_idx, dtype=np.int64)
        lens = self.lens[idx]
        starts = self.feed_off[:-1][idx]
        new_off = np.zeros(len(idx) + 1, np.int64)
        new_off[1:] = np.cumsum(lens)
        rows = np.arange(new_off[-1]) + np.repeat(starts - new_off[:-1], lens)
        return SplitData(
            X=self.X[rows], E=self.E[rows], cat=self.cat[rows], y=self.y[rows],
            item_id=self.item_id[rows], pos=self.pos[rows], feed_off=new_off,
            feed_id_u=self.feed_id_u[idx], user_id_u=self.user_id_u[idx], ts_u=self.ts_u[idx],
            U=self.U[idx], feed_w=self.feed_w[idx],
        )

    def with_weight(self, w: float) -> "SplitData":
        out = self.take(np.arange(self.n_feeds))
        out.feed_w = np.full(self.n_feeds, w, np.float32)
        return out

    @staticmethod
    def concat(parts: list["SplitData"]) -> "SplitData":
        parts = [p for p in parts if p.n_feeds]
        offs, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            offs.append(p.feed_off[1:] + base)
            base += p.n_rows
        cat = lambda k: np.concatenate([getattr(p, k) for p in parts])  # noqa: E731
        return SplitData(
            X=cat("X"), E=cat("E"), cat=cat("cat"), y=cat("y"), item_id=cat("item_id"), pos=cat("pos"),
            feed_off=np.concatenate(offs), feed_id_u=cat("feed_id_u"), user_id_u=cat("user_id_u"),
            ts_u=cat("ts_u"), U=cat("U"), feed_w=cat("feed_w"),
        )


def save_part(path: Path, arrays: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")  # не подпадает под glob part-*.npz
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.rename(path)


def load_split(cache_dir, split: str, max_feeds: int = 0) -> SplitData:
    files = sorted((Path(cache_dir) / split).glob("part-*.npz"))
    if not files:
        raise FileNotFoundError(f"Нет кэша признаков для {split}: сначала запустите `features`.")
    names = feature_names()
    parts, total = [], 0
    for f in files:
        z = np.load(f)
        if z["X"].shape[1] != len(names):
            raise ValueError(f"{f}: число признаков не совпадает с кодом — пересоберите кэш.")
        p = SplitData(**{k: z[k] for k in ARRAY_KEYS + FEED_KEYS}, feed_off=z["feed_off"])
        parts.append(p)
        total += p.n_feeds
        if max_feeds and total >= max_feeds:
            break
    sd = SplitData.concat(parts)
    if max_feeds and sd.n_feeds > max_feeds:
        sd = sd.take(np.arange(max_feeds))
    return sd


def user_fold(user_id: np.ndarray, nfolds: int) -> np.ndarray:
    """Фолд пользователя: хэш user_id, чтобы один пользователь не попадал в train и eval."""
    h = (user_id.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return (h % np.uint64(nfolds)).astype(np.int64)


def make_sets(cache_dir, fold: int, nfolds: int, val_weight: float, max_train_feeds: int = 0):
    """(train, eval, test).

    fold == -1: обучаемся только на train, оцениваемся на всём val.
    fold >= 0 : train + (val без fold)*val_weight; оценка на val[fold].
    Val собран как test (равномерно перемешанные выдачи), а train смещён позицией —
    поэтому val-ленты в обучении получают повышенный вес.
    """
    train = load_split(cache_dir, "train", max_train_feeds)
    val = load_split(cache_dir, "val")
    test = load_split(cache_dir, "test")
    if fold < 0:
        return train, val, test
    f = user_fold(val.user_id_u, nfolds)
    tr_val = val.take(np.flatnonzero(f != fold)).with_weight(val_weight)
    ev = val.take(np.flatnonzero(f == fold))
    return SplitData.concat([train, tr_val]), ev, test

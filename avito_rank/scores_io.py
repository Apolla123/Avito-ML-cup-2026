"""Единый формат предсказаний модели: npz с feed_id, item_id, score, y, feed_off."""

from pathlib import Path

import numpy as np

from .data import SplitData


def save_scores(path: Path, sd: SplitData, score: np.ndarray) -> None:
    np.savez(path, feed_id=sd.row_feed_id, item_id=sd.item_id, score=np.asarray(score, np.float32),
             y=sd.y, feed_off=sd.feed_off)


def load_scores(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}

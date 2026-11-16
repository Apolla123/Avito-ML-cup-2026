"""Синтетический датасет в формате соревнования — для проверки пайплайна без реальных данных.

Сигнал: релевантность зависит от user·item, f1, совпадения категории с историей и повторного
просмотра товара. В train клики смещены исходной позицией (как у реального ранжирования).
"""

import json
from pathlib import Path

import numpy as np

from .config import EMB_DIM


def make_world(seed=0, n_items=3000, n_users=400, n_cats=20):
    rng = np.random.default_rng(seed)
    item_vec = rng.integers(-40, 40, size=(n_items, EMB_DIM)).astype(np.int8)
    cat_of_item = rng.integers(1, n_cats + 1, size=n_items)
    return {"rng": rng, "item_vec": item_vec, "cat": cat_of_item, "n_users": n_users, "n_items": n_items}


def make_feeds(world, n_feeds, split, feed_id0=0, ts0=1_000_000):
    """Список лент-словарей (как строки parquet). Индексы эмбеддингов = item_id - 1."""
    rng, iv, cat_of = world["rng"], world["item_vec"], world["cat"]
    feeds = []
    for i in range(n_feeds):
        u = int(rng.integers(1, world["n_users"] + 1))
        urng = np.random.default_rng(u)
        user_vec = urng.integers(-40, 40, size=EMB_DIM).astype(np.int8)
        fav = urng.integers(1, 21, size=3)
        H = int(rng.integers(0, 60))
        h_items = rng.integers(1, world["n_items"] + 1, size=H)
        # история смещена к любимым категориям
        pick = rng.random(H) < 0.6
        for j in np.flatnonzero(pick):
            cands = np.flatnonzero(np.isin(cat_of, fav)) + 1
            h_items[j] = rng.choice(cands)
        M = int(rng.integers(8, 60))
        items = rng.choice(world["n_items"], size=M, replace=False) + 1
        f = rng.random((M, 5)).astype(np.float32)
        sim = iv[items - 1].astype(np.float32) @ user_vec.astype(np.float32) / (40 * 40 * EMB_DIM / 3)
        in_fav = np.isin(cat_of[items - 1], fav).astype(np.float32)
        seen = np.isin(items, h_items).astype(np.float32)
        logit = -3.0 + 2.0 * sim + 1.2 * f[:, 0] + 1.0 * in_fav + 1.0 * seen
        pos = np.sort(rng.permutation(M)) if split == "train" else None
        if split == "train":  # положение влияет на клик
            logit = logit + 1.5 / np.log2(np.arange(M) + 2.0)
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(M) < p).astype(np.int8)
        if y.sum() == 0:
            y[int(np.argmax(p))] = 1
        order = rng.permutation(M) if split != "train" else np.arange(M)
        miss = rng.random(M) < 0.05
        emb_idx = np.where(miss, -1, items - 1)
        feed = {
            "feed_id": feed_id0 + i, "timestamp": ts0 + i, "user_id": u,
            "category_id": cat_of[items - 1].astype(np.int8)[order].tolist(),
            "f1": f[order, 0].tolist(), "f2": f[order, 1].tolist(), "f3": f[order, 2].tolist(),
            "f4": f[order, 3].tolist(), "f5": f[order, 4].tolist(),
            "item_id__embedding_idx": emb_idx[order].tolist(),
            "item_id": items[order].tolist(),
            "user_embedding": user_vec.tolist(),
            "history__item_id": h_items.tolist(),
            "history__category_id": cat_of[h_items - 1].astype(np.int8).tolist() if H else [],
            "history__event_type": rng.integers(1, 5, size=H).tolist(),
            "history__event_source": rng.integers(1, 3, size=H).tolist(),
            "history__item_id__embedding_idx": (h_items - 1).tolist(),
        }
        if split != "test":
            feed["target"] = y[order].tolist()
        if split == "train":
            feed["pos"] = pos[order].tolist() if pos is not None else None
        feeds.append(feed)
    return feeds


def write_competition(out_dir, n_train=600, n_val=200, n_test=200, seed=0, parts=2):
    """Пишет competition/{train,val,test} + sample_submission.parquet (нужен polars)."""
    import polars as pl

    out = Path(out_dir)
    world = make_world(seed)
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    fid0 = 0
    sub_rows = []
    for split, n in sizes.items():
        feeds = make_feeds(world, n, split, feed_id0=fid0, ts0=1_000_000 + fid0)
        fid0 += n
        d = out / split
        (d / "impressions").mkdir(parents=True, exist_ok=True)
        (d / "item_embeddings").mkdir(parents=True, exist_ok=True)
        iv = world["item_vec"]
        iv.tofile(d / "item_embeddings" / "data.memmap")
        (d / "item_embeddings" / "data.meta").write_text(
            json.dumps({"dtype": "int8", "shape": list(iv.shape), "order": "C"}))
        schema = {
            "feed_id": pl.Int32, "timestamp": pl.Int32, "user_id": pl.Int32,
            "category_id": pl.List(pl.Int8), "f1": pl.List(pl.Float32), "f2": pl.List(pl.Float32),
            "f3": pl.List(pl.Float32), "f4": pl.List(pl.Float32), "f5": pl.List(pl.Float32),
            "item_id__embedding_idx": pl.List(pl.Int32), "item_id": pl.List(pl.Int32),
            "user_embedding": pl.List(pl.Int8), "history__item_id": pl.List(pl.Int32),
            "history__category_id": pl.List(pl.Int8), "history__event_type": pl.List(pl.Int8),
            "history__event_source": pl.List(pl.Int8),
            "history__item_id__embedding_idx": pl.List(pl.Int32),
        }
        if split != "test":
            schema["target"] = pl.List(pl.Int8)
        if split == "train":
            schema["pos"] = pl.List(pl.Int16)
        chunk = max(1, -(-len(feeds) // parts))
        for k in range(parts):
            sub = feeds[k * chunk:(k + 1) * chunk]
            if not sub:
                continue
            df = pl.DataFrame({c: [f[c] for f in sub] for c in schema}, schema=schema)
            df.write_parquet(d / "impressions" / f"part-{k:05d}.parquet", compression="zstd")
        (d / "impressions" / "_SUCCESS").touch()
        if split == "test":
            for f in feeds:
                for it in f["item_id"]:
                    sub_rows.append((f["feed_id"], it))
    pl.DataFrame(
        {"feed_id": [r[0] for r in sub_rows], "item_id": [r[1] for r in sub_rows],
         "score": [0.0] * len(sub_rows)},
        schema={"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32},
    ).write_parquet(out / "sample_submission.parquet")

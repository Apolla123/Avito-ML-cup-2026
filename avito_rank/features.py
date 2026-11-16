"""Признаки кандидатов. Чистый numpy: одна функция на ленту, без polars/torch.

Для каждого кандидата считаются:
  * исходные f1..f5, сходство эмбеддингов user<->item;
  * внутриленточные относительные признаки (ранг-перцентиль, z-score, отставание от максимума);
  * совпадения с историей: тот же товар / та же категория (счётчики, доли, давность, по типам событий);
  * сходство эмбеддинга кандидата с эмбеддингами истории (max/mean/top3/недавние/взвешенные/по типам событий);
  * размер ленты, длина истории, доля категории в ленте.
Пропуски эмбеддингов дают NaN (деревья умеют, для NN есть индикаторы).
"""

import warnings
from functools import lru_cache

import numpy as np

from .config import EMB_DIM, EVENT_SIM_TYPES, EVENT_TYPES, HIST_DECAY, HIST_RECENT, NUM_FEATS


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def gather(E, idx) -> tuple[np.ndarray, np.ndarray]:
    """float32-векторы по индексам; -1 -> нулевая строка + маска missing."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    miss = idx < 0
    if len(idx) == 0:
        return np.zeros((0, EMB_DIM), np.float32), miss
    v = np.asarray(E[np.maximum(idx, 0)]).astype(np.float32)
    v[miss] = 0
    return v, miss


def rank_pct(x: np.ndarray) -> np.ndarray:
    """Ранг-перцентиль внутри ленты в [0,1] (ничьи усредняются; NaN считаются худшими)."""
    m = len(x)
    if m < 2:
        return np.zeros(m, np.float32)
    v = np.where(np.isnan(x), -np.inf, x).astype(np.float64)
    s = np.sort(v)
    lo = np.searchsorted(s, v, "left")
    hi = np.searchsorted(s, v, "right")
    return ((lo + hi - 1) / 2.0 / (m - 1)).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return np.full(len(x), np.nan, np.float32)
    mu, sd = x[ok].mean(), x[ok].std()
    return ((x - mu) / (sd + 1e-6)).astype(np.float32)


def _first_idx(match: np.ndarray) -> np.ndarray:
    """Индекс первого (самого свежего) совпадения в истории или NaN."""
    if match.shape[1] == 0:
        return np.full(match.shape[0], np.nan, np.float32)
    any_ = match.any(1)
    return np.where(any_, match.argmax(1), np.nan).astype(np.float32)


def _arr(feed, key, dtype):
    v = feed.get(key)
    if v is None:
        return np.zeros(0, dtype)
    return np.asarray(v, dtype=dtype)


def feature_columns(feed: dict, E) -> dict[str, np.ndarray]:
    """Словарь name -> массив длины M (порядок вставки = порядок колонок)."""
    item = _arr(feed, "item_id", np.int64)
    M = len(item)
    cat = _arr(feed, "category_id", np.int64)
    F = np.stack([_arr(feed, k, np.float32) for k in NUM_FEATS], 1)
    user = np.asarray(feed["user_embedding"], np.float32)

    cv, cmiss = gather(E, feed["item_id__embedding_idx"])
    cn = _l2(cv)
    un = _l2(user[None])[0]

    c: dict[str, np.ndarray] = {}
    for j, k in enumerate(NUM_FEATS):
        c[k] = F[:, j]
    c["emb_missing"] = cmiss.astype(np.float32)
    c["emb_dot"] = np.where(cmiss, np.nan, cv @ user).astype(np.float32)
    c["emb_cos"] = np.where(cmiss, np.nan, cn @ un).astype(np.float32)
    c["emb_norm"] = np.where(cmiss, np.nan, np.linalg.norm(cv, axis=1)).astype(np.float32)

    # ---- похожесть на центроид ленты (разнообразие/«типичность» кандидата)
    if (~cmiss).any():
        cen = _l2(cn[~cmiss].mean(0, keepdims=True))[0]
        c["cand_centroid_cos"] = np.where(cmiss, np.nan, cn @ cen).astype(np.float32)
    else:
        c["cand_centroid_cos"] = np.full(M, np.nan, np.float32)

    # ---- категории внутри ленты
    uc, inv, cnt = np.unique(cat, return_inverse=True, return_counts=True)
    c["cat_cnt_in_feed"] = cnt[inv].astype(np.float32)
    c["cat_share_in_feed"] = (cnt[inv] / max(M, 1)).astype(np.float32)

    # ---- история
    h_item = _arr(feed, "history__item_id", np.int64)
    H = len(h_item)
    h_cat = _arr(feed, "history__category_id", np.int64)
    h_evt = _arr(feed, "history__event_type", np.int64)
    h_idx = _arr(feed, "history__item_id__embedding_idx", np.int64)
    K = len(EVENT_TYPES)
    w = np.exp(-np.arange(H) / HIST_DECAY).astype(np.float32)
    wsum = max(float(w.sum()), 1e-6)
    evt_oh = np.zeros((H, K), np.float32)
    for k, t in enumerate(EVENT_TYPES):
        evt_oh[:, k] = h_evt == t

    c["feed_size"] = np.full(M, M, np.float32)
    c["hist_len"] = np.full(M, H, np.float32)

    # тот же товар в истории
    mi = (item[:, None] == h_item[None, :]) & (item[:, None] > 0) & (h_item[None, :] > 0)
    mif = mi.astype(np.float32)
    c["hist_item_cnt"] = mif.sum(1)
    c["hist_item_wcnt"] = mif @ w if H else np.zeros(M, np.float32)
    c["hist_item_first"] = _first_idx(mi)
    ie = mif @ evt_oh if H else np.zeros((M, K), np.float32)
    for k, t in enumerate(EVENT_TYPES):
        c[f"hist_item_evt{t}"] = ie[:, k]

    # та же категория в истории
    mc = (cat[:, None] == h_cat[None, :]) & (h_cat[None, :] > 0)
    mcf = mc.astype(np.float32)
    c["hist_cat_cnt"] = mcf.sum(1)
    c["hist_cat_share"] = c["hist_cat_cnt"] / max(H, 1)
    c["hist_cat_wshare"] = (mcf @ w / wsum) if H else np.zeros(M, np.float32)
    c["hist_cat_first"] = _first_idx(mc)
    ce = mcf @ evt_oh if H else np.zeros((M, K), np.float32)
    for k, t in enumerate(EVENT_TYPES):
        c[f"hist_cat_evt{t}"] = ce[:, k]
    nz = h_cat[h_cat > 0]
    if len(nz):
        vals, cts = np.unique(nz, return_counts=True)
        c["cat_is_top_hist"] = (cat == vals[cts.argmax()]).astype(np.float32)
    else:
        c["cat_is_top_hist"] = np.zeros(M, np.float32)

    # сходство эмбеддингов кандидата с эмбеддингами истории
    valid = h_idx >= 0
    hv, _ = gather(E, h_idx[valid])
    Hv = len(hv)
    c["hist_emb_frac"] = np.full(M, Hv / max(H, 1), np.float32)
    nan = np.full(M, np.nan, np.float32)
    names = ["hist_sim_max", "hist_sim_mean", "hist_sim_top3", "hist_sim_recent", "hist_sim_wmean", "hist_sim_cent"]
    sims = {n: nan.copy() for n in names}
    sims_evt = {t: nan.copy() for t in EVENT_SIM_TYPES}
    if Hv > 0:
        hn = _l2(hv)
        wv = w[valid]
        S = cn @ hn.T  # (M, Hv)
        sims["hist_sim_max"] = S.max(1)
        sims["hist_sim_mean"] = S.mean(1)
        if Hv >= 3:
            sims["hist_sim_top3"] = np.partition(S, Hv - 3, axis=1)[:, -3:].mean(1)
        else:
            sims["hist_sim_top3"] = S.max(1)
        sims["hist_sim_recent"] = S[:, :HIST_RECENT].mean(1)
        sims["hist_sim_wmean"] = S @ wv / max(float(wv.sum()), 1e-6)
        cen_h = _l2((hn * wv[:, None]).sum(0, keepdims=True))[0]
        sims["hist_sim_cent"] = cn @ cen_h
        ev = h_evt[valid]
        for t in EVENT_SIM_TYPES:
            msk = ev == t
            if msk.any():
                sims_evt[t] = S[:, msk].mean(1)
    for n in names:
        c[n] = np.where(cmiss, np.nan, sims[n]).astype(np.float32)
    for t in EVENT_SIM_TYPES:
        c[f"hist_sim_evt{t}"] = np.where(cmiss, np.nan, sims_evt[t]).astype(np.float32)

    # ---- относительные признаки внутри ленты
    rel = [*NUM_FEATS, "emb_dot", "emb_cos", "hist_sim_max", "hist_sim_wmean", "hist_sim_recent",
           "hist_sim_cent", "hist_cat_wshare", "hist_item_wcnt"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in rel:
            x = c[k]
            c[f"{k}_rpct"] = rank_pct(x)
            c[f"{k}_z"] = zscore(x)
            mx = np.nanmax(x) if np.isfinite(x).any() else np.nan
            c[f"{k}_dmax"] = (x - mx).astype(np.float32)
    return c


@lru_cache(maxsize=1)
def feature_names() -> tuple[str, ...]:
    rng = np.random.default_rng(0)
    E = rng.integers(-128, 127, size=(8, EMB_DIM), dtype=np.int8)
    feed = {
        "item_id": [1, 2, 3], "category_id": [1, 2, 2],
        "f1": [0.1, 0.2, 0.3], "f2": [0.1, 0.2, 0.3], "f3": [0.1, 0.2, 0.3],
        "f4": [0.1, 0.2, 0.3], "f5": [0.1, 0.2, 0.3],
        "item_id__embedding_idx": [0, 1, -1], "user_embedding": [1] * EMB_DIM,
        "history__item_id": [1, 5], "history__category_id": [1, 3], "history__event_type": [1, 2],
        "history__event_source": [1, 1], "history__item_id__embedding_idx": [2, 3],
    }
    return tuple(feature_columns(feed, E).keys())


def featurize_feed(feed: dict, E) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает X (M, F) float32 и эмбеддинги кандидатов (M, 128) int8."""
    cols = feature_columns(feed, E)
    X = np.stack(list(cols.values()), axis=1).astype(np.float32)
    cv, _ = gather(E, feed["item_id__embedding_idx"])
    return X, cv.astype(np.int8)


def featurize_rows(rows, E) -> dict[str, np.ndarray]:
    """Пакет лент -> плоские массивы для кэша (см. data.SplitData)."""
    Xs, Es, cats, ys, items, poss, sizes = [], [], [], [], [], [], []
    fid, uid, ts, U = [], [], [], []
    has_target = False
    for feed in rows:
        X, Ei = featurize_feed(feed, E)
        M = len(X)
        Xs.append(X)
        Es.append(Ei)
        cats.append(_arr(feed, "category_id", np.int16))
        items.append(_arr(feed, "item_id", np.int32))
        if feed.get("target") is not None:
            has_target = True
            ys.append(_arr(feed, "target", np.int8))
        else:
            ys.append(np.full(M, -1, np.int8))
        p = feed.get("pos")
        poss.append(
            np.nan_to_num(np.asarray(p, dtype=float), nan=-1).astype(np.int16) if p is not None
            else np.full(M, -1, np.int16)
        )
        sizes.append(M)
        fid.append(feed["feed_id"])
        uid.append(feed["user_id"])
        ts.append(feed["timestamp"])
        U.append(np.asarray(feed["user_embedding"], np.int8))
    off = np.zeros(len(sizes) + 1, np.int64)
    off[1:] = np.cumsum(sizes)
    n_feat = len(feature_names())
    return {
        "X": np.concatenate(Xs) if Xs else np.zeros((0, n_feat), np.float32),
        "E": np.concatenate(Es) if Es else np.zeros((0, EMB_DIM), np.int8),
        "cat": np.concatenate(cats) if cats else np.zeros(0, np.int16),
        "y": np.concatenate(ys) if ys else np.zeros(0, np.int8),
        "item_id": np.concatenate(items) if items else np.zeros(0, np.int32),
        "pos": np.concatenate(poss) if poss else np.zeros(0, np.int16),
        "feed_off": off,
        "feed_id_u": np.asarray(fid, np.int64),
        "user_id_u": np.asarray(uid, np.int64),
        "ts_u": np.asarray(ts, np.int64),
        "U": np.stack(U) if U else np.zeros((0, EMB_DIM), np.int8),
        "has_target": np.asarray(has_target),
    }

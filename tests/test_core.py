"""Проверки без polars/catboost/torch: признаки, метрика, кэш, ансамбль, сабмит, сигнал на синтетике."""

import numpy as np
import pytest

from avito_rank.config import EMB_DIM, EVENT_TYPES
from avito_rank.data import SplitData, load_split, make_sets, save_part, user_fold
from avito_rank.features import feature_names, featurize_feed, featurize_rows, rank_pct
from avito_rank.metrics import ndcg_per_feed
from avito_rank.synth import make_feeds, make_world


def brute_ndcg(y, s, cut=None):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0:
        return 1.0
    # пессимистичные ничьи
    order = sorted(range(len(y)), key=lambda i: (-s[i], y[i]))
    disc = lambda k: 1 / np.log2(k + 2)  # noqa: E731
    n = len(y) if cut is None else min(cut, len(y))
    dcg = sum(y[order[k]] * disc(k) for k in range(n))
    ideal = sorted(y, reverse=True)
    idcg = sum(ideal[k] * disc(k) for k in range(n))
    return dcg / idcg


def test_ndcg_matches_bruteforce_with_ties():
    rng = np.random.default_rng(1)
    ys, ss, off = [], [], [0]
    for _ in range(50):
        m = int(rng.integers(5, 40))
        ys.append((rng.random(m) < 0.15).astype(int))
        ss.append(rng.integers(0, 4, m).astype(float))  # много ничьих
        off.append(off[-1] + m)
    y, s, off = np.concatenate(ys), np.concatenate(ss), np.array(off)
    for cut in (None, 10, 30):
        got = ndcg_per_feed(y, s, off, cut)
        exp = [brute_ndcg(a, b, cut) for a, b in zip(ys, ss)]
        np.testing.assert_allclose(got, exp, atol=1e-12)


def test_ndcg_perfect_and_empty():
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    off = np.array([0, 5, 10])
    assert ndcg_per_feed(y, np.array([1, 9, 2, 3, 4, 1, 2, 3, 4, 5.0]), off).tolist() == [1.0, 1.0]
    # лента без позитивов -> 1
    assert ndcg_per_feed(y, np.zeros(10), off)[1] == 1.0


def _world_feeds(n=30, split="val", seed=0):
    w = make_world(seed)
    return w, make_feeds(w, n, split)


def test_feature_names_unique_and_shape():
    names = feature_names()
    assert len(names) == len(set(names))
    w, feeds = _world_feeds(5)
    X, E = featurize_feed(feeds[0], w["item_vec"])
    assert X.shape == (len(feeds[0]["item_id"]), len(names)) and E.shape[1] == EMB_DIM
    assert X.dtype == np.float32


def test_features_vs_bruteforce():
    w, feeds = _world_feeds(25)
    names = list(feature_names())
    ix = {n: i for i, n in enumerate(names)}
    iv = w["item_vec"].astype(np.float32)
    for feed in feeds:
        X, _ = featurize_feed(feed, w["item_vec"])
        items, cats = feed["item_id"], feed["category_id"]
        h_items, h_cats, h_evt = feed["history__item_id"], feed["history__category_id"], feed["history__event_type"]
        H = len(h_items)
        for i, (it, ct) in enumerate(zip(items, cats)):
            assert X[i, ix["hist_item_cnt"]] == sum(1 for h in h_items if h == it)
            assert X[i, ix["hist_cat_cnt"]] == sum(1 for h in h_cats if h == ct)
            for t in EVENT_TYPES:
                assert X[i, ix[f"hist_item_evt{t}"]] == sum(1 for h, e in zip(h_items, h_evt) if h == it and e == t)
            if feed["item_id__embedding_idx"][i] >= 0 and H:
                v = iv[feed["item_id__embedding_idx"][i]]
                sims = [v @ iv[h - 1] / (np.linalg.norm(v) * np.linalg.norm(iv[h - 1])) for h in h_items]
                assert X[i, ix["hist_sim_max"]] == pytest.approx(max(sims), abs=1e-4)
                assert X[i, ix["hist_sim_mean"]] == pytest.approx(np.mean(sims), abs=1e-4)
            elif feed["item_id__embedding_idx"][i] < 0:
                assert np.isnan(X[i, ix["emb_cos"]]) and X[i, ix["emb_missing"]] == 1
        assert X[:, ix["f1_rpct"]].min() >= 0 and X[:, ix["f1_rpct"]].max() <= 1
        assert (X[:, ix["feed_size"]] == len(items)).all()


def test_empty_history_ok():
    w, feeds = _world_feeds(40)
    f = next(f for f in feeds if len(f["history__item_id"]) == 0) if any(
        len(f["history__item_id"]) == 0 for f in feeds) else dict(feeds[0], **{
            "history__item_id": [], "history__category_id": [], "history__event_type": [],
            "history__event_source": [], "history__item_id__embedding_idx": []})
    X, _ = featurize_feed(f, w["item_vec"])
    assert np.isfinite(X[:, list(feature_names()).index("hist_len")]).all()


def test_rank_pct_ties():
    r = rank_pct(np.array([1.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(r, [1 / 6, 1 / 6, 2 / 3, 1.0], atol=1e-6)


def _cache(tmp_path, n_tr=120, n_val=60, n_te=30):
    w = make_world(3)
    for split, n, f0 in (("train", n_tr, 0), ("val", n_val, 10_000), ("test", n_te, 20_000)):
        feeds = make_feeds(w, n, split, feed_id0=f0)
        d = tmp_path / split
        d.mkdir(parents=True)
        h = n // 2
        save_part(d / "part-00000.npz", featurize_rows(feeds[:h], w["item_vec"]))
        save_part(d / "part-00001.npz", featurize_rows(feeds[h:], w["item_vec"]))
    return tmp_path


def test_cache_take_concat_folds(tmp_path):
    c = _cache(tmp_path)
    tr = load_split(c, "train")
    assert tr.n_feeds == 120 and tr.feed_off[-1] == tr.n_rows
    assert tr.has_target
    sub = tr.take([5, 2, 77])
    assert sub.n_feeds == 3 and sub.n_rows == int(tr.lens[[5, 2, 77]].sum())
    a, b = tr.feed_off[5], tr.feed_off[6]
    np.testing.assert_array_equal(sub.X[: b - a], tr.X[a:b])
    cc = SplitData.concat([tr.take([0, 1]), tr.take([2])])
    assert cc.n_feeds == 3 and cc.feed_off[-1] == cc.n_rows
    assert load_split(c, "train", max_feeds=50).n_feeds == 50
    # фолды: пользователь целиком в одном фолде, фолды не пересекаются
    val = load_split(c, "val")
    f = user_fold(val.user_id_u, 3)
    for u in np.unique(val.user_id_u):
        assert len(set(f[val.user_id_u == u])) == 1
    train, ev, test = make_sets(c, 0, 3, 5.0)
    assert ev.n_feeds + int((train.feed_w == 5.0).sum()) == val.n_feeds
    assert not set(ev.feed_id_u) & set(train.feed_id_u)
    assert test.n_feeds == 30 and not test.has_target


def test_signal_learnable(tmp_path):
    """Признаки несут сигнал: простая GBM на них бьёт случайный порядок и один f1."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    c = _cache(tmp_path, n_tr=500, n_val=200, n_te=10)
    tr, va = load_split(c, "train"), load_split(c, "val")
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0)
    clf.fit(tr.X, tr.y)
    s = clf.predict_proba(va.X)[:, 1]
    model = ndcg_per_feed(va.y, s, va.feed_off, 30).mean()
    rnd = ndcg_per_feed(va.y, np.random.default_rng(0).random(va.n_rows), va.feed_off, 30).mean()
    f1 = ndcg_per_feed(va.y, va.X[:, list(feature_names()).index("f1")], va.feed_off, 30).mean()
    assert model > rnd + 0.03 and model > f1 + 0.01, (model, rnd, f1)


def test_blend_and_submission(tmp_path):
    from avito_rank.blend import _align, _keys, blend, blend_test, group_zscore
    from avito_rank.scores_io import save_scores
    from avito_rank.submission import align_scores

    c = _cache(tmp_path / "c")
    va, te = load_split(c, "val"), load_split(c, "test")
    rng = np.random.default_rng(0)
    f1 = va.X[:, list(feature_names()).index("f1")]
    for name, noise in (("a", 0.2), ("b", 2.0)):
        for k in range(2):
            d = tmp_path / "runs" / name / f"fold{k}"
            d.mkdir(parents=True)
            idx = np.flatnonzero(user_fold(va.user_id_u, 2) == k)
            sd = va.take(idx)
            save_scores(d / "val_scores.npz", sd, sd.X[:, list(feature_names()).index("f1")] + rng.normal(0, noise, sd.n_rows))
            save_scores(d / "test_scores.npz", te, rng.normal(size=te.n_rows))
    runs = [str(tmp_path / "runs" / "a"), str(tmp_path / "runs" / "b")]
    w = blend(runs, tmp_path / "blend.json")
    assert abs(w.sum() - 1) < 1e-9 and w[0] > w[1]          # точная модель получает больший вес
    pred = blend_test(runs, w.tolist())
    # сэмпл в другом порядке строк
    perm = rng.permutation(te.n_rows)
    sf, si = te.row_feed_id[perm], te.item_id[perm]
    sc = align_scores(sf, si, pred)
    assert len(sc) == te.n_rows and np.isfinite(sc).all()
    with pytest.raises(ValueError):
        align_scores(sf, si, {k: v[:-1] for k, v in pred.items()})
    z = group_zscore(np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0]), np.array([1, 1, 1, 2, 2, 2]))
    assert abs(z[:3].mean()) < 1e-9 and np.allclose(z[3:], 0)

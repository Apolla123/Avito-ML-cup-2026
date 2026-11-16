"""CatBoostRanker (YetiRank) на инженерных признаках."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .log import logger

from .data import SplitData, make_sets
from .features import feature_names
from .metrics import ndcg_report
from .scores_io import save_scores


def to_pool(sd: SplitData, labels: bool, weights: bool):
    from catboost import Pool

    df = pd.DataFrame(sd.X, columns=list(feature_names()))
    df["category_id"] = sd.cat.astype(np.int32)
    kw = {"group_id": sd.row_feed_id}
    if labels:
        kw["label"] = sd.y.astype(np.float32)
    if weights and not np.allclose(sd.feed_w, 1.0):
        kw["group_weight"] = sd.row_weight
    return Pool(df, cat_features=["category_id"], **kw)


def run(cache, out, fold, nfolds, val_weight, max_train_feeds, params):
    from catboost import CatBoostRanker

    out = Path(out) / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    train, ev, test = make_sets(cache, fold, nfolds, val_weight, max_train_feeds)
    logger.info("train: {} лент / {} строк; eval: {} лент; test: {} лент",
                train.n_feeds, train.n_rows, ev.n_feeds, test.n_feeds)

    p = dict(
        loss_function=params["loss"], eval_metric="NDCG:top=30;type=Base",
        learning_rate=params["lr"], iterations=params["iterations"], depth=params["depth"],
        l2_leaf_reg=params["l2"], random_seed=params["seed"], early_stopping_rounds=params["patience"],
        allow_writing_files=False, thread_count=params["threads"], nan_mode="Min",
        boosting_type="Plain", max_ctr_complexity=1, verbose=100,
    )
    if params["device"] == "gpu":
        p.update(task_type="GPU", devices=params["gpu_devices"], gpu_ram_part=0.8)
    else:
        p.update(task_type="CPU")

    model = CatBoostRanker(**p)
    model.fit(to_pool(train, True, True), eval_set=to_pool(ev, True, False), use_best_model=True)
    model.save_model(str(out / "model.cbm"))

    ev_score = model.predict(to_pool(ev, False, False))
    metrics = ndcg_report(ev.y, ev_score, ev.feed_off)
    logger.info("fold {} eval: {}", fold, metrics)
    save_scores(out / "val_scores.npz", ev, ev_score)
    save_scores(out / "test_scores.npz", test, model.predict(to_pool(test, False, False)))

    try:
        imp = dict(zip([*feature_names(), "category_id"], model.get_feature_importance().tolist()))
        top = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:30])
    except Exception as e:  # важности — вспомогательный отчёт, не должен ронять обучение
        logger.warning("feature importance недоступна: {}", e)
        top = {}
    (out / "run.json").write_text(json.dumps(
        {"fold": fold, "nfolds": nfolds, "val_weight": val_weight, "params": p,
         "best_iteration": int(model.get_best_iteration()), "metrics": metrics, "top_features": top},
        indent=2, default=str))
    return metrics

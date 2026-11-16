"""Listwise Transformer: токен пользователя + токены кандидатов ленты.

Вход кандидата: стандартизованные инженерные признаки (NaN -> 0 + индикаторы пропусков),
эмбеддинг категории и L2-нормированный вектор товара. Loss: listwise softmax-CE (мульти-позитив)
+ небольшой BCE. Веса лент (val_weight) входят в loss.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from .log import logger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import SplitData, make_sets
from .metrics import ndcg_report
from .scores_io import save_scores


class Normalizer:
    def __init__(self, X: np.ndarray, max_rows: int = 2_000_000):
        rng = np.random.default_rng(0)
        S = X if len(X) <= max_rows else X[rng.choice(len(X), max_rows, replace=False)]
        self.nan_cols = np.flatnonzero(np.isnan(S).any(0))
        with np.errstate(all="ignore"):
            mu = np.nanmean(S, 0)
            sd = np.nanstd(S, 0)
        self.mu = np.nan_to_num(mu).astype(np.float32)
        self.sd = (np.nan_to_num(sd) + 1e-3).astype(np.float32)

    @property
    def n_out(self) -> int:
        return len(self.mu) + len(self.nan_cols)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        isn = np.isnan(X[:, self.nan_cols]).astype(np.float32)
        Z = np.clip((X - self.mu) / self.sd, -10, 10)
        Z = np.nan_to_num(Z, nan=0.0)
        return np.concatenate([Z, isn], 1).astype(np.float32)


class FeedDataset(Dataset):
    def __init__(self, sd: SplitData, norm: Normalizer):
        self.sd, self.norm = sd, norm

    def __len__(self):
        return self.sd.n_feeds

    def __getitem__(self, i):
        sd = self.sd
        a, b = sd.feed_off[i], sd.feed_off[i + 1]
        x = torch.from_numpy(self.norm(sd.X[a:b]))
        e = torch.from_numpy(sd.E[a:b].astype(np.float32))
        e = F.normalize(e, dim=1)
        cat = torch.from_numpy(sd.cat[a:b].astype(np.int64) + 128).clamp(0, 255)
        u = F.normalize(torch.from_numpy(sd.U[i].astype(np.float32)), dim=0)
        y = torch.from_numpy(np.maximum(sd.y[a:b], 0).astype(np.float32))
        return x, cat, e, u, y, torch.tensor(float(sd.feed_w[i]))


def collate(batch):
    xs, cats, es, us, ys, ws = zip(*batch)
    n = max(len(x) for x in xs)
    B = len(xs)
    X = torch.zeros(B, n, xs[0].shape[1])
    C = torch.zeros(B, n, dtype=torch.long)
    E = torch.zeros(B, n, es[0].shape[1])
    Y = torch.zeros(B, n)
    pad = torch.ones(B, n, dtype=torch.bool)
    for i in range(B):
        m = len(xs[i])
        X[i, :m], C[i, :m], E[i, :m], Y[i, :m] = xs[i], cats[i], es[i], ys[i]
        pad[i, :m] = False
    return X, C, E, torch.stack(us), Y, pad, torch.stack(ws)


class FeedRanker(nn.Module):
    def __init__(self, n_in: int, d: int = 128, layers: int = 3, heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.cat = nn.Embedding(256, 16)
        self.inp = nn.Sequential(nn.Linear(n_in + 16 + 128, d), nn.GELU(), nn.LayerNorm(d))
        self.user = nn.Sequential(nn.Linear(128, d), nn.GELU(), nn.LayerNorm(d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.skip = nn.Linear(n_in, 1)

    def forward(self, x, cat, item, user, pad):
        cand = self.inp(torch.cat([x, self.cat(cat), item], -1))
        tok = torch.cat([self.user(user)[:, None], cand], 1)
        mask = F.pad(pad, (1, 0), value=False)
        ctx = self.enc(tok, src_key_padding_mask=mask)[:, 1:]
        return self.out(ctx).squeeze(-1) + self.skip(x).squeeze(-1)


def loss_fn(logits, y, pad, w, bce_w=0.3):
    lg = logits.masked_fill(pad, -1e9)
    logp = F.log_softmax(lg, dim=1)
    valid = ~pad
    npos = (y * valid).sum(1).clamp(min=1.0)
    lst = -(logp * y * valid).sum(1) / npos
    wrow = w[:, None].expand_as(y)[valid]
    bce = F.binary_cross_entropy_with_logits(logits[valid], y[valid], reduction="none")
    return (lst * w).sum() / w.sum() + bce_w * (bce * wrow).sum() / wrow.sum()


@torch.no_grad()
def predict(model, sd: SplitData, norm: Normalizer, device, bs: int = 64) -> np.ndarray:
    model.eval()
    dl = DataLoader(FeedDataset(sd, norm), batch_size=bs, shuffle=False, collate_fn=collate)
    outs = []
    for X, C, E, U, _, pad, _ in dl:
        lg = model(X.to(device), C.to(device), E.to(device), U.to(device), pad.to(device)).cpu()
        outs.append(lg[~pad].numpy())
    return np.concatenate(outs)


def run(cache, out, fold, nfolds, val_weight, max_train_feeds, params):
    out = Path(out) / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    device = torch.device(params["device"])
    train, ev, test = make_sets(cache, fold, nfolds, val_weight, max_train_feeds)
    logger.info("train: {} лент / {} строк; eval: {} лент; test: {} лент",
                train.n_feeds, train.n_rows, ev.n_feeds, test.n_feeds)
    norm = Normalizer(train.X)
    model = FeedRanker(norm.n_out, params["width"], params["layers"], 4, params["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    dl = DataLoader(FeedDataset(train, norm), batch_size=params["batch_size"], shuffle=True,
                    collate_fn=collate, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=params["lr"], total_steps=params["epochs"] * len(dl),
                                                pct_start=0.15)
    best, best_state, history, bad = -1.0, None, [], 0
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        tot, nb = 0.0, 0
        for X, C, E, U, Y, pad, W in dl:
            X, C, E, U, Y, pad, W = (t.to(device) for t in (X, C, E, U, Y, pad, W))
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X, C, E, U, pad), Y, pad, W)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        m = ndcg_report(ev.y, predict(model, ev, norm, device), ev.feed_off)
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1), **m})
        logger.info(history[-1])
        if m["ndcg@30"] > best:
            best, best_state, bad = m["ndcg@30"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= params["patience"]:
                break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "params": params}, out / "model.pt")
    ev_score = predict(model, ev, norm, device)
    save_scores(out / "val_scores.npz", ev, ev_score)
    save_scores(out / "test_scores.npz", test, predict(model, test, norm, device))
    metrics = ndcg_report(ev.y, ev_score, ev.feed_off)
    (out / "run.json").write_text(json.dumps(
        {"fold": fold, "nfolds": nfolds, "val_weight": val_weight, "params": params,
         "history": history, "metrics": metrics}, indent=2, default=str))
    return metrics

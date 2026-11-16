"""Командная строка проекта: python -m avito_rank <команда> --help"""

import json
from collections import Counter
from itertools import islice
from pathlib import Path

import click
import numpy as np
from .log import logger


@click.group()
def main():
    """Ранжирование рекомендаций Авито."""


@main.command()
@click.option("--data", type=click.Path(path_type=Path), required=True, help="корень competition/")
@click.option("--feeds", default=3000, help="сколько лент train просмотреть")
def eda(data: Path, feeds: int):
    """Быстрая разведка: коды событий, CTR по позициям, пропуски эмбеддингов."""
    from .io_utils import list_parts, read_part_rows

    evt, src, pos_hit, pos_all = Counter(), Counter(), Counter(), Counter()
    sizes, hlen, npos, miss, tot = [], [], [], 0, 0
    parts = list_parts(data / "train")
    rows = islice((r for p in parts[:2] for r in read_part_rows(p)), feeds)
    for r in rows:
        evt.update(r["history__event_type"] or [])
        src.update(r["history__event_source"] or [])
        M = len(r["item_id"])
        sizes.append(M)
        hlen.append(len(r["history__item_id"] or []))
        npos.append(sum(r["target"]))
        idx = r["item_id__embedding_idx"]
        miss += sum(1 for i in idx if i < 0)
        tot += M
        for p, t in zip(r["pos"], r["target"]):
            if p is not None:
                b = min(int(p) // 10, 30)
                pos_all[b] += 1
                pos_hit[b] += t
    click.echo(f"кандидатов в ленте: mean={np.mean(sizes):.1f} min={min(sizes)} max={max(sizes)}")
    click.echo(f"длина истории: mean={np.mean(hlen):.1f}; позитивов на ленту: mean={np.mean(npos):.2f}")
    click.echo(f"доля кандидатов без эмбеддинга: {miss / max(tot, 1):.3f}")
    click.echo(f"event_type (код: число): {dict(sorted(evt.items()))}")
    click.echo(f"event_source (код: число): {dict(sorted(src.items()))}")
    click.echo("CTR train по бинам исходной позиции (по 10 позиций) — смещение позицией:")
    for b in sorted(pos_all):
        click.echo(f"  pos {b * 10:>3}+: {pos_hit[b] / pos_all[b]:.4f}  (n={pos_all[b]})")
    click.echo("Проверьте EVENT_TYPES / EVENT_SIM_TYPES в config.py под реальные коды событий.")


@main.command()
@click.option("--data", type=click.Path(path_type=Path), required=True)
@click.option("--cache", type=click.Path(path_type=Path), required=True)
@click.option("--splits", default="train,val,test")
@click.option("--workers", default=4)
@click.option("--limit-parts", default=0, help="только первые N parquet-файлов сплита (для отладки)")
def features(data, cache, splits, workers, limit_parts):
    """Считает признаки и кэширует их в .npz (возобновляемо)."""
    from .build_features import build_split

    for s in splits.split(","):
        build_split(data, cache, s, workers, limit_parts)


@main.command()
@click.argument("model", type=click.Choice(["catboost", "nn"]))
@click.option("--cache", type=click.Path(path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True, help="папка модели, напр. runs/cb")
@click.option("--fold", default=-1, help="-1: train->val; k>=0: фолд CV по val (см. README)")
@click.option("--nfolds", default=3)
@click.option("--val-weight", default=5.0, help="вес лент val при обучении (val ~ test по распределению)")
@click.option("--max-train-feeds", default=0)
@click.option("--seed", default=42)
@click.option("--device", default=None, help="catboost: gpu|cpu; nn: cuda|cpu")
@click.option("--threads", default=8)
# catboost
@click.option("--loss", default="YetiRank")
@click.option("--lr", default=None, type=float)
@click.option("--depth", default=8)
@click.option("--l2", default=5.0)
@click.option("--iterations", default=3000)
@click.option("--patience", default=None, type=int)
@click.option("--gpu-devices", default="0")
# nn
@click.option("--epochs", default=12)
@click.option("--batch-size", default=32)
@click.option("--width", default=128)
@click.option("--layers", default=3)
@click.option("--dropout", default=0.1)
@click.option("--wd", default=1e-2)
def train(model, cache, out, fold, nfolds, val_weight, max_train_feeds, seed, device, threads,
          loss, lr, depth, l2, iterations, patience, gpu_devices,
          epochs, batch_size, width, layers, dropout, wd):
    """Обучает модель; пишет out/fold{k}/{model, val_scores.npz, test_scores.npz, run.json}."""
    if model == "catboost":
        from .train_catboost import run

        params = dict(loss=loss, lr=lr or 0.08, iterations=iterations, depth=depth, l2=l2, seed=seed,
                      patience=patience or 100, threads=threads, device=device or "gpu",
                      gpu_devices=gpu_devices)
    else:
        from .train_nn import run

        params = dict(lr=lr or 1e-3, epochs=epochs, batch_size=batch_size, width=width, layers=layers,
                      dropout=dropout, wd=wd, seed=seed, patience=patience or 4, device=device or "cuda")
    m = run(cache, out, fold, nfolds, val_weight, max_train_feeds, params)
    click.echo(json.dumps(m))


@main.command()
@click.argument("runs", nargs=-1, required=True)
@click.option("--sample", type=click.Path(path_type=Path), required=True, help="sample_submission.parquet")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="куда писать submission.parquet")
@click.option("--weights-json", type=click.Path(path_type=Path), default=Path("blend.json"))
def blend(runs, sample, out, weights_json):
    """Подбирает веса ансамбля по OOF на val и пишет сабмит. RUNS — папки моделей (runs/cb runs/nn)."""
    from .blend import blend as fit_blend
    from .blend import blend_test
    from .submission import write_submission

    w = fit_blend(list(runs), weights_json)
    pred = blend_test(list(runs), w.tolist())
    write_submission(sample, pred, out)
    logger.info("Сабмит записан и проверен: {}", out)


@main.command()
@click.option("--sample", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--sub", type=click.Path(exists=True, path_type=Path), required=True)
def validate(sample, sub):
    """Проверяет формат сабмита относительно sample_submission."""
    from .submission import validate_submission

    validate_submission(sample, sub)
    click.echo("OK")


@main.command()
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--train", "n_train", default=600)
@click.option("--val", "n_val", default=200)
@click.option("--test", "n_test", default=200)
def synth(out, n_train, n_val, n_test):
    """Генерирует маленький синтетический датасет в формате соревнования (для проверки пайплайна)."""
    from .synth import write_competition

    write_competition(out, n_train, n_val, n_test)
    click.echo(f"готово: {out}")


if __name__ == "__main__":
    main()

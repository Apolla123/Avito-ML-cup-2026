"""Параллельная сборка кэша признаков: один parquet-файл -> один .npz. Возобновляемо."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .log import logger

from .data import save_part
from .features import featurize_rows
from .io_utils import list_parts, open_embeddings, read_part_rows


def _process(args):
    part, split_dir, out = args
    out = Path(out)
    if out.exists():
        return str(out), 0
    E = open_embeddings(split_dir)
    arrays = featurize_rows(read_part_rows(part), E)
    save_part(out, arrays)
    return str(out), len(arrays["feed_id_u"])


def build_split(data_dir, cache_dir, split: str, workers: int = 4, limit_parts: int = 0) -> None:
    # потоки BLAS внутри процессов отключаем, параллелим по файлам
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    split_dir = Path(data_dir) / split
    parts = list_parts(split_dir)
    if limit_parts:
        parts = parts[:limit_parts]
    out_dir = Path(cache_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(split_dir), str(out_dir / f"{p.stem}.npz")) for p in parts]
    logger.info("{}: {} parquet-файлов, воркеров: {}", split, len(jobs), workers)
    if workers <= 1:
        results = map(_process, jobs)
        for name, n in results:
            logger.info("{} feeds={}", name, n)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, n in ex.map(_process, jobs):
            logger.info("{} feeds={}", name, n)

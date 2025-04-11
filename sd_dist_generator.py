import numpy as np
import os
from joblib import Parallel, delayed

UPLOAD_DIR = "data/sd_dist/"


def distribution_worker(max_target: int, ranks: np.array) -> np.ndarray:
    """Generate distribution for standard deviation calculation."""
    arr = np.zeros(max_target)
    cs = np.random.choice(ranks, max_target, replace=False).cumsum()
    for i in range(0, max_target):
        amr = cs[i] / (i + 1)
        # arr[i] = np.min([amr, 1 - amr])  # One-sided test
        arr[i] = amr  # Two-sided test
    return arr


def get_sd(k: int, n: int, iters: int) -> np.ndarray:
    """Load or generate standard deviation distribution."""
    ranks = np.linspace(start=1, stop=n, num=n)
    ranks = (ranks - 0.5) / n

    dist = Parallel(n_jobs=-1, verbose=3, backend="multiprocessing")(
        delayed(distribution_worker)(k, ranks) for _ in range(iters)
    )
    dist = np.std(np.array(dist).T, axis=1)
    return dist


if __name__ == "__main__":
    k = [100, 150, 200, 300]
    n = [1000, 4000, 7000, 9000]
    iters = 10000
    for i in range(4):
        print(f"Generating distribution for k={k[i]}, n={n[i]}")
        get_sd(k[i], n[i], iters)

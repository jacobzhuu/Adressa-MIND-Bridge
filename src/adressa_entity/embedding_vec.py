from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def load_vec_subset(path: str | Path, keep_keys: set[str]) -> dict[str, np.ndarray]:
    path = Path(path)
    vectors: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if not parts:
                continue
            key = parts[0]
            if key not in keep_keys:
                continue
            vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            vectors[key] = vec
    return vectors


def write_vec(path: str | Path, keys: Iterable[str], matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")

    with path.open("w", encoding="utf-8") as f:
        for i, key in enumerate(keys):
            row = matrix[i]
            f.write(key)
            f.write("\t")
            f.write("\t".join(f"{float(x):.6f}" for x in row))
            f.write("\n")


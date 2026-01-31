#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur.parent] + list(cur.parents):
        if (parent / "src").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"Could not find repo root from {cur}")


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.embedding_vec import write_vec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export entity_embedding.vec in MIND format (QID + 100 dims).")
    p.add_argument("--entity_vocab", type=Path, required=True, help="entity_vocab.txt (one QID per line).")
    p.add_argument("--entity_matrix", type=Path, required=True, help="entity_trained.npy or entity_init.npy")
    p.add_argument("--output_vec", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qids = [line.strip() for line in args.entity_vocab.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = np.load(args.entity_matrix).astype(np.float32)
    if matrix.shape[0] != len(qids):
        raise ValueError(f"Row mismatch: vocab has {len(qids)} QIDs, matrix has {matrix.shape[0]} rows")
    write_vec(args.output_vec, qids, matrix)


if __name__ == "__main__":
    main()

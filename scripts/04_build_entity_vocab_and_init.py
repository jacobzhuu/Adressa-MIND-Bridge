#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.embedding_vec import load_vec_subset, write_vec
from adressa_entity.news_tsv import iter_news_tsv


DEFAULT_MIND_VEC = Path("data/mind/MINDsmall/train/entity_embedding.vec")
LEGACY_MIND_VEC = Path("MINDsmall/train/entity_embedding.vec")
if LEGACY_MIND_VEC.exists() and not DEFAULT_MIND_VEC.exists():
    DEFAULT_MIND_VEC = LEGACY_MIND_VEC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Adressa entity vocab and initialize vectors from MIND embeddings.")
    p.add_argument(
        "--news_tsv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Adressa news.tsv files with populated title_entities.",
    )
    p.add_argument(
        "--mind_entity_vec",
        type=Path,
        nargs="+",
        default=[DEFAULT_MIND_VEC],
        help="One or more MIND entity_embedding.vec paths (used for initialization; later files fill missing QIDs).",
    )
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_qids(news_paths: list[Path]) -> list[str]:
    qids: set[str] = set()
    for path in news_paths:
        for row in iter_news_tsv(path):
            te = row.title_entities.strip()
            if not te or te == "[]":
                continue
            try:
                ents = json.loads(te)
            except json.JSONDecodeError:
                continue
            for e in ents:
                qid = str(e.get("WikidataId") or "")
                if qid.startswith("Q"):
                    qids.add(qid)
    return sorted(qids)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qids = collect_qids(args.news_tsv)
    qid_set = set(qids)

    mind_vecs: dict[str, np.ndarray] = {}
    remaining = set(qid_set)
    for vec_path in args.mind_entity_vec:
        if not remaining:
            break
        loaded = load_vec_subset(vec_path, remaining)
        mind_vecs.update(loaded)
        remaining -= set(loaded.keys())
    if mind_vecs:
        dim = next(iter(mind_vecs.values())).shape[0]
    else:
        dim = 100

    rng = np.random.default_rng(args.seed)
    matrix = rng.normal(loc=0.0, scale=0.01, size=(len(qids), dim)).astype(np.float32)
    is_pretrained = np.zeros((len(qids),), dtype=np.int8)

    hit = 0
    for i, qid in enumerate(tqdm(qids, desc="init", unit="ent")):
        vec = mind_vecs.get(qid)
        if vec is None:
            continue
        if vec.shape[0] != dim:
            raise ValueError(f"Dimension mismatch for {qid}: got {vec.shape[0]}, expected {dim}")
        matrix[i] = vec
        is_pretrained[i] = 1
        hit += 1

    (args.output_dir / "entity_vocab.txt").write_text("\n".join(qids) + "\n", encoding="utf-8")
    np.save(args.output_dir / "entity_init.npy", matrix)
    np.save(args.output_dir / "entity_init_mask.npy", is_pretrained)

    stats = {
        "num_entities": len(qids),
        "mind_hits": hit,
        "mind_misses": len(qids) - hit,
        "coverage": (hit / len(qids)) if qids else 0.0,
        "dim": dim,
    }
    (args.output_dir / "entity_init_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    write_vec(args.output_dir / "entity_init.vec", qids, matrix)


if __name__ == "__main__":
    main()

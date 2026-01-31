#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _is_qid(x: str) -> bool:
    return x.startswith("Q") and x[1:].isdigit()


def _base_pid(rel: str) -> str:
    return rel[:-4] if rel.endswith("_inv") else rel


def _load_seed_vocab(path: Path) -> list[str]:
    qids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [q for q in qids if _is_qid(q)]


def _load_relation_vec(path: Path) -> dict[str, np.ndarray]:
    rels: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            pid = parts[0]
            if not (pid.startswith("P") and pid[1:].isdigit()):
                continue
            vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            rels[pid] = vec
    return rels


def _iter_triples(path: Path) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid triple at {path}:{ln}: {line!r}")
        out.append((parts[0], parts[1], parts[2]))
    return out


@dataclass(frozen=True)
class SplitStats:
    num_forward_triples: int
    num_lines_with_inv: int
    relation_counts: dict[str, int]


def _write_split_file(path: Path, forward: list[tuple[str, str, str]]) -> SplitStats:
    rel_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in forward:
            f.write(f"{h}\t{r}\t{t}\n")
            f.write(f"{t}\t{r}_inv\t{h}\n")
            rel_counts[r] += 2
    return SplitStats(
        num_forward_triples=int(len(forward)),
        num_lines_with_inv=int(len(forward) * 2),
        relation_counts=dict(rel_counts),
    )


def _make_splits(
    *,
    kg_triples: Path,
    seed_entity_vocab: Path,
    mind_relation_vec: Path,
    output_dir: Path,
    split: tuple[float, float, float],
    seed: int,
    strip_inv: bool,
    limit_forward: int | None,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "split_meta.json"
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    test_path = output_dir / "test.txt"

    if meta_path.exists() and not overwrite:
        raise FileExistsError(f"{meta_path} already exists. Use --overwrite to regenerate splits.")
    if overwrite:
        for p in (meta_path, train_path, val_path, test_path):
            if p.exists():
                p.unlink()

    seed_qids = _load_seed_vocab(seed_entity_vocab)
    seed_set = set(seed_qids)
    rel_vec = _load_relation_vec(mind_relation_vec)

    forward_set: set[tuple[str, str, str]] = set()
    skipped_unknown_rel = 0
    skipped_oov_ent = 0
    skipped_non_item = 0
    raw = 0

    for h, r, t in _iter_triples(kg_triples):
        raw += 1
        if strip_inv and r.endswith("_inv"):
            continue
        if not (_is_qid(h) and _is_qid(t)):
            skipped_non_item += 1
            continue
        r0 = _base_pid(r)
        if r0 not in rel_vec:
            skipped_unknown_rel += 1
            continue
        if h not in seed_set or t not in seed_set:
            skipped_oov_ent += 1
            continue
        forward_set.add((h, r0, t))

    forward = sorted(forward_set)
    rng = random.Random(int(seed))
    rng.shuffle(forward)
    if limit_forward is not None:
        forward = forward[: int(limit_forward)]

    s_train, s_val, s_test = split
    if not np.isclose(s_train + s_val + s_test, 1.0):
        raise ValueError("--split must sum to 1.0")
    n = len(forward)
    n_train = int(n * s_train)
    n_val = int(n * s_val)
    train = forward[:n_train]
    val = forward[n_train : n_train + n_val]
    test = forward[n_train + n_val :]

    train_stats = _write_split_file(train_path, train)
    val_stats = _write_split_file(val_path, val)
    test_stats = _write_split_file(test_path, test)

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "kg_triples_path": str(kg_triples),
            "kg_triples_sha1": _sha1_file(kg_triples),
            "seed_entity_vocab_path": str(seed_entity_vocab),
            "seed_entity_vocab_sha1": _sha1_file(seed_entity_vocab),
            "mind_relation_vec_path": str(mind_relation_vec),
            "mind_relation_vec_sha1": _sha1_file(mind_relation_vec),
        },
        "config": {
            "split": [float(s_train), float(s_val), float(s_test)],
            "seed": int(seed),
            "strip_inv": bool(strip_inv),
            "limit_forward": int(limit_forward) if limit_forward is not None else None,
        },
        "stats": {
            "num_seed_entities": int(len(seed_qids)),
            "num_relations_allowed": int(len(rel_vec)),
            "raw_lines": int(raw),
            "num_forward_triples": int(n),
            "skipped_non_item": int(skipped_non_item),
            "skipped_unknown_rel": int(skipped_unknown_rel),
            "skipped_oov_ent": int(skipped_oov_ent),
            "train": train_stats.__dict__,
            "val": val_stats.__dict__,
            "test": test_stats.__dict__,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_forward_index_triples(
    path: Path, *, ent2idx: dict[str, int], rel_vec: dict[str, np.ndarray]
) -> list[tuple[int, str, int]]:
    out: list[tuple[int, str, int]] = []
    for h, r, t in _iter_triples(path):
        if r.endswith("_inv"):
            continue
        if not (_is_qid(h) and _is_qid(t)):
            continue
        if h not in ent2idx or t not in ent2idx:
            continue
        if r not in rel_vec:
            continue
        out.append((ent2idx[h], r, ent2idx[t]))
    return out


def _rank(dist: np.ndarray, true_idx: int, *, tie_mode: str) -> int:
    true_score = float(dist[true_idx])
    if tie_mode == "optimistic":
        return 1 + int(np.sum(dist < true_score))
    if tie_mode == "pessimistic":
        # Count strictly better + ties excluding the true target itself.
        ties = int(np.sum(dist == true_score)) - 1
        better = int(np.sum(dist < true_score))
        return 1 + better + max(0, ties)
    raise ValueError(f"Unknown tie_mode: {tie_mode}")


def _eval_link_prediction(
    *,
    triples: list[tuple[int, str, int]],
    ent: np.ndarray,
    rel_vec: dict[str, np.ndarray],
    all_true_tails: dict[tuple[int, str], set[int]],
    all_true_heads: dict[tuple[str, int], set[int]],
    ks: list[int],
    mode: str,
    tie_mode: str,
    max_eval: int | None,
) -> dict[str, float]:
    if max_eval is not None:
        triples = triples[: int(max_eval)]

    mrr = 0.0
    mean_rank = 0.0
    hits = {k: 0.0 for k in ks}
    n = 0

    def dist_tail(h: int, r: str) -> np.ndarray:
        q = ent[h] + rel_vec[r]
        return np.abs(ent - q).sum(axis=1)

    def dist_head(t: int, r: str) -> np.ndarray:
        q = ent[t] - rel_vec[r]
        return np.abs(ent - q).sum(axis=1)

    for h, r, t in tqdm(triples, desc=f"eval[{mode}]", unit="tri"):
        if mode in {"tail", "both"}:
            d = dist_tail(h, r)
            for tt in all_true_tails.get((h, r), set()):
                if tt != t:
                    d[tt] = np.inf
            rnk = _rank(d, t, tie_mode=tie_mode)
            mrr += 1.0 / float(rnk)
            mean_rank += float(rnk)
            for k in ks:
                hits[k] += float(rnk <= k)
            n += 1

        if mode in {"head", "both"}:
            d = dist_head(t, r)
            for hh in all_true_heads.get((r, t), set()):
                if hh != h:
                    d[hh] = np.inf
            rnk = _rank(d, h, tie_mode=tie_mode)
            mrr += 1.0 / float(rnk)
            mean_rank += float(rnk)
            for k in ks:
                hits[k] += float(rnk <= k)
            n += 1

    denom = float(max(1, n))
    out: dict[str, float] = {
        "n": float(n),
        "MRR": float(mrr / denom),
        "mean_rank": float(mean_rank / denom),
    }
    for k in ks:
        out[f"Hits@{k}"] = float(hits[k] / denom)
    return out


def _cmd_split(args: argparse.Namespace) -> None:
    _make_splits(
        kg_triples=args.kg_triples,
        seed_entity_vocab=args.seed_entity_vocab,
        mind_relation_vec=args.mind_relation_vec,
        output_dir=args.output_dir,
        split=(float(args.split[0]), float(args.split[1]), float(args.split[2])),
        seed=int(args.seed),
        strip_inv=bool(args.strip_inv),
        limit_forward=args.limit_forward,
        overwrite=bool(args.overwrite),
    )
    print(f"OK: wrote splits to {args.output_dir}")


def _cmd_eval(args: argparse.Namespace) -> None:
    split_dir = args.split_dir
    meta_path = split_dir / "split_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}

    seed_qids = _load_seed_vocab(args.seed_entity_vocab)
    ent2idx = {q: i for i, q in enumerate(seed_qids)}
    rel_vec = _load_relation_vec(args.mind_relation_vec)
    if not seed_qids:
        raise ValueError("seed_entity_vocab is empty.")
    dim = int(next(iter(rel_vec.values())).shape[0]) if rel_vec else 100

    train = _load_forward_index_triples(split_dir / "train.txt", ent2idx=ent2idx, rel_vec=rel_vec)
    val = _load_forward_index_triples(split_dir / "val.txt", ent2idx=ent2idx, rel_vec=rel_vec)
    test = _load_forward_index_triples(split_dir / "test.txt", ent2idx=ent2idx, rel_vec=rel_vec)

    all_true_tails: dict[tuple[int, str], set[int]] = {}
    all_true_heads: dict[tuple[str, int], set[int]] = {}
    for h, r, t in train + val + test:
        all_true_tails.setdefault((h, r), set()).add(t)
        all_true_heads.setdefault((r, t), set()).add(h)

    ks = [int(k) for k in args.ks]
    mode = str(args.mode)
    tie_mode = str(args.tie_mode)

    results: dict[str, object] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_dir": str(split_dir),
        "meta": meta,
        "eval_config": {
            "mode": mode,
            "tie_mode": tie_mode,
            "ks": ks,
            "max_eval_triples": int(args.max_eval_triples) if args.max_eval_triples is not None else None,
        },
        "stats": {
            "num_entities": int(len(seed_qids)),
            "dim": int(dim),
            "num_train_forward": int(len(train)),
            "num_val_forward": int(len(val)),
            "num_test_forward": int(len(test)),
        },
        "metrics": {},
    }

    for name, path in args.entity_matrix:
        mat = np.load(Path(path)).astype(np.float32)
        if mat.shape != (len(seed_qids), dim):
            raise ValueError(f"{name}: expected shape {(len(seed_qids), dim)}, got {mat.shape}")
        metrics: dict[str, object] = {}
        if args.eval_split in {"val", "both"}:
            metrics["val"] = _eval_link_prediction(
                triples=val,
                ent=mat,
                rel_vec=rel_vec,
                all_true_tails=all_true_tails,
                all_true_heads=all_true_heads,
                ks=ks,
                mode=mode,
                tie_mode=tie_mode,
                max_eval=args.max_eval_triples,
            )
        if args.eval_split in {"test", "both"}:
            metrics["test"] = _eval_link_prediction(
                triples=test,
                ent=mat,
                rel_vec=rel_vec,
                all_true_tails=all_true_tails,
                all_true_heads=all_true_heads,
                ks=ks,
                mode=mode,
                tie_mode=tie_mode,
                max_eval=args.max_eval_triples,
            )
        results["metrics"][name] = metrics

    out_path = split_dir / "metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["metrics"], ensure_ascii=False, indent=2))
    print(f"OK: wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KG holdout split + filtered link prediction eval (MRR/Hits@K) for TransE.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="Create holdout splits (train/val/test) without inverse leakage.")
    sp.add_argument("--kg_triples", type=Path, required=True)
    sp.add_argument("--seed_entity_vocab", type=Path, required=True)
    sp.add_argument("--mind_relation_vec", type=Path, required=True)
    sp.add_argument("--output_dir", type=Path, required=True)
    sp.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--strip_inv", action=argparse.BooleanOptionalAction, default=True, help="Ignore *_inv lines in input kg_triples.")
    sp.add_argument("--limit_forward", type=int, default=None, help="Limit number of forward triples (debug).")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=_cmd_split)

    ev = sub.add_parser("eval", help="Evaluate filtered link prediction on val/test splits.")
    ev.add_argument("--split_dir", type=Path, required=True, help="Directory containing train.txt/val.txt/test.txt.")
    ev.add_argument("--seed_entity_vocab", type=Path, required=True)
    ev.add_argument("--mind_relation_vec", type=Path, required=True)
    ev.add_argument(
        "--entity_matrix",
        nargs=2,
        action="append",
        metavar=("NAME", "NPY_PATH"),
        required=True,
        help="Repeatable: NAME entity_matrix.npy. Compare multiple matrices (e.g. init vs trained).",
    )
    ev.add_argument("--eval_split", type=str, choices=["val", "test", "both"], default="both")
    ev.add_argument("--mode", type=str, choices=["tail", "head", "both"], default="both")
    ev.add_argument("--ks", type=int, nargs="+", default=[1, 3, 10])
    ev.add_argument("--tie_mode", type=str, choices=["optimistic", "pessimistic"], default="optimistic")
    ev.add_argument("--max_eval_triples", type=int, default=None, help="Limit number of eval triples (debug).")
    ev.set_defaults(func=_cmd_eval)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

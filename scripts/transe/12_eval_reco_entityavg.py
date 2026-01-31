#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys

def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur.parent] + list(cur.parents):
        if (parent / "src").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"Could not find repo root from {cur}")


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.news_tsv import iter_news_tsv


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _is_qid(x: str) -> bool:
    return x.startswith("Q") and x[1:].isdigit()


def _load_vocab(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and _is_qid(ln.strip())]


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, ord=2, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0)
    return (mat / norms).astype(np.float32)


def _l2_norm(vec: np.ndarray) -> float:
    return float(np.sqrt(np.dot(vec, vec)))


def _harmonic_prefix(n: int) -> list[float]:
    h = [0.0] * (n + 1)
    for i in range(1, n + 1):
        h[i] = h[i - 1] + 1.0 / float(i)
    return h


def _expected_rr(greater: int, tie_count: int, *, h_prefix: list[float]) -> float:
    # Positions are 1-indexed.
    start = int(greater) + 1
    end = start + int(tie_count) - 1
    if tie_count <= 0:
        return 0.0
    return (h_prefix[end] - h_prefix[start - 1]) / float(tie_count)


def _auc_one_positive(pos_score: float, neg_scores: list[float]) -> float:
    if not neg_scores:
        return float("nan")
    better = 0.0
    for ns in neg_scores:
        if pos_score > ns:
            better += 1.0
        elif pos_score == ns:
            better += 0.5
    return better / float(len(neg_scores))


def _mrr_one_positive(pos_score: float, other_scores: list[float], *, tie_mode: str, h_prefix: list[float]) -> float:
    greater = sum(1 for s in other_scores if s > pos_score)
    ties = sum(1 for s in other_scores if s == pos_score)
    tie_count = 1 + ties  # include the positive itself
    if tie_mode == "optimistic":
        return 1.0 / float(greater + 1)
    if tie_mode == "pessimistic":
        return 1.0 / float(greater + tie_count)
    if tie_mode == "expected":
        return _expected_rr(greater, tie_count, h_prefix=h_prefix)
    raise ValueError(f"Unknown tie_mode: {tie_mode}")


@dataclass(frozen=True)
class _NewsEntry:
    news_id: str
    entity_idx: np.ndarray  # int64
    entity_w: np.ndarray  # float32


def _load_news_entities(
    news_tsv_paths: list[Path], *, qid2idx: dict[str, int], entity_weight: str
) -> tuple[list[str], dict[str, int], list[_NewsEntry], dict[str, float]]:
    # Union all news by id (first occurrence wins).
    seen: set[str] = set()
    entries: list[_NewsEntry] = []
    news_ids: list[str] = []
    stats: dict[str, float] = defaultdict(float)

    for path in news_tsv_paths:
        for row in iter_news_tsv(path):
            nid = row.news_id
            if nid in seen:
                continue
            seen.add(nid)

            idxs: list[int] = []
            ws: list[float] = []
            te = row.title_entities.strip()
            if te and te != "[]":
                try:
                    ents = json.loads(te)
                except Exception:
                    ents = []
                if isinstance(ents, list):
                    for e in ents:
                        if not isinstance(e, dict):
                            continue
                        qid = str(e.get("WikidataId") or "")
                        if qid not in qid2idx:
                            continue
                        idxs.append(int(qid2idx[qid]))
                        if entity_weight == "confidence":
                            try:
                                ws.append(float(e.get("Confidence") or 0.0))
                            except Exception:
                                ws.append(0.0)
                        else:
                            ws.append(1.0)

            if idxs:
                stats["news_with_entities"] += 1.0
                stats["sum_entities_per_news"] += float(len(idxs))
            stats["num_news"] += 1.0

            ent_idx = np.asarray(idxs, dtype=np.int64)
            ent_w = np.asarray(ws, dtype=np.float32)
            entries.append(_NewsEntry(news_id=nid, entity_idx=ent_idx, entity_w=ent_w))
            news_ids.append(nid)

    news2i = {nid: i for i, nid in enumerate(news_ids)}
    return news_ids, news2i, entries, dict(stats)


def _build_news_matrix(
    *,
    entries: list[_NewsEntry],
    entity_matrix: np.ndarray,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dim = int(entity_matrix.shape[1])
    news_mat = np.zeros((len(entries), dim), dtype=np.float32)
    has_vec = np.zeros((len(entries),), dtype=np.int8)
    for i, e in enumerate(entries):
        if e.entity_idx.size == 0:
            continue
        vecs = entity_matrix[e.entity_idx]
        if vecs.size == 0:
            continue
        w = e.entity_w
        if w.size != e.entity_idx.size:
            w = np.ones((e.entity_idx.size,), dtype=np.float32)
        w_sum = float(w.sum())
        if w_sum <= 0:
            v = vecs.mean(axis=0)
        else:
            v = (vecs * w[:, None]).sum(axis=0) / w_sum
        news_mat[i] = v.astype(np.float32)
        has_vec[i] = 1
    if normalize:
        news_mat = _l2_normalize_rows(news_mat)
    return news_mat, has_vec.astype(bool)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Downstream recommendation baseline: entity-avg news vectors + user history sum; report AUC/MRR."
    )
    p.add_argument("--news_tsv", type=Path, nargs="+", required=True, help="One or more news.tsv (union for coverage).")
    p.add_argument("--behaviors_tsv", type=Path, required=True)
    p.add_argument("--entity_vocab", type=Path, required=True)
    p.add_argument(
        "--entity_matrix",
        nargs=2,
        action="append",
        metavar=("NAME", "NPY_PATH"),
        required=True,
        help="Repeatable: NAME entity_matrix.npy (aligned with entity_vocab).",
    )
    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument("--entity_weight", choices=["uniform", "confidence"], default="confidence")
    p.add_argument("--score", choices=["cosine", "dot"], default="cosine")
    p.add_argument("--tie_mode", choices=["expected", "optimistic", "pessimistic"], default="expected")
    p.add_argument("--max_history", type=int, default=0, help="Window size (0 means unlimited).")
    p.add_argument("--max_impressions", type=int, default=None, help="Limit number of behaviors rows (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    vocab = _load_vocab(args.entity_vocab)
    if not vocab:
        raise ValueError("entity_vocab is empty.")
    qid2idx = {q: i for i, q in enumerate(vocab)}

    news_ids, news2i, entries, news_stats = _load_news_entities(
        args.news_tsv, qid2idx=qid2idx, entity_weight=str(args.entity_weight)
    )
    if not news_ids:
        raise ValueError("No news loaded from news_tsv.")

    models: list[tuple[str, Path]] = [(str(n), Path(p)) for (n, p) in args.entity_matrix]
    mats: dict[str, np.ndarray] = {}
    for name, path in models:
        m = np.load(path).astype(np.float32)
        if m.shape[0] != len(vocab):
            raise ValueError(f"{name}: row mismatch: vocab={len(vocab)}, matrix={m.shape[0]}")
        if m.shape[1] != 100:
            raise ValueError(f"{name}: dim mismatch: expected 100, got {m.shape[1]}")
        mats[name] = m

    normalize_news = str(args.score) == "cosine"
    news_vecs: dict[str, np.ndarray] = {}
    news_has_vec: dict[str, np.ndarray] = {}
    for name in mats:
        nm, hv = _build_news_matrix(entries=entries, entity_matrix=mats[name], normalize=normalize_news)
        news_vecs[name] = nm
        news_has_vec[name] = hv

    # Harmonic prefix for expected MRR under ties. Candidate count is typically small; 1024 is plenty.
    h_prefix = _harmonic_prefix(1024)

    # Per-model accumulators.
    metrics: dict[str, dict[str, float]] = {name: defaultdict(float) for name in mats}
    # Per-model user state: sum of clicked news vectors (+ optional queue for windowed history).
    user_sum: dict[str, dict[str, np.ndarray]] = {name: {} for name in mats}
    user_queue: dict[str, dict[str, list[int]]] = {name: {} for name in mats} if int(args.max_history) > 0 else {}

    # Global stats.
    g: dict[str, float] = defaultdict(float)

    def ensure_user(uid: str, history_ids: list[str]) -> None:
        if uid in user_sum[next(iter(mats))]:
            return

        # Apply max_history window on provided history.
        if int(args.max_history) > 0 and len(history_ids) > int(args.max_history):
            history_ids = history_ids[-int(args.max_history) :]

        idxs: list[int] = []
        missing = 0
        for nid in history_ids:
            ni = news2i.get(nid)
            if ni is None:
                missing += 1
                continue
            idxs.append(int(ni))

        g["init_history_items_total"] += float(len(history_ids))
        g["init_history_items_missing_news"] += float(missing)

        # has_vec is model-dependent but in practice should be identical across models here.
        any_name = next(iter(mats))
        usable = [i for i in idxs if news_has_vec[any_name][i]]
        g["init_history_items_used"] += float(len(usable))

        for name in mats:
            if usable:
                s = news_vecs[name][usable].sum(axis=0).astype(np.float32)
            else:
                s = np.zeros((100,), dtype=np.float32)
            user_sum[name][uid] = s
            if int(args.max_history) > 0:
                user_queue[name][uid] = list(usable)

        g["users_seen"] += 1.0

    def score_candidates(name: str, uid: str, cand_idx: list[int]) -> tuple[list[float], float]:
        u = user_sum[name].get(uid)
        if u is None:
            u = np.zeros((100,), dtype=np.float32)
        if str(args.score) == "cosine":
            u_norm = _l2_norm(u)
            if u_norm <= 0:
                return [0.0 for _ in cand_idx], 0.0
            out: list[float] = []
            for i in cand_idx:
                if i < 0:
                    out.append(0.0)
                else:
                    out.append(float(np.dot(news_vecs[name][i], u) / u_norm))
            return out, u_norm

        # dot
        out = []
        for i in cand_idx:
            if i < 0:
                out.append(0.0)
            else:
                out.append(float(np.dot(news_vecs[name][i], u)))
        return out, _l2_norm(u)

    def update_user(uid: str, clicked_news_idx: int) -> None:
        if clicked_news_idx < 0:
            return
        any_name = next(iter(mats))
        if not news_has_vec[any_name][clicked_news_idx]:
            return
        for name in mats:
            user_sum[name][uid] += news_vecs[name][clicked_news_idx]
            if int(args.max_history) > 0:
                q = user_queue[name][uid]
                q.append(clicked_news_idx)
                if len(q) > int(args.max_history):
                    old = q.pop(0)
                    user_sum[name][uid] -= news_vecs[name][old]

    # Pre-count lines for tqdm if feasible (fast enough for ~300k).
    try:
        total_lines = sum(1 for _ in args.behaviors_tsv.open("rb"))
    except Exception:
        total_lines = None

    with args.behaviors_tsv.open("r", encoding="utf-8") as f:
        pbar = tqdm(f, desc="eval", unit="imp", total=total_lines)
        for line_idx, line in enumerate(pbar):
            if args.max_impressions is not None and line_idx >= int(args.max_impressions):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            uid = parts[1]
            hist = parts[3].strip()
            imps = parts[4].strip()
            g["impressions_total"] += 1.0

            history_ids = hist.split() if hist else []
            ensure_user(uid, history_ids)

            cand_ids: list[str] = []
            labels: list[int] = []
            pos_nid: str | None = None
            for tok in imps.split():
                try:
                    nid, lab = tok.rsplit("-", 1)
                except ValueError:
                    continue
                cand_ids.append(nid)
                y = 1 if lab == "1" else 0
                labels.append(y)
                if y == 1:
                    pos_nid = nid

            if pos_nid is None or not cand_ids:
                continue

            cand_idx: list[int] = [int(news2i.get(nid, -1)) for nid in cand_ids]
            pos_idx = int(news2i.get(pos_nid, -1))
            pos_has_vec = False
            if pos_idx >= 0:
                # has_vec is model-dependent (but should be identical here); take first model.
                any_name = next(iter(mats))
                pos_has_vec = bool(news_has_vec[any_name][pos_idx])

            # Compute metrics for each model.
            for name in mats:
                scores, u_norm = score_candidates(name, uid, cand_idx)
                # Find the (single) positive score.
                try:
                    pos_pos = labels.index(1)
                except ValueError:
                    continue
                pos_score = scores[pos_pos]
                neg_scores = [s for s, y in zip(scores, labels) if y == 0]

                auc = _auc_one_positive(pos_score, neg_scores)
                if not math.isnan(auc):
                    metrics[name]["auc_sum"] += float(auc)
                    metrics[name]["auc_count"] += 1.0

                rr = _mrr_one_positive(
                    pos_score,
                    [s for i, s in enumerate(scores) if i != pos_pos],
                    tie_mode=str(args.tie_mode),
                    h_prefix=h_prefix,
                )
                metrics[name]["mrr_sum"] += float(rr)
                metrics[name]["mrr_count"] += 1.0

                # Filtered subsets.
                if u_norm > 0:
                    metrics[name]["auc_sum_user_nonzero"] += 0.0 if math.isnan(auc) else float(auc)
                    metrics[name]["auc_count_user_nonzero"] += 0.0 if math.isnan(auc) else 1.0
                    metrics[name]["mrr_sum_user_nonzero"] += float(rr)
                    metrics[name]["mrr_count_user_nonzero"] += 1.0
                if pos_has_vec:
                    metrics[name]["auc_sum_pos_has_vec"] += 0.0 if math.isnan(auc) else float(auc)
                    metrics[name]["auc_count_pos_has_vec"] += 0.0 if math.isnan(auc) else 1.0
                    metrics[name]["mrr_sum_pos_has_vec"] += float(rr)
                    metrics[name]["mrr_count_pos_has_vec"] += 1.0
                if u_norm > 0 and pos_has_vec:
                    metrics[name]["auc_sum_user_nonzero_pos_has_vec"] += 0.0 if math.isnan(auc) else float(auc)
                    metrics[name]["auc_count_user_nonzero_pos_has_vec"] += 0.0 if math.isnan(auc) else 1.0
                    metrics[name]["mrr_sum_user_nonzero_pos_has_vec"] += float(rr)
                    metrics[name]["mrr_count_user_nonzero_pos_has_vec"] += 1.0

            # Update user state with clicked positive.
            update_user(uid, pos_idx)

    # Summarize.
    results: dict[str, object] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": float(time.time() - t0),
        "inputs": {
            "behaviors_tsv": {"path": str(args.behaviors_tsv), "sha1": _sha1_file(args.behaviors_tsv)},
            "news_tsv": [{"path": str(p), "sha1": _sha1_file(p)} for p in args.news_tsv],
            "entity_vocab": {"path": str(args.entity_vocab), "sha1": _sha1_file(args.entity_vocab)},
            "entity_matrices": [{"name": n, "path": str(p), "sha1": _sha1_file(Path(p))} for n, p in models],
        },
        "config": {
            "entity_weight": str(args.entity_weight),
            "score": str(args.score),
            "tie_mode": str(args.tie_mode),
            "max_history": int(args.max_history),
            "max_impressions": int(args.max_impressions) if args.max_impressions is not None else None,
        },
        "stats": {
            **news_stats,
            **g,
        },
        "metrics": {},
    }

    def _safe_div(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    for name in mats:
        m = metrics[name]
        results["metrics"][name] = {
            "AUC": _safe_div(float(m["auc_sum"]), float(m["auc_count"])),
            "MRR": _safe_div(float(m["mrr_sum"]), float(m["mrr_count"])),
            "AUC_user_nonzero": _safe_div(float(m["auc_sum_user_nonzero"]), float(m["auc_count_user_nonzero"])),
            "MRR_user_nonzero": _safe_div(float(m["mrr_sum_user_nonzero"]), float(m["mrr_count_user_nonzero"])),
            "AUC_pos_has_vec": _safe_div(float(m["auc_sum_pos_has_vec"]), float(m["auc_count_pos_has_vec"])),
            "MRR_pos_has_vec": _safe_div(float(m["mrr_sum_pos_has_vec"]), float(m["mrr_count_pos_has_vec"])),
            "AUC_user_nonzero_pos_has_vec": _safe_div(
                float(m["auc_sum_user_nonzero_pos_has_vec"]), float(m["auc_count_user_nonzero_pos_has_vec"])
            ),
            "MRR_user_nonzero_pos_has_vec": _safe_div(
                float(m["mrr_sum_user_nonzero_pos_has_vec"]), float(m["mrr_count_user_nonzero_pos_has_vec"])
            ),
            "n_impressions": int(m["mrr_count"]),
        }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

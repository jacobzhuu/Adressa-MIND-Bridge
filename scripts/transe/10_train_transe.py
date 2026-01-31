#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _file_meta(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "path": str(path),
        "sha1": _sha1_file(path),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _is_qid(x: str) -> bool:
    return x.startswith("Q") and x[1:].isdigit()


def _base_pid(rel: str) -> str:
    return rel[:-4] if rel.endswith("_inv") else rel


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


def _load_entity_vec_subset(paths: list[Path], keep_keys: set[str]) -> dict[str, np.ndarray]:
    """
    Load only vectors for keys in keep_keys from one or more MIND entity_embedding.vec files.
    Later files fill missing keys.
    """
    out: dict[str, np.ndarray] = {}
    remaining = set(keep_keys)
    for path in paths:
        if not remaining:
            break
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split()
                if not parts:
                    continue
                key = parts[0]
                if key not in remaining:
                    continue
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
                out[key] = vec
                remaining.remove(key)
                if not remaining:
                    break
    return out


@dataclass(frozen=True)
class _Triples:
    h: np.ndarray  # int64
    r: np.ndarray  # int64
    t: np.ndarray  # int64
    num_triples_raw: int
    num_triples_kept: int
    num_triples_skipped_unknown_rel: int


def _parse_triples(
    *,
    kg_triples: Path,
    seed_qids: list[str],
    rel_vec: dict[str, np.ndarray],
    allow_neighbors: bool,
    rng: np.random.Generator,
) -> tuple[_Triples, list[str], list[str], dict[str, int], dict[str, int]]:
    seed_set = set(seed_qids)

    # First pass: collect entities + relations encountered.
    entities_in_triples: set[str] = set()
    relations_in_triples: set[str] = set()
    raw = 0
    skipped_unknown_rel = 0
    triples_text: list[tuple[str, str, str]] = []

    for ln, line in enumerate(kg_triples.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        raw += 1
        parts = s.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid triple at {kg_triples}:{ln}: {line!r}")
        h, r, t = parts[0], parts[1], parts[2]
        if not (_is_qid(h) and _is_qid(t)):
            continue
        base = _base_pid(r)
        if base not in rel_vec:
            skipped_unknown_rel += 1
            continue
        if h not in seed_set and (not allow_neighbors):
            continue
        if t not in seed_set and (not allow_neighbors):
            continue
        entities_in_triples.add(h)
        entities_in_triples.add(t)
        relations_in_triples.add(r)
        triples_text.append((h, r, t))

    neighbor_qids = sorted([q for q in entities_in_triples if q not in seed_set])
    all_entities = list(seed_qids) + neighbor_qids
    ent2idx = {q: i for i, q in enumerate(all_entities)}

    rel_ids = sorted(relations_in_triples)
    rel2idx = {r: i for i, r in enumerate(rel_ids)}

    # Materialize index arrays.
    h_idx: list[int] = []
    r_idx: list[int] = []
    t_idx: list[int] = []
    for h, r, t in triples_text:
        hi = ent2idx.get(h)
        ti = ent2idx.get(t)
        ri = rel2idx.get(r)
        if hi is None or ti is None or ri is None:
            continue
        h_idx.append(hi)
        r_idx.append(ri)
        t_idx.append(ti)

    # Shuffle once for a bit of mixing (DataLoader will shuffle too).
    if h_idx:
        order = rng.permutation(len(h_idx))
        h_arr = np.asarray(h_idx, dtype=np.int64)[order]
        r_arr = np.asarray(r_idx, dtype=np.int64)[order]
        t_arr = np.asarray(t_idx, dtype=np.int64)[order]
    else:
        h_arr = np.zeros((0,), dtype=np.int64)
        r_arr = np.zeros((0,), dtype=np.int64)
        t_arr = np.zeros((0,), dtype=np.int64)

    triples = _Triples(
        h=h_arr,
        r=r_arr,
        t=t_arr,
        num_triples_raw=int(raw),
        num_triples_kept=int(h_arr.size),
        num_triples_skipped_unknown_rel=int(skipped_unknown_rel),
    )
    return triples, all_entities, neighbor_qids, ent2idx, rel2idx


def _build_relation_embedding(rel_ids: list[str], rel_vec: dict[str, np.ndarray], *, dim: int) -> torch.Tensor:
    mat = np.zeros((len(rel_ids), dim), dtype=np.float32)
    for i, rid in enumerate(rel_ids):
        base = _base_pid(rid)
        v = rel_vec.get(base)
        if v is None:
            continue
        if v.shape[0] != dim:
            raise ValueError(f"Relation dim mismatch for {base}: got {v.shape[0]}, expected {dim}")
        mat[i] = (-v) if rid.endswith("_inv") else v
    return torch.from_numpy(mat)

def _build_relation_matrix_np(rel_ids: list[str], rel_vec: dict[str, np.ndarray], *, dim: int) -> np.ndarray:
    mat = np.zeros((len(rel_ids), dim), dtype=np.float32)
    for i, rid in enumerate(rel_ids):
        base = _base_pid(rid)
        v = rel_vec.get(base)
        if v is None:
            continue
        if v.shape[0] != dim:
            raise ValueError(f"Relation dim mismatch for {base}: got {v.shape[0]}, expected {dim}")
        mat[i] = (-v) if rid.endswith("_inv") else v
    return mat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransE on Wikidata subgraph with anchor-locked MIND alignment.")
    p.add_argument("--kg_triples", type=Path, required=True, help="kg_triples.txt from 09_fetch_wikidata_triples.py")
    p.add_argument("--seed_entity_vocab", type=Path, required=True, help="entity_vocab.txt from step 04 (export order).")
    p.add_argument("--entity_init", type=Path, required=True, help="entity_init.npy from step 04.")
    p.add_argument("--entity_init_mask", type=Path, required=True, help="entity_init_mask.npy from step 04 (anchors==1).")
    p.add_argument(
        "--mind_entity_vec",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional: one or more MIND entity_embedding.vec paths. If provided and --allow_neighbors is enabled, "
            "neighbor QIDs found in these vecs are initialized and frozen as extra anchors."
        ),
    )
    p.add_argument("--mind_relation_vec", type=Path, required=True, help="MIND relation_embedding.vec (PID + 100 dims).")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--neg_ratio", type=int, default=1, help="Negatives per positive triple.")
    p.add_argument(
        "--filtered_negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filtered negative sampling: resample corrupted triples that collide with true triples.",
    )
    p.add_argument(
        "--neg_resample_max",
        type=int,
        default=50,
        help="Max resample attempts per batch for filtered negatives (higher is slower but cleaner).",
    )
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_entity_norm", type=float, default=1.0, help="Entity norm clipping (<=0 to disable).")
    p.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0, mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_seed_coverage", type=float, default=0.001, help="Fallback if seed_triple_coverage < this.")
    p.add_argument("--allow_neighbors", action=argparse.BooleanOptionalAction, default=True, help="Allow training with neighbor entities.")
    p.add_argument("--anchor_eps", type=float, default=1e-6, help="Max allowed anchor drift (Linf).")
    p.add_argument(
        "--init_from_anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize non-anchor entity vectors from anchored neighbors using TransE equations (1-hop).",
    )
    p.add_argument(
        "--relation_weighting",
        choices=["none", "sqrt_inv"],
        default="none",
        help="Optional loss weighting by inverse relation frequency (downweight very common relations).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.txt"
    cfg_path = args.output_dir / "config.json"
    out_npy = args.output_dir / "entity_trained.npy"

    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Config / reproducibility record.
    config: dict[str, object] = {
        "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "margin": float(args.margin),
            "batch_size": int(args.batch_size),
            "neg_ratio": int(args.neg_ratio),
            "filtered_negatives": bool(args.filtered_negatives),
            "neg_resample_max": int(args.neg_resample_max),
            "weight_decay": float(args.weight_decay),
            "max_entity_norm": float(args.max_entity_norm),
            "device": str(args.device) if args.device else None,
            "seed": int(args.seed),
            "min_seed_coverage": float(args.min_seed_coverage),
            "allow_neighbors": bool(args.allow_neighbors),
            "anchor_eps": float(args.anchor_eps),
            "init_from_anchors": bool(args.init_from_anchors),
            "relation_weighting": str(args.relation_weighting),
            "mind_entity_vec": [str(p) for p in (args.mind_entity_vec or [])],
        },
        "inputs": {
            "kg_triples": _file_meta(args.kg_triples),
            "seed_entity_vocab": _file_meta(args.seed_entity_vocab),
            "entity_init": _file_meta(args.entity_init),
            "entity_init_mask": _file_meta(args.entity_init_mask),
            "mind_entity_vec": [_file_meta(p) for p in (args.mind_entity_vec or [])],
            "mind_relation_vec": _file_meta(args.mind_relation_vec),
        },
        "stats": {},
    }
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    seed_qids = [q for q in _read_lines(args.seed_entity_vocab) if _is_qid(q)]
    if not seed_qids:
        raise ValueError("seed_entity_vocab is empty (no QIDs).")

    entity_init = np.load(args.entity_init).astype(np.float32)
    entity_mask = np.load(args.entity_init_mask)
    if entity_init.ndim != 2:
        raise ValueError(f"entity_init must be 2D, got {entity_init.shape}")
    if entity_init.shape[0] != len(seed_qids):
        raise ValueError(f"Row mismatch: vocab has {len(seed_qids)} QIDs, init has {entity_init.shape[0]} rows")
    dim = int(entity_init.shape[1])
    if dim != 100:
        raise ValueError(f"Embedding dim must be 100 to match MIND, got {dim}")
    if entity_mask.shape[0] != len(seed_qids):
        raise ValueError(f"Mask length mismatch: vocab has {len(seed_qids)} QIDs, mask has {entity_mask.shape[0]} rows")

    anchor_idx_seed = np.where(entity_mask.astype(np.int32) == 1)[0].astype(np.int64)
    num_seed_anchor_entities = int(anchor_idx_seed.size)

    rel_vec = _load_relation_vec(args.mind_relation_vec)
    rng = np.random.default_rng(int(args.seed))
    triples, all_entities, neighbor_qids, ent2idx, rel2idx = _parse_triples(
        kg_triples=args.kg_triples,
        seed_qids=seed_qids,
        rel_vec=rel_vec,
        allow_neighbors=bool(args.allow_neighbors),
        rng=rng,
    )

    rel_ids = [r for r, _ in sorted(rel2idx.items(), key=lambda x: x[1])]

    # Extra anchors from neighbors (if neighbor subgraph enabled and MIND vecs provided).
    neighbor_mind_vecs: dict[str, np.ndarray] = {}
    neighbor_anchor_idx_all = np.zeros((0,), dtype=np.int64)
    if args.mind_entity_vec and neighbor_qids and bool(args.allow_neighbors):
        neighbor_mind_vecs = _load_entity_vec_subset([Path(p) for p in args.mind_entity_vec], set(neighbor_qids))
        if neighbor_mind_vecs:
            idxs: list[int] = []
            for j, qid in enumerate(neighbor_qids):
                if qid in neighbor_mind_vecs:
                    idxs.append(len(seed_qids) + j)
            neighbor_anchor_idx_all = np.asarray(idxs, dtype=np.int64)

    num_neighbor_anchor_entities = int(neighbor_anchor_idx_all.size)
    num_anchor_entities_total = int(num_seed_anchor_entities + num_neighbor_anchor_entities)

    seeds_with_triples: set[str] = set()
    if triples.h.size:
        seed_set = set(seed_qids)
        inv_ent = {i: q for q, i in ent2idx.items()}
        for hi, ti in zip(triples.h.tolist(), triples.t.tolist()):
            hq = inv_ent.get(int(hi))
            tq = inv_ent.get(int(ti))
            if hq in seed_set:
                seeds_with_triples.add(hq)
            if tq in seed_set:
                seeds_with_triples.add(tq)

    num_seed_with_triples = int(len(seeds_with_triples))
    seed_triple_coverage = float(num_seed_with_triples) / float(len(seed_qids)) if seed_qids else 0.0

    # Anchor-with-triples counts: use indices for both seed anchors and neighbor anchors.
    if triples.h.size:
        used_idx = set(triples.h.tolist()) | set(triples.t.tolist())
    else:
        used_idx = set()
    num_seed_anchors_with_triples = int(sum(1 for i in anchor_idx_seed.tolist() if i in used_idx))
    num_neighbor_anchors_with_triples = int(sum(1 for i in neighbor_anchor_idx_all.tolist() if i in used_idx))
    num_anchor_with_triples_total = int(num_seed_anchors_with_triples + num_neighbor_anchors_with_triples)

    config["stats"] = {
        "num_seed_entities": int(len(seed_qids)),
        "num_seed_with_triples": int(num_seed_with_triples),
        "seed_triple_coverage": float(seed_triple_coverage),
        "num_seed_anchor_entities": int(num_seed_anchor_entities),
        "num_seed_anchors_with_triples": int(num_seed_anchors_with_triples),
        "num_neighbor_anchor_entities": int(num_neighbor_anchor_entities),
        "num_neighbor_anchors_with_triples": int(num_neighbor_anchors_with_triples),
        "num_anchor_entities_total": int(num_anchor_entities_total),
        "num_anchor_with_triples_total": int(num_anchor_with_triples_total),
        "num_neighbor_entities": int(len(neighbor_qids)),
        "num_entities_total": int(len(all_entities)),
        "num_triples_raw": int(triples.num_triples_raw),
        "num_triples_kept": int(triples.num_triples_kept),
        "num_triples_skipped_unknown_rel": int(triples.num_triples_skipped_unknown_rel),
        "neighbor_mind_anchor_hits": int(len(neighbor_mind_vecs)),
        "neighbor_mind_anchor_coverage": (float(len(neighbor_mind_vecs)) / float(len(neighbor_qids))) if neighbor_qids else 0.0,
    }
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(
        f"seed_entities={len(seed_qids)} "
        f"seed_anchors={num_seed_anchor_entities} neighbor_anchors={num_neighbor_anchor_entities} "
        f"neighbors={len(neighbor_qids)}"
    )
    log(
        "coverage: "
        f"seed_with_triples={num_seed_with_triples}/{len(seed_qids)} ({seed_triple_coverage:.4f}), "
        f"anchors_with_triples={num_anchor_with_triples_total}/{num_anchor_entities_total if num_anchor_entities_total else 0}"
    )
    log(
        f"triples: raw={triples.num_triples_raw} kept={triples.num_triples_kept} "
        f"skipped_unknown_rel={triples.num_triples_skipped_unknown_rel}"
    )
    if neighbor_qids and args.mind_entity_vec:
        log(
            f"neighbor_mind_anchors={len(neighbor_mind_vecs)}/{len(neighbor_qids)} "
            f"({(len(neighbor_mind_vecs)/len(neighbor_qids)):.3f})"
        )

    # Fallback conditions.
    fallback_reasons: list[str] = []
    if triples.num_triples_kept == 0:
        fallback_reasons.append("kg_triples is empty after filtering")
    if num_anchor_entities_total == 0:
        fallback_reasons.append("no anchor entities (seed anchors empty and no neighbor anchors from mind_entity_vec)")
    if num_anchor_entities_total > 0 and num_anchor_with_triples_total == 0:
        fallback_reasons.append("no anchors appear in kg_triples (cannot lock coordinate system)")
    if seed_triple_coverage < float(args.min_seed_coverage):
        fallback_reasons.append(
            f"seed_triple_coverage {seed_triple_coverage:.6f} < min_seed_coverage {float(args.min_seed_coverage):.6f}"
        )

    if fallback_reasons:
        log("FALLBACK: writing entity_init as entity_trained.npy")
        for r in fallback_reasons:
            log(f"  - {r}")
        np.save(out_npy, entity_init.astype(np.float32))
        config["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        config["fallback"] = True
        config["fallback_reasons"] = fallback_reasons
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    # Device selection.
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    log(f"device={device}")

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    # Build initial entity matrix (seed + optional neighbors).
    seed_init_np = entity_init.astype(np.float32, copy=True)
    num_neighbors = int(len(all_entities) - len(seed_qids))
    neighbor_init_np: np.ndarray | None = None
    if num_neighbors > 0:
        rng_init = np.random.default_rng(int(args.seed))
        neighbor_init_np = rng_init.normal(0.0, 0.01, size=(num_neighbors, dim)).astype(np.float32)
        # If provided, initialize neighbor QIDs that exist in MIND and freeze them as anchors.
        if neighbor_mind_vecs:
            for j, qid in enumerate(neighbor_qids):
                vec = neighbor_mind_vecs.get(qid)
                if vec is None:
                    continue
                if vec.shape[0] != dim:
                    raise ValueError(f"Neighbor dim mismatch for {qid}: got {vec.shape[0]}, expected {dim}")
                neighbor_init_np[j] = vec.astype(np.float32, copy=False)

    # Anchor indices over all_entities (seed anchors + neighbor anchors from MIND vecs).
    if anchor_idx_seed.size or neighbor_anchor_idx_all.size:
        anchor_idx_all_np = np.concatenate([a for a in (anchor_idx_seed, neighbor_anchor_idx_all) if a.size], axis=0)
    else:
        anchor_idx_all_np = np.zeros((0,), dtype=np.int64)

    # Optional: initialize non-anchor entities from anchored neighbors (1-hop).
    init_from_anchors_stats: dict[str, float] = {}
    if bool(args.init_from_anchors) and triples.num_triples_kept > 0 and anchor_idx_all_np.size > 0 and rel_ids:
        rel_mat_np = _build_relation_matrix_np(rel_ids, rel_vec, dim=dim)
        num_total = int(len(all_entities))
        init_all_np = seed_init_np if neighbor_init_np is None else np.concatenate([seed_init_np, neighbor_init_np], axis=0)

        sum_vec = np.zeros((num_total, dim), dtype=np.float32)
        cnt = np.zeros((num_total,), dtype=np.int32)
        is_anchor = np.zeros((num_total,), dtype=bool)
        is_anchor[anchor_idx_all_np] = True

        for hi, ri, ti in zip(triples.h.tolist(), triples.r.tolist(), triples.t.tolist()):
            rvec = rel_mat_np[int(ri)]
            hi_a = bool(is_anchor[int(hi)])
            ti_a = bool(is_anchor[int(ti)])
            if hi_a and (not ti_a):
                # t ≈ h + r
                sum_vec[int(ti)] += init_all_np[int(hi)] + rvec
                cnt[int(ti)] += 1
            elif ti_a and (not hi_a):
                # h ≈ t - r
                sum_vec[int(hi)] += init_all_np[int(ti)] - rvec
                cnt[int(hi)] += 1

        updated_seed = 0
        votes_sum_seed = 0
        updated_nei = 0
        votes_sum_nei = 0
        for i in range(num_total):
            if is_anchor[i]:
                continue
            c = int(cnt[i])
            if c <= 0:
                continue
            init_all_np[i] = (sum_vec[i] / float(c)).astype(np.float32)
            if i < len(seed_qids):
                updated_seed += 1
                votes_sum_seed += c
            else:
                updated_nei += 1
                votes_sum_nei += c

        seed_init_np = init_all_np[: len(seed_qids)].astype(np.float32, copy=False)
        if neighbor_init_np is not None:
            neighbor_init_np = init_all_np[len(seed_qids) :].astype(np.float32, copy=False)

        init_from_anchors_stats = {
            "seed_nonanchor_init_from_anchors": float(updated_seed),
            "seed_nonanchor_init_votes_avg": (float(votes_sum_seed) / float(updated_seed)) if updated_seed else 0.0,
            "neighbor_init_from_anchors": float(updated_nei),
            "neighbor_init_votes_avg": (float(votes_sum_nei) / float(updated_nei)) if updated_nei else 0.0,
        }
        log(
            "init_from_anchors: "
            f"seed_updated={updated_seed}, seed_votes_avg={init_from_anchors_stats['seed_nonanchor_init_votes_avg']:.2f}, "
            f"neighbor_updated={updated_nei}"
        )
        config["stats"]["init_from_anchors"] = init_from_anchors_stats
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    seed_weight = torch.from_numpy(seed_init_np).to(device)
    if num_neighbors > 0:
        if neighbor_init_np is not None:
            neighbor_weight = torch.from_numpy(neighbor_init_np).to(device)
        else:
            neighbor_weight = torch.randn((num_neighbors, dim), device=device, dtype=torch.float32) * 0.01
        init_weight = torch.cat([seed_weight, neighbor_weight], dim=0)
    else:
        init_weight = seed_weight.clone()

    emb = torch.nn.Embedding.from_pretrained(init_weight, freeze=False)

    anchor_idx_all = torch.from_numpy(anchor_idx_all_np).to(device)
    anchor_init_all = init_weight.index_select(0, anchor_idx_all).clone()

    rel_emb = _build_relation_embedding(rel_ids, rel_vec, dim=dim).to(device)

    dataset = TensorDataset(
        torch.from_numpy(triples.h),
        torch.from_numpy(triples.r),
        torch.from_numpy(triples.t),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(emb.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    rel_weight: torch.Tensor | None = None
    if str(args.relation_weighting) != "none" and triples.r.size:
        freq = np.bincount(triples.r.astype(np.int64), minlength=len(rel_ids)).astype(np.float32)
        if str(args.relation_weighting) == "sqrt_inv":
            w = 1.0 / np.sqrt(np.maximum(freq, 1.0))
        else:
            w = np.ones_like(freq)
        # Normalize to mean=1 for stable loss scale.
        w = (w / float(np.mean(w))) if w.size else w
        rel_weight = torch.from_numpy(w.astype(np.float32)).to(device)
        log(f"relation_weighting={args.relation_weighting} (mean={float(w.mean()) if w.size else 1.0:.3f})")
        config["stats"]["relation_weighting"] = {
            "mode": str(args.relation_weighting),
            "num_relations": int(len(rel_ids)),
            "min_freq": float(freq.min()) if freq.size else 0.0,
            "max_freq": float(freq.max()) if freq.size else 0.0,
        }
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def score(h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (h + r - t).abs().sum(dim=-1)

    num_entities_total = int(init_weight.shape[0])
    if num_entities_total < 2:
        raise RuntimeError("Need at least 2 entities for negative sampling.")

    # Filtered negative sampling precompute (CPU): encode triples into sortable ids for fast membership check.
    neg_rng = np.random.default_rng(int(args.seed))
    num_relations_total = int(len(rel_ids))
    true_ids_sorted: np.ndarray | None = None
    use_filtered_neg = bool(args.filtered_negatives) and (triples.num_triples_kept > 0)
    if use_filtered_neg:
        true_ids = ((triples.h * num_relations_total + triples.r) * num_entities_total + triples.t).astype(np.int64, copy=False)
        true_ids_sorted = np.sort(true_ids)

    def _is_true_triple_ids(ids: np.ndarray) -> np.ndarray:
        assert true_ids_sorted is not None
        pos = np.searchsorted(true_ids_sorted, ids)
        ok = pos < true_ids_sorted.size
        if not np.any(ok):
            return ok
        pos_safe = np.clip(pos, 0, true_ids_sorted.size - 1)
        ok &= true_ids_sorted[pos_safe] == ids
        return ok

    for epoch in range(1, int(args.epochs) + 1):
        emb.train()
        total_loss = 0.0
        total_count = 0
        total_neg = 0
        total_collide0 = 0
        total_resampled = 0
        for h_idx, r_idx, t_idx in loader:
            # Keep CPU copies for negative sampling; move to device for embedding lookup.
            h_idx_cpu = h_idx.numpy()
            r_idx_cpu = r_idx.numpy()
            t_idx_cpu = t_idx.numpy()

            h_idx = h_idx.to(device, non_blocking=True)
            r_idx = r_idx.to(device, non_blocking=True)
            t_idx = t_idx.to(device, non_blocking=True)

            h = emb(h_idx)
            r = rel_emb.index_select(0, r_idx)
            t = emb(t_idx)
            pos_score = score(h, r, t)

            neg_ratio = int(args.neg_ratio)
            if neg_ratio < 1:
                raise ValueError("--neg_ratio must be >= 1")
            num_pos = int(pos_score.shape[0])
            num_neg = num_pos * neg_ratio

            # Negative sampling (CPU; supports filtered negatives).
            h_rep = np.repeat(h_idx_cpu.astype(np.int64, copy=False), neg_ratio)
            r_rep = np.repeat(r_idx_cpu.astype(np.int64, copy=False), neg_ratio)
            t_rep = np.repeat(t_idx_cpu.astype(np.int64, copy=False), neg_ratio)
            total_neg += int(h_rep.size)

            corrupt_head = neg_rng.random(int(h_rep.size)) < 0.5
            new_ent = neg_rng.integers(0, num_entities_total, size=int(h_rep.size), dtype=np.int64)
            orig_ent = np.where(corrupt_head, h_rep, t_rep)
            same = new_ent == orig_ent
            if np.any(same):
                new_ent[same] = (new_ent[same] + 1) % num_entities_total

            nh_np = np.where(corrupt_head, new_ent, h_rep).astype(np.int64, copy=False)
            nr_np = r_rep.astype(np.int64, copy=False)
            nt_np = np.where(corrupt_head, t_rep, new_ent).astype(np.int64, copy=False)

            if use_filtered_neg and true_ids_sorted is not None and true_ids_sorted.size and int(args.neg_resample_max) > 0:
                ids0 = ((nh_np * num_relations_total + nr_np) * num_entities_total + nt_np).astype(np.int64, copy=False)
                bad = _is_true_triple_ids(ids0)
                total_collide0 += int(bad.sum())
                attempt = 0
                while np.any(bad) and attempt < int(args.neg_resample_max):
                    idx = np.flatnonzero(bad)
                    total_resampled += int(idx.size)

                    new2 = neg_rng.integers(0, num_entities_total, size=int(idx.size), dtype=np.int64)
                    orig2 = np.where(corrupt_head[idx], h_rep[idx], t_rep[idx])
                    same2 = new2 == orig2
                    if np.any(same2):
                        new2[same2] = (new2[same2] + 1) % num_entities_total

                    head_mask = corrupt_head[idx]
                    if np.any(head_mask):
                        nh_np[idx[head_mask]] = new2[head_mask]
                    if np.any(~head_mask):
                        nt_np[idx[~head_mask]] = new2[~head_mask]

                    ids = ((nh_np * num_relations_total + nr_np) * num_entities_total + nt_np).astype(np.int64, copy=False)
                    bad = _is_true_triple_ids(ids)
                    attempt += 1

            nh = torch.from_numpy(nh_np).to(device, non_blocking=True)
            nr = torch.from_numpy(nr_np).to(device, non_blocking=True)
            nt = torch.from_numpy(nt_np).to(device, non_blocking=True)

            neg_score = score(emb(nh), rel_emb.index_select(0, nr), emb(nt))
            pos_rep = pos_score.repeat_interleave(neg_ratio)
            per = torch.relu(float(args.margin) + pos_rep - neg_score)
            if rel_weight is not None:
                w = rel_weight.index_select(0, nr).detach()
                loss = (per * w).sum() / (w.sum() + 1e-12)
            else:
                loss = per.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                if float(args.max_entity_norm) > 0:
                    w = emb.weight.data
                    norms = torch.linalg.norm(w, ord=2, dim=1, keepdim=True)
                    scale = torch.clamp(float(args.max_entity_norm) / (norms + 1e-12), max=1.0)
                    w.mul_(scale)
                # Restore anchors exactly (avoid drift from weight decay / clipping).
                emb.weight.data.index_copy_(0, anchor_idx_all, anchor_init_all)

            bs = int(num_pos)
            total_loss += float(loss.detach().cpu().item()) * bs
            total_count += bs

        avg_loss = total_loss / max(1, total_count)
        if use_filtered_neg and total_neg > 0:
            frac = float(total_collide0) / float(total_neg)
            log(
                f"epoch={epoch} loss={avg_loss:.6f} "
                f"neg_filtered_collide0={total_collide0}/{total_neg} ({frac:.4f}) "
                f"neg_resampled={total_resampled}"
            )
        else:
            log(f"epoch={epoch} loss={avg_loss:.6f}")

    emb.eval()
    with torch.no_grad():
        trained_seed = emb.weight[: len(seed_qids)].detach().cpu().numpy().astype(np.float32)

    # Anchor invariance check (seed anchors + neighbor anchors).
    if int(anchor_idx_all.numel()) > 0:
        with torch.no_grad():
            max_diff_all = float((emb.weight.index_select(0, anchor_idx_all) - anchor_init_all).abs().max().item())
    else:
        max_diff_all = 0.0
    log(f"anchor_max_abs_diff_all={max_diff_all:.8f}")
    if max_diff_all > float(args.anchor_eps):
        raise RuntimeError(
            f"Anchor drift too large: max_abs_diff={max_diff_all} > anchor_eps={float(args.anchor_eps)}. "
            "This violates anchor-locked requirement."
        )

    np.save(out_npy, trained_seed)
    config["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["fallback"] = False
    config["anchor_max_abs_diff_all"] = float(max_diff_all)
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

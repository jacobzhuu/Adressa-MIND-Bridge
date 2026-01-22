#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.news_tsv import iter_news_tsv


@dataclass(frozen=True)
class MentionSample:
    title: str
    start: int
    end: int
    qid_index: int


def _load_vocab(path: Path) -> dict[str, int]:
    qids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {qid: i for i, qid in enumerate(qids)}


def _iter_mention_samples(news_tsv: Path, vocab: dict[str, int]) -> list[MentionSample]:
    samples: list[MentionSample] = []
    for row in iter_news_tsv(news_tsv):
        te = row.title_entities.strip()
        if not te or te == "[]":
            continue
        try:
            ents = json.loads(te)
        except json.JSONDecodeError:
            continue

        for e in ents:
            qid = str(e.get("WikidataId") or "")
            if qid not in vocab:
                continue
            idx = vocab[qid]
            offsets = e.get("OccurrenceOffsets") or []
            surfaces = e.get("SurfaceForms") or []
            for off, surf in zip(offsets, surfaces):
                try:
                    start = int(off)
                except Exception:
                    continue
                surface = str(surf)
                end = start + len(surface)
                if start < 0 or end <= start or end > len(row.title):
                    continue
                if row.title[start:end] != surface:
                    continue
                samples.append(MentionSample(title=row.title, start=start, end=end, qid_index=idx))
    return samples


def _auto_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _precompute_mention_embeddings(
    *,
    samples: list[MentionSample],
    model_name: str,
    max_length: int,
    batch_titles: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'transformers'. Install it first, e.g. `pip install transformers`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    mention_vecs: list[np.ndarray] = []
    mention_qids: list[int] = []

    by_title: dict[str, list[MentionSample]] = {}
    for s in samples:
        by_title.setdefault(s.title, []).append(s)

    titles = list(by_title.keys())
    for start in tqdm(
        range(0, len(titles), batch_titles),
        total=(len(titles) + batch_titles - 1) // batch_titles,
        desc="encode",
        unit="batch",
    ):
        batch = titles[start : start + batch_titles]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping").to("cpu").numpy()
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1].to("cpu")  # (B, T, H)

        for bi, title in enumerate(batch):
            title_offsets = offsets[bi].tolist()
            title_hidden = hidden[bi]  # (T, H)
            for s in by_title[title]:
                token_idx = [
                    ti
                    for ti, (os, oe) in enumerate(title_offsets)
                    if not (os == 0 and oe == 0) and os < s.end and oe > s.start
                ]
                if not token_idx:
                    continue
                vec = title_hidden[token_idx].mean(dim=0).numpy().astype(np.float32)
                mention_vecs.append(vec)
                mention_qids.append(s.qid_index)

    mention_matrix = np.stack(mention_vecs, axis=0) if mention_vecs else np.zeros((0, 768), dtype=np.float32)
    qid_idx = np.asarray(mention_qids, dtype=np.int64)
    return mention_matrix, qid_idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate entity embeddings via mention→entity retrieval (Recall@K, MRR).")
    p.add_argument("--news_tsv", type=Path, required=True)
    p.add_argument(
        "--train_news_tsv",
        type=Path,
        default=None,
        help="Optional: train/news.tsv used to split eval mentions into seen/unseen by entity presence in train.",
    )
    p.add_argument("--entity_vocab", type=Path, required=True)
    p.add_argument("--entity_matrix", type=Path, required=True, help="entity_trained.npy or entity_init.npy")
    p.add_argument("--projection", type=Path, required=True, help="projection.pt from training output_dir")
    p.add_argument("--model", type=str, default="NbAiLab/nb-bert-base-ner")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_titles", type=int, default=16)
    p.add_argument("--batch_mentions", type=int, default=1024)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--save_npz", type=Path, default=None, help="Optional: save mention vectors and labels to .npz")
    return p.parse_args()


def _seen_entity_indices(train_news_tsv: Path, vocab: dict[str, int]) -> set[int]:
    seen: set[int] = set()
    for row in iter_news_tsv(train_news_tsv):
        te = row.title_entities.strip()
        if not te or te == "[]":
            continue
        try:
            ents = json.loads(te)
        except json.JSONDecodeError:
            continue
        for e in ents:
            qid = str(e.get("WikidataId") or "")
            idx = vocab.get(qid)
            if idx is not None:
                seen.add(int(idx))
    return seen


def _eval_retrieval(
    *,
    mention_matrix: np.ndarray,
    qid_idx: np.ndarray,
    entity_t: torch.Tensor,
    proj: nn.Module,
    device: torch.device,
    ks: list[int],
    batch_mentions: int,
) -> tuple[dict[int, float], float]:
    n = int(qid_idx.shape[0])
    if n == 0:
        return {k: 0.0 for k in ks}, 0.0

    correct = {k: 0 for k in ks}
    mrr_sum = 0.0

    with torch.no_grad():
        for start in range(0, n, batch_mentions):
            m = torch.from_numpy(mention_matrix[start : start + batch_mentions]).to(device)
            y = torch.from_numpy(qid_idx[start : start + batch_mentions]).to(device)
            m = proj(m)
            m = nn.functional.normalize(m, dim=-1)
            sim = m @ entity_t.T  # (B, N)

            true_sim = sim.gather(1, y.unsqueeze(1))
            rank = (sim > true_sim).sum(dim=1) + 1
            mrr_sum += float((1.0 / rank.float()).sum().item())

            for k in ks:
                topk = sim.topk(k, dim=1).indices
                correct[k] += int((topk == y.unsqueeze(1)).any(dim=1).sum().item())

    recall = {k: correct[k] / n for k in ks}
    return recall, (mrr_sum / n)


def _print_block(prefix: str, *, qid_idx: np.ndarray, vocab_size: int, recall: dict[int, float], mrr: float) -> None:
    n = int(qid_idx.shape[0])
    uniq = int(np.unique(qid_idx).shape[0]) if n else 0
    print(f"{prefix}mentions={n}")
    print(f"{prefix}unique_entities_in_eval={uniq}")
    print(f"{prefix}entity_vocab={vocab_size}")
    for k, v in recall.items():
        print(f"{prefix}recall@{k}={v:.6f}")
    print(f"{prefix}mrr={mrr:.6f}")


def main() -> None:
    args = parse_args()
    device = _auto_device(args.device)

    vocab = _load_vocab(args.entity_vocab)
    samples = _iter_mention_samples(args.news_tsv, vocab)
    mention_matrix, qid_idx = _precompute_mention_embeddings(
        samples=samples,
        model_name=args.model,
        max_length=args.max_length,
        batch_titles=args.batch_titles,
        device=device,
    )
    if mention_matrix.shape[0] == 0:
        raise RuntimeError("No mention vectors computed for evaluation; check news.tsv:title_entities content.")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_npz, mentions=mention_matrix, qid_idx=qid_idx)

    entity = np.load(args.entity_matrix).astype(np.float32)
    ckpt = torch.load(args.projection, map_location="cpu")

    proj = nn.Linear(mention_matrix.shape[1], entity.shape[1], bias=False)
    proj.load_state_dict(ckpt["proj"])
    proj.to(device)
    proj.eval()

    entity_t = torch.from_numpy(entity).to(device)
    entity_t = nn.functional.normalize(entity_t, dim=-1)

    ks = sorted(set(int(x) for x in args.k))
    full_recall, full_mrr = _eval_retrieval(
        mention_matrix=mention_matrix,
        qid_idx=qid_idx,
        entity_t=entity_t,
        proj=proj,
        device=device,
        ks=ks,
        batch_mentions=args.batch_mentions,
    )
    _print_block("full_", qid_idx=qid_idx, vocab_size=len(vocab), recall=full_recall, mrr=full_mrr)

    if args.train_news_tsv is None:
        return

    seen = _seen_entity_indices(args.train_news_tsv, vocab)
    if seen:
        seen_mask = np.isin(qid_idx, np.fromiter(seen, dtype=np.int64))
    else:
        seen_mask = np.zeros_like(qid_idx, dtype=bool)

    seen_mentions = mention_matrix[seen_mask]
    seen_y = qid_idx[seen_mask]
    unseen_mentions = mention_matrix[~seen_mask]
    unseen_y = qid_idx[~seen_mask]

    seen_recall, seen_mrr = _eval_retrieval(
        mention_matrix=seen_mentions,
        qid_idx=seen_y,
        entity_t=entity_t,
        proj=proj,
        device=device,
        ks=ks,
        batch_mentions=args.batch_mentions,
    )
    _print_block("seen_", qid_idx=seen_y, vocab_size=len(vocab), recall=seen_recall, mrr=seen_mrr)

    unseen_recall, unseen_mrr = _eval_retrieval(
        mention_matrix=unseen_mentions,
        qid_idx=unseen_y,
        entity_t=entity_t,
        proj=proj,
        device=device,
        ks=ks,
        batch_mentions=args.batch_mentions,
    )
    _print_block("unseen_", qid_idx=unseen_y, vocab_size=len(vocab), recall=unseen_recall, mrr=unseen_mrr)


if __name__ == "__main__":
    main()

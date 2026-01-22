#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.news_tsv import iter_news_tsv


def _load_vocab(path: Path) -> dict[str, int]:
    qids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {qid: i for i, qid in enumerate(qids)}


def _sha1_text(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _precompute_meta(*, args: argparse.Namespace, vocab_path: Path, news_tsv: Path) -> dict[str, object]:
    st = news_tsv.stat()
    return {
        "entity_vocab_sha1": _sha1_text(vocab_path),
        "news_tsv_path": str(news_tsv),
        "news_tsv_size": int(st.st_size),
        "news_tsv_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "model": str(args.model),
        "max_length": int(args.max_length),
        "batch_titles": int(args.batch_titles),
    }


def _meta_matches(existing: dict[str, object], expected: dict[str, object]) -> bool:
    keys = [
        "entity_vocab_sha1",
        "news_tsv_path",
        "news_tsv_size",
        "news_tsv_mtime_ns",
        "model",
        "max_length",
        "batch_titles",
    ]
    return all(existing.get(k) == expected.get(k) for k in keys)


@dataclass(frozen=True)
class MentionSample:
    title: str
    start: int
    end: int
    qid_index: int


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


def _precompute_mention_embeddings(
    *,
    samples: list[MentionSample],
    model_name: str,
    max_length: int,
    batch_titles: int,
    device: torch.device,
    output_prefix: Path,
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

    # Group by title to avoid re-encoding the same title multiple times.
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

    np.save(output_prefix.with_suffix(".mentions.npy"), mention_matrix)
    np.save(output_prefix.with_suffix(".qid_idx.npy"), qid_idx)
    return mention_matrix, qid_idx


class MentionVecDataset(Dataset):
    def __init__(self, mention_matrix: np.ndarray, qid_idx: np.ndarray) -> None:
        if mention_matrix.shape[0] != qid_idx.shape[0]:
            raise ValueError("mention_matrix and qid_idx must have same length")
        self._m = torch.from_numpy(mention_matrix)
        self._y = torch.from_numpy(qid_idx)

    def __len__(self) -> int:
        return int(self._y.shape[0])

    def __getitem__(self, idx: int):
        return self._m[idx], self._y[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Norwegian entity embeddings initialized from MIND.")
    p.add_argument("--news_tsv", type=Path, required=True, help="Adressa train/news.tsv with title_entities.")
    p.add_argument("--entity_vocab", type=Path, required=True, help="entity_vocab.txt from step 04.")
    p.add_argument("--entity_init", type=Path, required=True, help="entity_init.npy from step 04.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="NbAiLab/nb-bert-base-ner", help="NbBERT model for contextual encoding.")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_titles", type=int, default=16, help="How many unique titles to encode per forward pass.")
    p.add_argument("--batch_size", type=int, default=256, help="Training batch size (mention vectors).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--negatives", type=int, default=50)
    p.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda, cuda:0")
    p.add_argument("--reuse_precomputed", action="store_true", help="Reuse precomputed mention vectors if present.")
    p.add_argument("--append_log", action="store_true", help="Append to train_log.txt instead of overwriting it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    vocab = _load_vocab(args.entity_vocab)

    pre_prefix = args.output_dir / "nbbert"
    mentions_path = pre_prefix.with_suffix(".mentions.npy")
    qid_idx_path = pre_prefix.with_suffix(".qid_idx.npy")
    meta_path = pre_prefix.with_suffix(".precompute_meta.json")

    expected_meta = _precompute_meta(args=args, vocab_path=args.entity_vocab, news_tsv=args.news_tsv)
    loaded_precomputed = False

    if args.reuse_precomputed and mentions_path.exists() and qid_idx_path.exists() and meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}
        if _meta_matches(existing_meta, expected_meta):
            mention_matrix = np.load(mentions_path)
            qid_idx = np.load(qid_idx_path)
            loaded_precomputed = True

    if not loaded_precomputed:
        samples = _iter_mention_samples(args.news_tsv, vocab)
        mention_matrix, qid_idx = _precompute_mention_embeddings(
            samples=samples,
            model_name=args.model,
            max_length=args.max_length,
            batch_titles=args.batch_titles,
            device=device,
            output_prefix=pre_prefix,
        )
        meta_path.write_text(json.dumps(expected_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Safety: if cache is stale / incompatible, recompute once.
    if qid_idx.size:
        max_idx = int(qid_idx.max())
        if max_idx >= len(vocab):
            samples = _iter_mention_samples(args.news_tsv, vocab)
            mention_matrix, qid_idx = _precompute_mention_embeddings(
                samples=samples,
                model_name=args.model,
                max_length=args.max_length,
                batch_titles=args.batch_titles,
                device=device,
                output_prefix=pre_prefix,
            )
            meta_path.write_text(json.dumps(expected_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if mention_matrix.shape[0] == 0:
        raise RuntimeError("No mention vectors computed; check title_entities content.")

    entity_init = np.load(args.entity_init).astype(np.float32)
    num_entities, dim = entity_init.shape
    hidden_dim = int(mention_matrix.shape[1])

    entity_emb = nn.Embedding(num_entities, dim)
    entity_emb.weight.data.copy_(torch.from_numpy(entity_init))
    proj = nn.Linear(hidden_dim, dim, bias=False)

    entity_emb.to(device)
    proj.to(device)

    opt = torch.optim.Adam(list(entity_emb.parameters()) + list(proj.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = MentionVecDataset(mention_matrix, qid_idx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    log_path = args.output_dir / "train_log.txt"
    if not args.append_log:
        log_path.write_text("", encoding="utf-8")

    def sample_negatives(pos: torch.Tensor, k: int) -> torch.Tensor:
        # Vectorized sampling without collisions: sample in [0, N-2], then shift >=pos.
        if num_entities <= 1:
            raise RuntimeError("Need at least 2 entities for negative sampling.")
        neg = torch.randint(0, num_entities - 1, (pos.shape[0], k), device=pos.device)
        neg = neg + (neg >= pos.unsqueeze(1)).long()
        return neg

    for epoch in range(args.epochs):
        entity_emb.train()
        proj.train()
        total = 0.0
        n = 0
        for m_vec, y in tqdm(dl, desc=f"train e{epoch+1}/{args.epochs}", unit="batch"):
            m_vec = m_vec.to(device)
            y = y.to(device)

            m_proj = proj(m_vec)
            m_proj = nn.functional.normalize(m_proj, dim=-1)

            pos_vec = entity_emb(y)
            pos_vec = nn.functional.normalize(pos_vec, dim=-1)

            neg_idx = sample_negatives(y, args.negatives)
            neg_vec = entity_emb(neg_idx)
            neg_vec = nn.functional.normalize(neg_vec, dim=-1)

            pos_score = (m_proj * pos_vec).sum(dim=-1, keepdim=True)
            neg_score = (m_proj.unsqueeze(1) * neg_vec).sum(dim=-1)
            logits = torch.cat([pos_score, neg_score], dim=1) / args.temperature
            labels = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)

            loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1

        avg = total / max(n, 1)
        with (args.output_dir / "train_log.txt").open("a", encoding="utf-8") as f:
            f.write(f"epoch={epoch+1}\tloss={avg:.6f}\n")

    trained = entity_emb.weight.detach().to("cpu").numpy().astype(np.float32)
    np.save(args.output_dir / "entity_trained.npy", trained)
    torch.save({"proj": proj.state_dict()}, args.output_dir / "projection.pt")


if __name__ == "__main__":
    main()

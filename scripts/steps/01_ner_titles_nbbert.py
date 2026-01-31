#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
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

from adressa_entity.news_tsv import read_news_ids_and_titles
from adressa_entity.ner import heuristic_mentions_from_title, load_token_classification_pipeline, mentions_from_pipeline_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NbBERT NER over Adressa news titles and output mentions JSONL.")
    p.add_argument("--news_tsv", type=Path, required=True, help="Input MIND-format news.tsv (Adressa).")
    p.add_argument("--output_jsonl", type=Path, required=True, help="Output JSONL with extracted mentions per news.")
    p.add_argument(
        "--model",
        type=str,
        default="NbAiLab/nb-bert-base-ner",
        help="HuggingFace model name for Norwegian NER.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument(
        "--aggregation_strategy",
        type=str,
        default="simple",
        choices=["none", "simple", "first", "average", "max"],
        help="TokenClassificationPipeline aggregation strategy (transformers).",
    )
    p.add_argument("--device", type=int, default=None, help="Pipeline device: -1 CPU, 0..N CUDA device.")
    p.add_argument(
        "--heuristic_mode",
        type=str,
        default="fallback",
        choices=["off", "fallback", "merge"],
        help="Heuristic titlecase extractor: off | fallback (only if model empty) | merge (union).",
    )
    p.add_argument("--heuristic_max_mentions", type=int, default=6, help="Max heuristic mentions per title.")
    p.add_argument("--heuristic_score", type=float, default=0.45, help="Pseudo-score assigned to heuristic mentions.")
    p.add_argument("--heuristic_max_span_chars", type=int, default=60, help="Max chars per heuristic mention span.")
    p.add_argument("--heuristic_max_span_tokens", type=int, default=6, help="Max tokens per heuristic mention span.")
    p.add_argument(
        "--heuristic_min_first_token_len",
        type=int,
        default=0,
        help="Allow single-token mention at title start only if token length >= this (unless ALLCAPS/digits/dash).",
    )
    p.add_argument("--max_mentions_per_title", type=int, default=10, help="Final cap on mentions written per title.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    items = read_news_ids_and_titles(args.news_tsv, limit=args.limit)
    ner = load_token_classification_pipeline(
        model_name=args.model,
        device=device,
        max_length=args.max_length,
        aggregation_strategy=args.aggregation_strategy,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(items), args.batch_size), total=(len(items) + args.batch_size - 1) // args.batch_size):
            batch_items = items[start : start + args.batch_size]
            batch_texts = [title for _, title in batch_items]
            try:
                batch_outputs = ner(
                    batch_texts,
                    batch_size=args.batch_size,
                    truncation=True,
                )
            except TypeError:
                # Some transformers versions don't accept tokenizer kwargs (e.g. `truncation`, `max_length`)
                # on TokenClassificationPipeline.__call__(). Titles are typically short, so we keep the call minimal.
                batch_outputs = ner(batch_texts, batch_size=args.batch_size)
            for (news_id, title), ents in zip(batch_items, batch_outputs):
                model_mentions = mentions_from_pipeline_output(title, ents)
                heur_mentions = []
                if args.heuristic_mode != "off":
                    heur_mentions = heuristic_mentions_from_title(
                        title,
                        max_mentions=args.heuristic_max_mentions,
                        score=args.heuristic_score,
                        max_span_chars=args.heuristic_max_span_chars,
                        max_span_tokens=args.heuristic_max_span_tokens,
                        min_first_token_len=args.heuristic_min_first_token_len,
                    )

                if args.heuristic_mode == "merge":
                    by_span = {(m.start, m.end): m for m in model_mentions}
                    for hm in heur_mentions:
                        by_span.setdefault((hm.start, hm.end), hm)
                    mentions = sorted(by_span.values(), key=lambda m: (m.start, m.end))
                elif args.heuristic_mode == "fallback":
                    mentions = model_mentions if model_mentions else heur_mentions
                else:
                    mentions = model_mentions

                if args.max_mentions_per_title and args.max_mentions_per_title > 0:
                    mentions = mentions[: int(args.max_mentions_per_title)]

                record = {
                    "news_id": news_id,
                    "title": title,
                    "mentions": [
                        {
                            "start": m.start,
                            "end": m.end,
                            "surface": m.surface,
                            "ner_type": m.ner_type,
                            "ner_score": m.ner_score,
                            "source": "heuristic" if m.ner_type == "HEUR" else "model",
                        }
                        for m in mentions
                    ],
                }
                out.write(json.dumps(record, ensure_ascii=False))
                out.write("\n")


if __name__ == "__main__":
    main()

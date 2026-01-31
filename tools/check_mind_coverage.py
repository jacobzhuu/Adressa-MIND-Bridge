#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def _discover_news_tsv(data_root: Path) -> list[Path]:
    out: list[Path] = []
    for sp in ("train", "val", "test"):
        p = data_root / sp / "news.tsv"
        if p.exists():
            out.append(p)
    return out


def _discover_behaviors_tsv(data_root: Path, split: str) -> Path:
    p = data_root / split / "behaviors.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return p


def _load_news_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                nid = line.split("\t", 1)[0].strip()
                if nid:
                    ids.add(nid)
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check that behaviors.tsv candidates/history are covered by news.tsv (MIND format).")
    p.add_argument("--data_root", type=Path, default=Path("data/work/adressa_one_week_mind_final"))
    p.add_argument("--eval_split", choices=["train", "val", "test"], default="test")
    p.add_argument(
        "--news_union",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, union news ids from all available splits; else use only eval_split/news.tsv.",
    )
    p.add_argument("--max_lines", type=int, default=None, help="Limit behaviors lines (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    if not data_root.exists():
        legacy = Path("adressa_one_week_mind_final")
        if legacy.exists():
            data_root = legacy
        else:
            raise FileNotFoundError(f"Missing data_root: {args.data_root}")

    news_paths = _discover_news_tsv(data_root)
    if not news_paths:
        raise FileNotFoundError(f"No news.tsv under {data_root}/{{train,val,test}}")
    if not bool(args.news_union):
        only = data_root / args.eval_split / "news.tsv"
        if not only.exists():
            raise FileNotFoundError(f"Missing {only}")
        news_paths = [only]

    news_ids = _load_news_ids(news_paths)
    beh = _discover_behaviors_tsv(data_root, args.eval_split)

    cand_total = 0
    cand_missing = 0
    hist_total = 0
    hist_missing = 0
    lines = 0

    with beh.open("r", encoding="utf-8") as f:
        for line in f:
            if args.max_lines is not None and lines >= int(args.max_lines):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            hist = parts[3].strip()
            impr = parts[4].strip()

            if hist:
                for nid in hist.split():
                    hist_total += 1
                    if nid not in news_ids:
                        hist_missing += 1

            if impr:
                for tok in impr.split():
                    nid = tok.rsplit("-", 1)[0]
                    cand_total += 1
                    if nid not in news_ids:
                        cand_missing += 1

            lines += 1

    def _fmt(missing: int, total: int) -> str:
        if total <= 0:
            return "0/0 (0.0%)"
        return f"{missing}/{total} ({missing/total:.1%})"

    scope = "union(train/val/test)" if bool(args.news_union) else f"{args.eval_split}/news.tsv only"
    print(f"data_root:    {data_root}")
    print(f"eval_split:   {args.eval_split}")
    print(f"news_scope:   {scope}")
    print(f"behaviors:    {beh}")
    print(f"news_ids:     {len(news_ids)}")
    print(f"rows_checked: {lines}")
    print("")
    print("Coverage:")
    print(f"  candidates_missing: {_fmt(cand_missing, cand_total)}")
    print(f"  history_missing:    {_fmt(hist_missing, hist_total)}")


if __name__ == "__main__":
    main()


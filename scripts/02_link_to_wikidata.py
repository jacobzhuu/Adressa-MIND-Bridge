#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.wikidata import WikidataSearcher, ner_type_to_mind_type


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Link NER mentions to Wikidata QIDs via Wikidata search API.")
    p.add_argument("--mentions_jsonl", type=Path, required=True, help="Input JSONL from 01_ner_titles_nbbert.py")
    p.add_argument("--output_jsonl", type=Path, required=True, help="Output JSONL with linked mentions.")
    p.add_argument("--cache_db", type=Path, default=Path("cache/wikidata_search.sqlite"))
    p.add_argument("--lang", type=str, default="nb")
    p.add_argument("--limit_n", type=int, default=10)
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between uncached API calls (seconds).")
    p.add_argument("--trust-env", action=argparse.BooleanOptionalAction, default=True, help="Respect system/env proxy settings.")
    p.add_argument("--max-retries", type=int, default=6, help="Max retries per Wikidata request.")
    p.add_argument("--retry-base-sleep", type=float, default=0.5, help="Base backoff sleep (seconds).")
    p.add_argument("--retry-max-sleep", type=float, default=20.0, help="Max backoff sleep (seconds).")
    p.add_argument("--max-consecutive-errors", type=int, default=100, help="Abort after N consecutive request errors.")
    p.add_argument("--resume", action="store_true", help="Resume by appending to existing output_jsonl.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_jsonl if it exists.")
    p.add_argument("--min_match", type=float, default=0.6, help="Minimum surface↔label similarity to accept.")
    p.add_argument(
        "--min_match_heur",
        type=float,
        default=0.85,
        help="Minimum surface↔label similarity for heuristic mentions (ner_type==HEUR).",
    )
    p.add_argument(
        "--heur_reject_lowercase_label_at_title_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For heuristic single-token mentions at title start (start==0), reject candidates whose Wikidata label starts "
            "with lowercase (often common nouns like 'boligmarked')."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug).")
    return p.parse_args()


def _sanitize_and_count_jsonl(path: Path) -> int:
    """
    Return number of valid JSON lines, truncating trailing partial line if needed.
    """
    if not path.exists():
        return 0
    good_bytes = 0
    count = 0
    with path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                json.loads(line.decode("utf-8"))
            except Exception:
                break
            count += 1
            good_bytes = f.tell()
    if good_bytes and path.stat().st_size != good_bytes:
        with path.open("rb+") as f:
            f.truncate(good_bytes)
    if count == 0 and path.exists() and path.stat().st_size > 0 and good_bytes == 0:
        with path.open("rb+") as f:
            f.truncate(0)
    return count


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    args = parse_args()
    if args.output_jsonl.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"{args.output_jsonl} already exists. Use --resume or --overwrite.")

    total_lines = _count_lines(args.mentions_jsonl)
    skip_lines = 0
    if args.resume and args.output_jsonl.exists() and not args.overwrite:
        skip_lines = _sanitize_and_count_jsonl(args.output_jsonl)

    searcher = WikidataSearcher(
        cache_db_path=args.cache_db,
        sleep_seconds=args.sleep,
        trust_env=args.trust_env,
        max_retries=args.max_retries,
        retry_base_sleep_seconds=args.retry_base_sleep,
        retry_max_sleep_seconds=args.retry_max_sleep,
    )
    try:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_mode = "a" if args.resume and args.output_jsonl.exists() and not args.overwrite else "w"
        consecutive_errors = 0
        with args.mentions_jsonl.open("r", encoding="utf-8") as f_in, args.output_jsonl.open(
            output_mode, encoding="utf-8"
        ) as f_out:
            if skip_lines:
                for _ in range(skip_lines):
                    next(f_in, None)
            pbar = tqdm(f_in, desc="link", unit="news", total=total_lines, initial=skip_lines)
            for idx, line in enumerate(pbar, start=skip_lines):
                if args.limit is not None and (idx - skip_lines) >= args.limit:
                    break
                rec = json.loads(line)
                mentions = rec.get("mentions") or []
                linked = []
                for m in mentions:
                    surface = str(m.get("surface") or "").strip()
                    if not surface:
                        continue
                    ner_type = str(m.get("ner_type") or "")
                    start = int(m.get("start") or 0)
                    token_count = len(surface.split())
                    is_all_caps = False
                    letters = [ch for ch in surface if ch.isalpha()]
                    if len(letters) >= 2 and all(ch.isupper() for ch in letters):
                        is_all_caps = True
                    min_match = float(args.min_match_heur if ner_type == "HEUR" else args.min_match)
                    try:
                        cand = searcher.best_candidate(surface, lang=args.lang, limit_n=args.limit_n)
                        consecutive_errors = 0
                    except Exception:
                        consecutive_errors += 1
                        if consecutive_errors >= args.max_consecutive_errors:
                            raise RuntimeError(
                                f"Too many consecutive Wikidata request errors ({consecutive_errors}). "
                                "Check network/proxy and resume later with --resume."
                            )
                        continue
                    if cand is None or cand.match_score < min_match:
                        continue
                    if (
                        ner_type == "HEUR"
                        and args.heur_reject_lowercase_label_at_title_start
                        and start == 0
                        and token_count == 1
                        and not is_all_caps
                        and "-" not in surface
                        and not any(ch.isdigit() for ch in surface)
                    ):
                        label = (cand.label or "").strip()
                        if label and label[0].islower():
                            continue
                    ner_score = float(m.get("ner_score") or 0.0)
                    confidence = ner_score * cand.match_score
                    linked.append(
                        {
                            "start": int(m["start"]),
                            "end": int(m["end"]),
                            "surface": surface,
                            "ner_type": ner_type,
                            "type": ner_type_to_mind_type(ner_type),
                            "ner_score": ner_score,
                            "wikidata_id": cand.qid,
                            "wikidata_label": cand.label,
                            "match_score": cand.match_score,
                            "confidence": confidence,
                        }
                    )
                out = {"news_id": rec["news_id"], "title": rec.get("title") or "", "linked_mentions": linked}
                f_out.write(json.dumps(out, ensure_ascii=False))
                f_out.write("\n")
                f_out.flush()
    finally:
        searcher.close()


if __name__ == "__main__":
    main()

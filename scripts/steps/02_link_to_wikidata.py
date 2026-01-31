#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm
import concurrent.futures
import threading

import sys

def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur.parent] + list(cur.parents):
        if (parent / "src").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"Could not find repo root from {cur}")


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.wikidata import WikidataSearcher, ner_type_to_mind_type


DEFAULT_CACHE_DB = Path("outputs/cache/wikidata_search.sqlite")
LEGACY_CACHE_DB = Path("cache/wikidata_search.sqlite")
if LEGACY_CACHE_DB.exists() and not DEFAULT_CACHE_DB.exists():
    DEFAULT_CACHE_DB = LEGACY_CACHE_DB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Link NER mentions to Wikidata QIDs via Wikidata search API.")
    p.add_argument("--mentions_jsonl", type=Path, required=True, help="Input JSONL from 01_ner_titles_nbbert.py")
    p.add_argument("--output_jsonl", type=Path, required=True, help="Output JSONL with linked mentions.")
    p.add_argument("--cache_db", type=Path, default=DEFAULT_CACHE_DB)
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
    p.add_argument("--workers", type=int, default=2, help="Number of concurrent worker threads.")
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


def process_line(line: str, searcher: WikidataSearcher, args: argparse.Namespace) -> str | None:
    rec = json.loads(line)
    mentions = rec.get("mentions") or []
    linked = []
    consecutive_errors = 0
    
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
        
        cand = None
        try:
            cand = searcher.best_candidate(surface, lang=args.lang, limit_n=args.limit_n)
            consecutive_errors = 0
        except Exception:
            # In thread pool, we just log/skip or return None, but here we continue
            # If we want to abort global processing, it is harder with map. 
            # We will catch and ignore for now or raise to stop.
            # Reraising will likely stop the executor.
            raise 

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
    return json.dumps(out, ensure_ascii=False)


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
        
        with args.mentions_jsonl.open("r", encoding="utf-8") as f_in, \
             args.output_jsonl.open(output_mode, encoding="utf-8", buffering=1) as f_out, \
             concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:

            if skip_lines:
                for _ in range(skip_lines):
                    next(f_in, None)
            
            # Helper to map inputs
            def _submitter():
                for line in f_in:
                    yield line

            # Use map to preserve order (though writing order strictly matters less for jsonl, usually input order is nice)
            # Process in chunks to avoid queuing all lines in memory
            # But executor.map is lazy.
            
            pbar = tqdm(total=total_lines, initial=skip_lines, desc="link", unit="news")
            
            # Lambda adapter to pass args
            fn = lambda l: process_line(l, searcher, args)
            
            for result_str in executor.map(fn, _submitter()):
                if result_str:
                    f_out.write(result_str + "\n")
                pbar.update(1)
            
            pbar.close()

    finally:
        searcher.close()


if __name__ == "__main__":
    main()

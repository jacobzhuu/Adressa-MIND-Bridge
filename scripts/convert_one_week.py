import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# 颜色定义
# ═══════════════════════════════════════════════════════════════════════════════
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
MAGENTA = '\033[0;35m'
BOLD = '\033[1m'
NC = '\033[0m'  # No Color


def log_header(msg):
    print()
    print(f"{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BOLD}{BLUE}║{NC}  {BOLD}{msg}{NC}")
    print(f"{BOLD}{BLUE}╚══════════════════════════════════════════════════════════════════════╝{NC}")


def log_step(msg):
    print(f"{CYAN}▶{NC} {BOLD}{msg}{NC}")


def log_info(msg):
    print(f"  {YELLOW}ℹ{NC} {msg}")


def log_success(msg):
    print(f"  {GREEN}✓{NC} {msg}")


def log_config(msg):
    print(f"  {MAGENTA}•{NC} {msg}")


def log_result(label, value, indent=True):
    prefix = "  │ " if indent else ""
    print(f"{prefix}{label}: {CYAN}{value}{NC}")


DEFAULT_CATEGORY = "news"
DEFAULT_SUBCATEGORY = "news"
DEFAULT_OUTPUT_DIR = "data/base/adressa_one_week_mind_base"
if os.path.isdir("adressa_one_week_mind_base") and not os.path.isdir(DEFAULT_OUTPUT_DIR):
    DEFAULT_OUTPUT_DIR = "adressa_one_week_mind_base"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert one_week to MINDsmall-like format")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing one_week files")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for MIND-style splits",
    )
    parser.add_argument("--neg-ratio", type=int, default=4, help="Number of negatives per positive impression")
    parser.add_argument(
        "--split",
        type=str,
        default="5,1,1",
        help="Train/val/test day counts, e.g. 5,1,1. Extra days go to train.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--timezone",
        choices=["utc", "local"],
        default="utc",
        help="Timestamp conversion timezone",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=0,
        help="Max history length (0 means no limit)",
    )
    parser.add_argument(
        "--news-scope",
        choices=["split", "all"],
        default="split",
        help="Write news.tsv per split or include all news in each split",
    )
    parser.add_argument(
        "--carry-history",
        action="store_true",
        help="Carry user history and seen news across splits",
    )
    parser.add_argument(
        "--within-day-history",
        choices=["rolling", "frozen"],
        default="rolling",
        help=(
            "How to build History within the same day file: "
            "'rolling' updates after each click (default); "
            "'frozen' keeps History fixed to the beginning of that day."
        ),
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=0,
        help="Number of initial days to use solely for history (no behaviors generated)",
    )
    return parser.parse_args()


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def sanitize_text(value):
    if value is None:
        return ""
    text = str(value)
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def format_time(ts, use_utc=True):
    try:
        ts_val = float(ts)
    except (TypeError, ValueError):
        return ""
    if ts_val > 1e12:
        ts_val /= 1000.0
    if use_utc:
        dt = datetime.utcfromtimestamp(ts_val)
    else:
        dt = datetime.fromtimestamp(ts_val)
    return dt.strftime("%m/%d/%Y %I:%M:%S %p")


def extract_profile_items(profile, group_name):
    items = []
    if not isinstance(profile, list):
        return items
    for item in profile:
        for group in item.get("groups", []):
            if group.get("group") == group_name:
                val = item.get("item")
                if val:
                    items.append(val)
                break
    return items


def extract_category(profile):
    taxonomy_items = extract_profile_items(profile, "taxonomy")
    if taxonomy_items:
        chosen = max(taxonomy_items, key=lambda x: (x.count("/"), len(x)))
        parts = [p.strip() for p in chosen.split("/") if p.strip()]
        if parts:
            category = parts[0].lower()
            subcategory = parts[1].lower() if len(parts) > 1 else category
            return category, subcategory

    category_items = extract_profile_items(profile, "category")
    if category_items:
        category = category_items[0].strip().lower()
        subcategory = category_items[1].strip().lower() if len(category_items) > 1 else category
        return category, subcategory

    return DEFAULT_CATEGORY, DEFAULT_SUBCATEGORY


def is_article_event(event):
    if not event.get("id") or not event.get("title") or not event.get("url"):
        return False
    profile = event.get("profile")
    if not profile:
        return False
    page_classes = extract_profile_items(profile, "pageclass")
    if page_classes and "article" not in page_classes:
        return False
    return True


def parse_split_spec(spec, total_files):
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--split must have 3 comma-separated integers, e.g. 5,1,1")
    counts = [int(p) for p in parts]
    if sum(counts) > total_files:
        raise ValueError("--split sum exceeds available files")
    if sum(counts) < total_files:
        counts[0] += total_files - sum(counts)
    return counts


def collect_news_metadata(files):
    key_to_newsid = {}
    news_metadata = {}
    news_order = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not is_article_event(event):
                    continue

                key = event.get("id")
                if not key:
                    continue

                if key not in key_to_newsid:
                    news_id = f"N{len(news_order) + 1}"
                    key_to_newsid[key] = news_id
                    news_order.append(news_id)

                    category, subcategory = extract_category(event.get("profile", []))
                    title = sanitize_text(event.get("title"))
                    url = sanitize_text(event.get("canonicalUrl") or event.get("url"))

                    news_metadata[news_id] = {
                        "category": category or DEFAULT_CATEGORY,
                        "subcategory": subcategory or DEFAULT_SUBCATEGORY,
                        "title": title,
                        "abstract": "",
                        "url": url,
                    }
                else:
                    news_id = key_to_newsid[key]
                    meta = news_metadata[news_id]

                    if not meta.get("title") and event.get("title"):
                        meta["title"] = sanitize_text(event.get("title"))

                    profile = event.get("profile", [])
                    category, subcategory = extract_category(profile)
                    if meta.get("category") in ("", DEFAULT_CATEGORY) and category != DEFAULT_CATEGORY:
                        meta["category"] = category
                    if meta.get("subcategory") in ("", DEFAULT_SUBCATEGORY) and subcategory != DEFAULT_SUBCATEGORY:
                        meta["subcategory"] = subcategory

                    if not meta.get("url") and (event.get("canonicalUrl") or event.get("url")):
                        meta["url"] = sanitize_text(event.get("canonicalUrl") or event.get("url"))

    return key_to_newsid, news_metadata, news_order


def load_daily_events(file_path, key_to_newsid):
    events = []
    daily_news_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_article_event(event):
                continue

            uid = event.get("userId")
            ts = event.get("time")
            if uid is None or ts is None:
                continue

            try:
                ts_val = float(ts)
            except (TypeError, ValueError):
                continue

            key = event.get("id")
            news_id = key_to_newsid.get(key)
            if not news_id:
                continue

            events.append((ts_val, uid, news_id))
            daily_news_set.add(news_id)

    events.sort(key=lambda x: x[0])
    return events, list(daily_news_set)


def sample_negatives(positive_id, pools, k):
    negatives = []
    used = {positive_id}

    for pool in pools:
        if len(negatives) >= k:
            break
        if not pool:
            continue
        attempts = 0
        max_attempts = k * 20
        while len(negatives) < k and attempts < max_attempts:
            cand = random.choice(pool)
            attempts += 1
            if cand in used:
                continue
            negatives.append(cand)
            used.add(cand)

    if len(negatives) < k:
        for pool in pools:
            for cand in pool:
                if cand in used:
                    continue
                negatives.append(cand)
                used.add(cand)
                if len(negatives) == k:
                    break
            if len(negatives) == k:
                break

    return negatives


def write_news_tsv(path, news_ids, news_metadata):
    with open(path, "w", encoding="utf-8") as f:
        for news_id in news_ids:
            meta = news_metadata[news_id]
            row = [
                news_id,
                meta.get("category", DEFAULT_CATEGORY),
                meta.get("subcategory", DEFAULT_SUBCATEGORY),
                sanitize_text(meta.get("title")),
                sanitize_text(meta.get("abstract")),
                sanitize_text(meta.get("url")),
                "[]",
                "[]",
            ]
            f.write("\t".join(row) + "\n")


def process_data(
    input_dir,
    output_dir,
    split_spec,
    neg_ratio,
    seed,
    use_utc,
    max_history,
    news_scope,
    carry_history,
    within_day_history,
    history_days,
):
    random.seed(seed)

    files = sorted(glob.glob(os.path.join(input_dir, "2017*")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No input files found in {input_dir}")

    if history_days >= len(files):
        raise ValueError(f"--history-days {history_days} cannot be >= total files {len(files)}")

    history_files = files[:history_days]
    split_files = files[history_days:]

    train_count, val_count, test_count = parse_split_spec(split_spec, len(split_files))
    train_files = split_files[:train_count]
    val_files = split_files[train_count : train_count + val_count]
    test_files = split_files[train_count + val_count : train_count + val_count + test_count]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    # ═══════════════════════════════════════════════════════════════════════════════
    # Configuration Display
    # ═══════════════════════════════════════════════════════════════════════════════
    log_header("CONVERT: Adressa → MIND Format")
    print(f"  {YELLOW}Paths:{NC}")
    log_config(f"input_dir:     {input_dir}")
    log_config(f"output_dir:    {output_dir}")
    print()
    print(f"  {YELLOW}Split Configuration:{NC}")
    log_config(f"history_days:  {history_days} files (history-only)")
    log_config(f"train:         {train_count} files")
    log_config(f"val:           {val_count} files")
    log_config(f"test:          {test_count} files")
    print()
    print(f"  {YELLOW}Behavior Settings:{NC}")
    log_config(f"neg_ratio:     {neg_ratio}")
    log_config(f"max_history:   {max_history if max_history else 'unlimited'}")
    log_config(f"carry_history: {carry_history}")
    log_config(f"within_day:    {within_day_history}")
    log_config(f"news_scope:    {news_scope}")

    ensure_dir(output_dir)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Pass 1: Extract News Metadata
    # ═══════════════════════════════════════════════════════════════════════════════
    log_header("Pass 1: Extracting News Metadata")
    key_to_newsid, news_metadata, news_order = collect_news_metadata(files)
    all_news_ids = list(news_order)
    log_success(f"Total unique news: {CYAN}{len(all_news_ids)}{NC}")

    news_in_split = {name: set() for name in splits}
    shared_history = defaultdict(list)
    shared_seen_set = set()
    shared_seen_list = []

    # Pre-process history days
    if history_files:
        log_step(f"Pre-processing {len(history_files)} history-only files...")
        for file_path in history_files:
            daily_events, _ = load_daily_events(file_path, key_to_newsid)
            if not daily_events:
                continue
            for _, uid, news_id in daily_events:
                shared_history[uid].append(news_id)
                if news_id not in shared_seen_set:
                    shared_seen_set.add(news_id)
                    shared_seen_list.append(news_id)
        log_success(f"History users: {CYAN}{len(shared_history)}{NC}, seen news: {CYAN}{len(shared_seen_list)}{NC}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # Pass 2: Generate Behaviors
    # ═══════════════════════════════════════════════════════════════════════════════
    log_header("Pass 2: Generating Behaviors")

    for split_name, split_files in splits.items():
        if not split_files:
            continue

        split_dir = os.path.join(output_dir, split_name)
        ensure_dir(split_dir)
        behaviors_path = os.path.join(split_dir, "behaviors.tsv")

        if carry_history:
            user_history = shared_history
            seen_news_set = shared_seen_set
            seen_news_list = shared_seen_list
        else:
            user_history = defaultdict(list, {k: v[:] for k, v in shared_history.items()})
            seen_news_set = set(shared_seen_set)
            seen_news_list = list(shared_seen_list)

        imp_id = 0
        total_events = 0
        written = 0

        log_step(f"Processing {BOLD}{split_name}{NC} split...")
        with open(behaviors_path, "w", encoding="utf-8") as f_out:
            for file_path in split_files:
                daily_events, daily_pool = load_daily_events(file_path, key_to_newsid)
                if not daily_events:
                    continue
                daily_pool_list = list(daily_pool)
                frozen_history_snapshot = {}

                for ts, uid, news_id in daily_events:
                    total_events += 1
                    news_in_split[split_name].add(news_id)

                    if within_day_history == "frozen":
                        history = frozen_history_snapshot.get(uid)
                        if history is None:
                            history = list(user_history[uid])
                            frozen_history_snapshot[uid] = history
                    else:
                        history = user_history[uid]

                    if max_history and max_history > 0:
                        history_slice = history[-max_history:]
                    else:
                        history_slice = history

                    hist_str = " ".join(history_slice)
                    negatives = sample_negatives(news_id, [seen_news_list, daily_pool_list, all_news_ids], neg_ratio)

                    if len(negatives) == neg_ratio:
                        impressions = [f"{news_id}-1"] + [f"{n}-0" for n in negatives]
                        random.shuffle(impressions)
                        impressions_str = " ".join(impressions)
                        time_str = format_time(ts, use_utc=use_utc)
                        if time_str:
                            imp_id += 1
                            uid_clean = sanitize_text(uid)
                            f_out.write(
                                f"{imp_id}\t{uid_clean}\t{time_str}\t{hist_str}\t{impressions_str}\n"
                            )
                            written += 1

                    user_history[uid].append(news_id)
                    if news_id not in seen_news_set:
                        seen_news_set.add(news_id)
                        seen_news_list.append(news_id)

        print(f"  {GREEN}✓{NC} {BOLD}{split_name}{NC} Results:")
        print(f"  │ Events:      {CYAN}{total_events:>8}{NC}")
        print(f"  │ Behaviors:   {CYAN}{written:>8}{NC}")
        print(f"  │ Unique News: {CYAN}{len(news_in_split[split_name]):>8}{NC}")
        print()

    # ═══════════════════════════════════════════════════════════════════════════════
    # Write news.tsv
    # ═══════════════════════════════════════════════════════════════════════════════
    log_header("Writing news.tsv Files")
    for split_name, split_files in splits.items():
        if not split_files:
            continue
        split_dir = os.path.join(output_dir, split_name)
        if news_scope == "all":
            news_ids = all_news_ids
        else:
            news_ids = [nid for nid in all_news_ids if nid in news_in_split[split_name]]
        write_news_tsv(os.path.join(split_dir, "news.tsv"), news_ids, news_metadata)
        log_success(f"{split_name}/news.tsv: {CYAN}{len(news_ids)}{NC} articles")

    # ═══════════════════════════════════════════════════════════════════════════════
    # Complete
    # ═══════════════════════════════════════════════════════════════════════════════
    print()
    print(f"{GREEN}{BOLD}════════════════════════════════════════════════════════════════════════{NC}")
    print(f"{GREEN}{BOLD}  ✓ CONVERSION COMPLETE{NC}")
    print(f"{GREEN}{BOLD}════════════════════════════════════════════════════════════════════════{NC}")
    print()


if __name__ == "__main__":
    args = parse_args()
    process_data(
        input_dir=args.input,
        output_dir=args.output,
        split_spec=args.split,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        use_utc=(args.timezone == "utc"),
        max_history=args.max_history,
        news_scope=args.news_scope,
        carry_history=args.carry_history,
        within_day_history=args.within_day_history,
        history_days=args.history_days,
    )

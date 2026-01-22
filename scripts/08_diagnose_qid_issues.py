#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.news_tsv import iter_news_tsv


QID_CANON_RE = re.compile(r"^Q[1-9]\d*$")
QID_LOWER_RE = re.compile(r"^q[1-9]\d*$")
QID_ANYWHERE_RE = re.compile(r"(Q[1-9]\d*)", re.IGNORECASE)
URL_QID_RE = re.compile(r"^https?://www\.wikidata\.org/entity/(Q[1-9]\d*)$", re.IGNORECASE)
WD_PREFIX_RE = re.compile(r"^wd:(Q[1-9]\d*)$", re.IGNORECASE)
WIKIDATA_WIKI_RE = re.compile(r"^https?://www\.wikidata\.org/wiki/(Q[1-9]\d*)$", re.IGNORECASE)
NUM_RE = re.compile(r"^[1-9]\d*$")


def _default_data_root() -> Path:
    preferred = Path("data/work/adressa_one_week_mind_final")
    legacy = Path("adressa_one_week_mind_final")
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def _default_artifacts_dir() -> Path:
    preferred = Path("outputs/artifacts")
    legacy = Path("artifacts")
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose low Wikidata QID coverage vs MIND entity embeddings (format issues, id type issues, EL quality)."
    )
    p.add_argument(
        "--news_tsv",
        type=Path,
        nargs="*",
        default=[],
        help="One or more news.tsv paths to inspect (if omitted, uses --data_root/{train,val,test}/news.tsv when present).",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=_default_data_root(),
        help="Dataset root containing {train,val,test}/news.tsv (used when --news_tsv is omitted).",
    )
    p.add_argument(
        "--mind_entity_vec",
        type=Path,
        nargs="*",
        default=[],
        help="One or more MIND entity_embedding.vec paths to compute overlap (optional).",
    )
    p.add_argument(
        "--mentions_jsonl",
        type=Path,
        nargs="*",
        default=[],
        help="One or more *.mentions.jsonl paths (optional; used for EL quality).",
    )
    p.add_argument(
        "--linked_jsonl",
        type=Path,
        nargs="*",
        default=[],
        help="One or more *.linked.jsonl paths (optional; used for EL quality).",
    )
    p.add_argument(
        "--artifacts_dir",
        type=Path,
        default=_default_artifacts_dir(),
        help="Artifacts dir used for auto-discovery of *.mentions.jsonl / *.linked.jsonl when args not provided.",
    )
    p.add_argument("--max_examples", type=int, default=20, help="Max examples to print per suspicious category.")
    p.add_argument("--output_json", type=Path, default=None, help="Write a machine-readable report JSON.")
    return p.parse_args()


def _escape(s: str) -> str:
    return s.encode("unicode_escape", errors="backslashreplace").decode("ascii")


def _has_invisible_or_weird_space(s: str) -> bool:
    for ch in s:
        if ch in {"\t", "\n", "\r"}:
            return True
        if ch == " ":
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("Z"):
            return True
        if ch in {"\u200b", "\ufeff"}:
            return True
    return False


def normalize_to_qid(value: Any) -> str | None:
    """
    Best-effort normalization to canonical 'Q123' (uppercase Q, digits).
    Returns None if no plausible QID is found.
    """
    if value is None:
        return None

    if isinstance(value, int):
        if value <= 0:
            return None
        return f"Q{value}"

    s = str(value)
    if not s:
        return None
    s = s.strip()
    if not s:
        return None

    m = WD_PREFIX_RE.match(s)
    if m:
        return m.group(1).upper()
    m = URL_QID_RE.match(s)
    if m:
        return m.group(1).upper()
    m = WIKIDATA_WIKI_RE.match(s)
    if m:
        return m.group(1).upper()

    if QID_CANON_RE.match(s):
        return s
    if QID_LOWER_RE.match(s):
        return "Q" + s[1:]
    if NUM_RE.match(s):
        return "Q" + s

    m = QID_ANYWHERE_RE.search(s)
    if m:
        return m.group(1).upper()
    return None


def classify_wikidata_id(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, int):
        return "int_number"
    s = str(value)
    if s == "":
        return "empty"
    if QID_CANON_RE.match(s):
        return "qid_canonical"
    if QID_LOWER_RE.match(s):
        return "qid_lowercase"
    if WD_PREFIX_RE.match(s):
        return "qid_wd_prefix"
    if URL_QID_RE.match(s) or WIKIDATA_WIKI_RE.match(s):
        return "qid_url"
    if NUM_RE.match(s):
        return "digits_only"
    if s.strip() != s:
        return "has_outer_whitespace"
    if _has_invisible_or_weird_space(s):
        return "has_invisible_space"
    if " " in s:
        return "has_internal_space"
    return "other"


@dataclass(frozen=True)
class TitleEntityObs:
    news_id: str
    title: str
    raw_value: Any

    @property
    def raw_str(self) -> str:
        return str(self.raw_value)

    @property
    def raw_type(self) -> str:
        return type(self.raw_value).__name__


def iter_title_entity_ids(news_tsv_paths: Iterable[Path]) -> Iterable[TitleEntityObs]:
    for path in news_tsv_paths:
        for row in iter_news_tsv(path):
            te = (row.title_entities or "").strip()
            if not te or te == "[]":
                continue
            try:
                ents = json.loads(te)
            except Exception:
                continue
            if not isinstance(ents, list):
                continue
            for e in ents:
                if not isinstance(e, dict):
                    continue
                yield TitleEntityObs(news_id=row.news_id, title=row.title, raw_value=e.get("WikidataId"))


def iter_vec_ids(vec_path: Path) -> Iterable[str]:
    with vec_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # vecs are TSV: "<id>\t<dim0>\t<dim1>..."
            # be tolerant: split on any whitespace.
            yield line.split()[0]


def _pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0


def _fmt_pct(n: int, d: int) -> str:
    return f"{_pct(n, d):.2f}%"


def _safe_mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _safe_median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = int(math.floor(q * (len(xs_sorted) - 1)))
    return xs_sorted[max(0, min(idx, len(xs_sorted) - 1))]


def load_linked_by_news(linked_path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with linked_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            out[str(rec.get("news_id") or "")].extend(rec.get("linked_mentions") or [])
    return out


def analyze_el_quality(mentions_paths: list[Path], linked_paths: list[Path], *, max_examples: int) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}}

    # index by filename stem prefix: "train"/"val"/"test"
    linked_map: dict[str, Path] = {}
    for p in linked_paths:
        stem = p.name.split(".", 1)[0]
        linked_map[stem] = p

    for mp in mentions_paths:
        split = mp.name.split(".", 1)[0]
        lp = linked_map.get(split)
        if lp is None:
            continue

        linked_by_news = load_linked_by_news(lp)

        total_news = 0
        total_mentions = 0
        total_linked = 0
        news_with_mentions = 0
        news_with_links = 0
        news_with_mentions_but_no_links = 0

        title_lens_all: list[int] = []
        title_lens_with_links: list[int] = []
        title_lens_mentions_no_links: list[int] = []

        match_scores: list[float] = []
        confidences: list[float] = []
        linked_surface_lens: list[int] = []

        low_match_examples: list[dict[str, Any]] = []

        with mp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                news_id = str(rec.get("news_id") or "")
                title = str(rec.get("title") or "")
                mentions = rec.get("mentions") or []
                if not isinstance(mentions, list):
                    mentions = []

                total_news += 1
                title_lens_all.append(len(title))

                mention_n = len(mentions)
                total_mentions += mention_n
                if mention_n > 0:
                    news_with_mentions += 1

                linked = linked_by_news.get(news_id, [])
                linked_n = len(linked)
                total_linked += linked_n
                if linked_n > 0:
                    news_with_links += 1
                    title_lens_with_links.append(len(title))
                if mention_n > 0 and linked_n == 0:
                    news_with_mentions_but_no_links += 1
                    title_lens_mentions_no_links.append(len(title))

                for lm in linked:
                    ms = float(lm.get("match_score") or 0.0)
                    cf = float(lm.get("confidence") or 0.0)
                    match_scores.append(ms)
                    confidences.append(cf)
                    surface = str(lm.get("surface") or "")
                    linked_surface_lens.append(len(surface))

                    if ms < 0.65 and len(low_match_examples) < max_examples:
                        low_match_examples.append(
                            {
                                "news_id": news_id,
                                "title": title,
                                "surface": surface,
                                "wikidata_id": lm.get("wikidata_id"),
                                "wikidata_label": lm.get("wikidata_label"),
                                "match_score": ms,
                                "confidence": cf,
                            }
                        )

        report["splits"][split] = {
            "mentions_jsonl": str(mp),
            "linked_jsonl": str(lp),
            "total_news": total_news,
            "total_mentions": total_mentions,
            "total_linked_mentions": total_linked,
            "link_rate_per_mention": (total_linked / total_mentions) if total_mentions else 0.0,
            "news_with_mentions": news_with_mentions,
            "news_with_links": news_with_links,
            "news_with_mentions_but_no_links": news_with_mentions_but_no_links,
            "title_len": {
                "mean_all": _safe_mean([float(x) for x in title_lens_all]),
                "mean_with_links": _safe_mean([float(x) for x in title_lens_with_links]),
                "mean_mentions_no_links": _safe_mean([float(x) for x in title_lens_mentions_no_links]),
            },
            "linked": {
                "match_score": {
                    "min": min(match_scores) if match_scores else 0.0,
                    "p10": _quantile(match_scores, 0.10),
                    "median": _safe_median(match_scores),
                    "p90": _quantile(match_scores, 0.90),
                    "max": max(match_scores) if match_scores else 0.0,
                },
                "confidence": {
                    "min": min(confidences) if confidences else 0.0,
                    "median": _safe_median(confidences),
                    "max": max(confidences) if confidences else 0.0,
                },
                "surface_len": {
                    "median": _safe_median([float(x) for x in linked_surface_lens]),
                },
                "low_match_examples": low_match_examples,
            },
        }

    return report


def main() -> None:
    args = parse_args()

    news_paths = list(args.news_tsv)
    if not news_paths:
        for sp in ["train", "val", "test"]:
            p = args.data_root / sp / "news.tsv"
            if p.exists():
                news_paths.append(p)
    if not news_paths:
        raise SystemExit("No news.tsv provided and none found under --data_root.")

    mentions_paths = list(args.mentions_jsonl)
    linked_paths = list(args.linked_jsonl)
    if not mentions_paths:
        for sp in ["train", "val", "test"]:
            p = args.artifacts_dir / f"{sp}.mentions.jsonl"
            if p.exists():
                mentions_paths.append(p)
    if not linked_paths:
        for sp in ["train", "val", "test"]:
            p = args.artifacts_dir / f"{sp}.linked.jsonl"
            if p.exists():
                linked_paths.append(p)

    # 1) Scan title_entities WikidataId fields.
    obs_total = 0
    raw_counts: Counter[str] = Counter()
    raw_type_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    examples_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    raw_set: set[str] = set()
    norm_set: set[str] = set()
    norm_map_examples: dict[str, set[str]] = defaultdict(set)  # raw -> {norm}

    for obs in iter_title_entity_ids(news_paths):
        obs_total += 1
        raw = obs.raw_value
        raw_type_counts[type(raw).__name__] += 1
        raw_s = str(raw)
        raw_counts[raw_s] += 1
        raw_set.add(raw_s)

        cls = classify_wikidata_id(raw)
        class_counts[cls] += 1
        if cls != "qid_canonical" and len(examples_by_class[cls]) < args.max_examples:
            examples_by_class[cls].append(
                {
                    "news_id": obs.news_id,
                    "title": obs.title,
                    "raw_type": obs.raw_type,
                    "raw_value": raw,
                    "raw_escaped": _escape(raw_s),
                }
            )

        norm = normalize_to_qid(raw)
        if norm is not None:
            norm_set.add(norm)
            if norm != raw_s and len(norm_map_examples[raw_s]) < 3:
                norm_map_examples[raw_s].add(norm)

    # 2) Load MIND vec ids for overlap, if provided.
    mind_ids: set[str] = set()
    if args.mind_entity_vec:
        for vp in args.mind_entity_vec:
            for eid in iter_vec_ids(vp):
                if eid:
                    mind_ids.add(eid)

    raw_overlap = len(raw_set & mind_ids) if mind_ids else 0
    norm_overlap = len(norm_set & mind_ids) if mind_ids else 0
    norm_gain = norm_overlap - raw_overlap

    # 3) EL quality report (optional).
    el_report: dict[str, Any] = {}
    if mentions_paths and linked_paths:
        el_report = analyze_el_quality(mentions_paths, linked_paths, max_examples=args.max_examples)

    report = {
        "inputs": {
            "news_tsv": [str(p) for p in news_paths],
            "mind_entity_vec": [str(p) for p in args.mind_entity_vec],
            "mentions_jsonl": [str(p) for p in mentions_paths],
            "linked_jsonl": [str(p) for p in linked_paths],
        },
        "qid_string_consistency": {
            "total_title_entities": obs_total,
            "unique_raw_ids": len(raw_set),
            "unique_normalized_qids": len(norm_set),
            "raw_type_counts": dict(raw_type_counts),
            "class_counts": dict(class_counts),
            "examples_by_class": examples_by_class,
            "normalization_examples": {k: sorted(v) for k, v in norm_map_examples.items()},
        },
        "mind_overlap": None,
        "el_quality": el_report or None,
    }

    if mind_ids:
        report["mind_overlap"] = {
            "mind_unique_ids": len(mind_ids),
            "raw_unique_ids": len(raw_set),
            "normalized_unique_qids": len(norm_set),
            "raw_overlap": raw_overlap,
            "raw_overlap_pct": _pct(raw_overlap, len(raw_set)),
            "normalized_overlap": norm_overlap,
            "normalized_overlap_pct": _pct(norm_overlap, len(norm_set)),
            "normalization_gain": norm_gain,
        }

    # Pretty print summary.
    print("== QID string checks (from title_entities) ==")
    print(f"news.tsv files: {len(news_paths)}")
    print(f"total title_entities entries: {obs_total}")
    print(f"unique raw ids: {len(raw_set)}")
    print(f"unique normalized qids: {len(norm_set)}")
    print("")

    print("Category counts (WikidataId):")
    for k, v in class_counts.most_common():
        print(f"  - {k}: {v} ({_fmt_pct(v, obs_total)})")
    print("")

    suspicious = [k for k in class_counts.keys() if k != "qid_canonical"]
    if suspicious:
        print("Examples of suspicious/non-canonical WikidataId values:")
        for k in sorted(suspicious):
            ex = examples_by_class.get(k) or []
            if not ex:
                continue
            print(f"  - {k}:")
            for item in ex[: args.max_examples]:
                print(
                    f"      news_id={item['news_id']} raw_type={item['raw_type']} raw={item['raw_escaped']} title={item['title']}"
                )
        print("")
    else:
        print("No suspicious WikidataId values found (all are canonical QIDs).")
        print("")

    if mind_ids:
        print("== MIND overlap checks ==")
        print(f"MIND unique ids loaded: {len(mind_ids)}")
        print(f"raw overlap: {raw_overlap} / {len(raw_set)} ({_fmt_pct(raw_overlap, len(raw_set))})")
        print(f"normalized overlap: {norm_overlap} / {len(norm_set)} ({_fmt_pct(norm_overlap, len(norm_set))})")
        print(f"normalization gain: {norm_gain}")
        print("")
        if norm_gain > 0:
            # show a few raw ids that only match after normalization
            raw_only = (raw_set - mind_ids)
            recovered = []
            for r in raw_only:
                n = normalize_to_qid(r)
                if n and n in mind_ids:
                    recovered.append((r, n))
                    if len(recovered) >= args.max_examples:
                        break
            if recovered:
                print("Examples that match MIND only after normalization:")
                for r, n in recovered:
                    print(f"  - raw={_escape(r)} -> norm={n}")
                print("")
    else:
        print("== MIND overlap checks ==")
        print("Skipped (no --mind_entity_vec provided).")
        print("")

    if el_report:
        print("== EL quality checks (mentions vs linked) ==")
        for split, srep in (el_report.get("splits") or {}).items():
            tm = int(srep.get("total_mentions") or 0)
            tl = int(srep.get("total_linked_mentions") or 0)
            tr = float(srep.get("link_rate_per_mention") or 0.0)
            nnews = int(srep.get("total_news") or 0)
            nml = int(srep.get("news_with_mentions_but_no_links") or 0)
            nwm = int(srep.get("news_with_mentions") or 0)
            print(f"- {split}:")
            print(f"    news: {nnews}")
            print(f"    mentions: {tm}, linked: {tl}, link_rate: {tr*100:.2f}%")
            print(f"    news_with_mentions: {nwm}, mentions_but_no_links: {nml} ({_fmt_pct(nml, nnews)})")
            ms = (srep.get("linked") or {}).get("match_score") or {}
            print(
                "    match_score: "
                f"min={ms.get('min', 0.0):.3f} p10={ms.get('p10', 0.0):.3f} "
                f"median={ms.get('median', 0.0):.3f} p90={ms.get('p90', 0.0):.3f} max={ms.get('max', 0.0):.3f}"
            )
        print("")
    else:
        print("== EL quality checks (mentions vs linked) ==")
        print("Skipped (need both *.mentions.jsonl and *.linked.jsonl; pass --mentions_jsonl/--linked_jsonl or use --artifacts_dir).")
        print("")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {args.output_json}")


if __name__ == "__main__":
    main()

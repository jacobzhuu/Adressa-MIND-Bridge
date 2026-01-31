#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur.parent] + list(cur.parents):
        if (parent / "src").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"Could not find repo root from {cur}")


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from adressa_entity.news_tsv import NewsTsvRow, iter_news_tsv, write_news_tsv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write MIND-style title_entities back into Adressa news.tsv.")
    p.add_argument("--news_tsv", type=Path, required=True)
    p.add_argument("--linked_jsonl", type=Path, required=True)
    p.add_argument("--output_tsv", type=Path, required=True)
    return p.parse_args()


def build_title_entities(linked_mentions: list[dict]) -> str:
    by_qid: dict[str, dict] = {}

    for m in linked_mentions:
        qid = str(m.get("wikidata_id") or "")
        if not qid:
            continue
        start = int(m.get("start") or 0)
        surface = str(m.get("surface") or "")
        label = str(m.get("wikidata_label") or surface)
        etype = str(m.get("type") or "")
        conf = float(m.get("confidence") or 0.0)

        entry = by_qid.get(qid)
        if entry is None:
            entry = {
                "Label": label,
                "Type": etype,
                "WikidataId": qid,
                "Confidence": conf,
                "OccurrenceOffsets": [],
                "SurfaceForms": [],
            }
            by_qid[qid] = entry
        entry["OccurrenceOffsets"].append(start)
        entry["SurfaceForms"].append(surface)
        entry["Confidence"] = max(float(entry["Confidence"]), conf)

    entities = list(by_qid.values())
    for e in entities:
        pairs = sorted(zip(e["OccurrenceOffsets"], e["SurfaceForms"]))
        e["OccurrenceOffsets"] = [p[0] for p in pairs]
        e["SurfaceForms"] = [p[1] for p in pairs]

    entities.sort(key=lambda x: (x["OccurrenceOffsets"][0] if x["OccurrenceOffsets"] else 10**9, x["WikidataId"]))
    return json.dumps(entities, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    linked_by_news: dict[str, list[dict]] = defaultdict(list)
    with args.linked_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            linked_by_news[str(rec["news_id"])].extend(rec.get("linked_mentions") or [])

    def iter_rows():
        for row in iter_news_tsv(args.news_tsv):
            linked = linked_by_news.get(row.news_id, [])
            title_entities = build_title_entities(linked) if linked else "[]"
            yield NewsTsvRow(
                news_id=row.news_id,
                category=row.category,
                subcategory=row.subcategory,
                title=row.title,
                abstract=row.abstract,
                url=row.url,
                title_entities=title_entities,
                abstract_entities=row.abstract_entities,
            )

    write_news_tsv(args.output_tsv, iter_rows())


if __name__ == "__main__":
    main()

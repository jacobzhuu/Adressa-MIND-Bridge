#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests import Response
from tqdm import tqdm


DEFAULT_CACHE_DB = Path("outputs/cache/wikidata_entities.sqlite")
LEGACY_CACHE_DB = Path("cache/wikidata_entities.sqlite")
if LEGACY_CACHE_DB.exists() and not DEFAULT_CACHE_DB.exists():
    DEFAULT_CACHE_DB = LEGACY_CACHE_DB


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _read_id_list(path: Path, *, prefix: str) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith(prefix):
            continue
        ids.append(s)
    return ids


def _load_allowed_pids(mind_relation_vec: Path) -> set[str]:
    allowed: set[str] = set()
    with mind_relation_vec.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            pid = parts[0].strip()
            if pid.startswith("P") and pid[1:].isdigit():
                allowed.add(pid)
    return allowed


@dataclass
class _WikidataClient:
    session: requests.Session
    sleep_seconds: float
    max_retries: int
    retry_base_sleep_seconds: float
    retry_max_sleep_seconds: float
    timeout_seconds: float

    def wbgetentities_claims(self, qids: list[str], *, lang: str) -> dict[str, dict[str, Any]]:
        if not qids:
            return {}
        params = {
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "props": "claims",
            "format": "json",
            "uselang": lang,
            "maxlag": "5",
        }
        data = self._request_json("https://www.wikidata.org/w/api.php", params=params)
        entities = data.get("entities") if isinstance(data, dict) else None
        if not isinstance(entities, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for qid in qids:
            e = entities.get(qid)
            if isinstance(e, dict):
                out[qid] = e
            else:
                out[qid] = {"id": qid, "missing": True, "claims": {}}
        return out

    def _request_json(self, url: str, *, params: dict[str, Any]) -> dict[str, Any]:
        resp: Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)
                resp = self.session.get(url, params=params, timeout=self.timeout_seconds)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "error" in data:
                    err = data.get("error") or {}
                    code = str(err.get("code") or "")
                    if code == "maxlag":
                        raise requests.exceptions.HTTPError("Wikidata maxlag", response=resp)
                return data if isinstance(data, dict) else {}
            except (requests.exceptions.RequestException, json.JSONDecodeError):
                if attempt >= self.max_retries:
                    raise
                base = min(self.retry_base_sleep_seconds * (2**attempt), self.retry_max_sleep_seconds)
                # Wikidata maxlag: optionally respect server-provided lag seconds to reduce retry churn.
                try:
                    if resp is not None:
                        payload = resp.json()
                        if isinstance(payload, dict) and "error" in payload:
                            err = payload.get("error") or {}
                            if str(err.get("code") or "") == "maxlag":
                                lag = err.get("lag")
                                if lag is not None:
                                    base = max(base, min(float(lag), self.retry_max_sleep_seconds))
                except Exception:
                    pass
                jitter = random.uniform(0.0, base / 2)
                time.sleep(base + jitter)
                continue
        return {}


class _EntityClaimsCache:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_claims_cache (
              qid TEXT PRIMARY KEY,
              response_json TEXT NOT NULL,
              created_at INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, qid: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT response_json FROM entity_claims_cache WHERE qid=?", (qid,)).fetchone()
        if row is None:
            return None
        try:
            v = json.loads(row[0])
        except Exception:
            return None
        return v if isinstance(v, dict) else None

    def set_many(self, entities: dict[str, dict[str, Any]]) -> None:
        now = int(time.time())
        rows = [(qid, json.dumps(e, ensure_ascii=False), now) for qid, e in entities.items()]
        self._conn.executemany(
            "INSERT OR REPLACE INTO entity_claims_cache(qid, response_json, created_at) VALUES (?, ?, ?)", rows
        )
        self._conn.commit()


class _KgState:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_seeds (
              qid TEXT PRIMARY KEY,
              num_forward_triples INTEGER NOT NULL,
              num_total_triples INTEGER NOT NULL,
              processed_at INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triples (
              head TEXT NOT NULL,
              relation TEXT NOT NULL,
              tail TEXT NOT NULL,
              PRIMARY KEY (head, relation, tail)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_triples_relation ON triples(relation)")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_meta(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM run_meta").fetchall()
        return {str(k): str(v) for k, v in rows}

    def set_meta(self, meta: dict[str, str]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO run_meta(key, value) VALUES (?, ?)",
            [(k, v) for k, v in meta.items()],
        )
        self._conn.commit()

    def processed_set(self) -> set[str]:
        rows = self._conn.execute("SELECT qid FROM processed_seeds").fetchall()
        return {str(r[0]) for r in rows}

    def mark_seed_processed(self, qid: str, *, num_forward: int, num_total: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO processed_seeds(qid, num_forward_triples, num_total_triples, processed_at) "
            "VALUES (?, ?, ?, ?)",
            (qid, int(num_forward), int(num_total), int(time.time())),
        )
        self._conn.commit()

    def insert_triples(self, triples: list[tuple[str, str, str]]) -> int:
        before = self._conn.total_changes
        self._conn.executemany(
            "INSERT OR IGNORE INTO triples(head, relation, tail) VALUES (?, ?, ?)",
            [(h, r, t) for (h, r, t) in triples],
        )
        self._conn.commit()
        return int(self._conn.total_changes - before)

    def iter_triples(self) -> list[tuple[str, str, str]]:
        rows = self._conn.execute("SELECT head, relation, tail FROM triples ORDER BY head, relation, tail").fetchall()
        return [(str(h), str(r), str(t)) for (h, r, t) in rows]

    def count_triples(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()
        return int(row[0]) if row else 0

    def count_seed_with_edges(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM processed_seeds WHERE num_forward_triples > 0").fetchone()
        return int(row[0]) if row else 0

    def relation_counts(self) -> Counter[str]:
        rows = self._conn.execute("SELECT relation, COUNT(*) FROM triples GROUP BY relation").fetchall()
        c: Counter[str] = Counter()
        for rel, n in rows:
            rel_s = str(rel)
            base = rel_s[:-4] if rel_s.endswith("_inv") else rel_s
            c[base] += int(n)
        return c

    def entities_in_triples(self) -> set[str]:
        rows = self._conn.execute("SELECT head, tail FROM triples").fetchall()
        ents: set[str] = set()
        for h, t in rows:
            hs = str(h)
            ts = str(t)
            if hs.startswith("Q") and hs[1:].isdigit():
                ents.add(hs)
            if ts.startswith("Q") and ts[1:].isdigit():
                ents.add(ts)
        return ents


def _extract_forward_triples_from_entity(
    *,
    head_qid: str,
    entity: dict[str, Any] | None,
    allowed_pids: set[str],
    seed_set: set[str],
    keep_neighbors: bool,
    max_triples_per_entity: int,
) -> list[tuple[str, str, str]]:
    if not entity:
        return []
    claims = entity.get("claims") or {}
    if not isinstance(claims, dict):
        return []

    out: list[tuple[str, str, str]] = []
    kept = 0
    for pid in sorted(claims.keys()):
        if pid not in allowed_pids:
            continue
        statements = claims.get(pid) or []
        if not isinstance(statements, list):
            continue
        for st in statements:
            if kept >= max_triples_per_entity:
                return out
            if not isinstance(st, dict):
                continue
            mainsnak = st.get("mainsnak") or {}
            if not isinstance(mainsnak, dict):
                continue
            datavalue = mainsnak.get("datavalue") or {}
            if not isinstance(datavalue, dict):
                continue
            dv_type = str(datavalue.get("type") or "")
            if dv_type != "wikibase-entityid":
                continue
            value = datavalue.get("value") or {}
            if not isinstance(value, dict):
                continue
            tail_qid = str(value.get("id") or "")
            if not (tail_qid.startswith("Q") and tail_qid[1:].isdigit()):
                continue
            if str(value.get("entity-type") or "item") != "item":
                continue
            if tail_qid == head_qid:
                continue
            if (not keep_neighbors) and (tail_qid not in seed_set):
                continue
            out.append((head_qid, pid, tail_qid))
            kept += 1
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Wikidata triples subgraph for TransE training (seed vocab → kg_triples).")
    p.add_argument("--seed_entity_vocab", type=Path, required=True, help="entity_vocab.txt from step 04 (one QID per line).")
    p.add_argument("--mind_relation_vec", type=Path, required=True, help="MIND relation_embedding.vec (PID + 100 dims).")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--cache_db", type=Path, default=DEFAULT_CACHE_DB, help="SQLite cache for wbgetentities entity claims.")
    p.add_argument("--state_db", type=Path, default=None, help="SQLite state DB (triples + resume). Defaults to <output_dir>/kg_state.sqlite")
    p.add_argument("--lang", type=str, default="nb")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep between uncached API calls (seconds).")
    p.add_argument("--trust-env", action=argparse.BooleanOptionalAction, default=True, help="Respect system/env proxy settings.")
    p.add_argument("--max-retries", type=int, default=12, help="Max retries per Wikidata request.")
    p.add_argument("--retry-base-sleep", type=float, default=2.0, help="Base backoff sleep (seconds).")
    p.add_argument("--retry-max-sleep", type=float, default=120.0, help="Max backoff sleep (seconds).")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds).")
    p.add_argument("--batch_size", type=int, default=50, help="How many QIDs per wbgetentities request.")
    p.add_argument(
        "--max_triples_per_entity",
        type=int,
        default=200,
        help="Max forward statements kept per seed entity (anti-explosion).",
    )
    p.add_argument("--keep_neighbors", action="store_true", help="Allow tail ∉ E_seed and include neighbors in kg_entities.txt.")
    p.add_argument("--top_relations_n", type=int, default=20)
    p.add_argument("--resume", action="store_true", help="Resume from existing state DB in output_dir.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_dir state/output files if they exist (dangerous).")
    p.add_argument(
        "--user_agent",
        type=str,
        default="adressa-entity-pipeline/0.1 (https://wikidata.org; contact: local)",
        help="Custom User-Agent for Wikidata API.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_db = args.state_db or (args.output_dir / "kg_state.sqlite")
    kg_triples_path = args.output_dir / "kg_triples.txt"
    kg_stats_path = args.output_dir / "kg_stats.json"
    kg_entities_path = args.output_dir / "kg_entities.txt"

    state_db_existed = state_db.exists()
    if state_db.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"{state_db} already exists. Use --resume or --overwrite.")
    if args.overwrite and state_db.exists():
        state_db.unlink()
        state_db_existed = False
    if args.overwrite:
        for p in (kg_triples_path, kg_stats_path, kg_entities_path):
            if p.exists():
                p.unlink()

    allowed_pids = _load_allowed_pids(args.mind_relation_vec)
    seed_qids = _read_id_list(args.seed_entity_vocab, prefix="Q")
    seed_set = set(seed_qids)

    state = _KgState(state_db)
    try:
        expected_meta = {
            "seed_entity_vocab_sha1": _sha1_file(args.seed_entity_vocab),
            "mind_relation_vec_sha1": _sha1_file(args.mind_relation_vec),
            "max_triples_per_entity": str(int(args.max_triples_per_entity)),
            "keep_neighbors": str(bool(args.keep_neighbors)),
        }
        if args.resume and state_db_existed:
            existing_meta = state.get_meta()
            for k, v in expected_meta.items():
                if existing_meta.get(k) != v:
                    raise ValueError(
                        f"--resume meta mismatch for {k}: expected {v}, got {existing_meta.get(k)}. "
                        "Delete output_dir or run with --overwrite."
                    )
        else:
            state.set_meta(expected_meta)

        processed = state.processed_set() if args.resume else set()
        total = len(seed_qids)
        todo = [qid for qid in seed_qids if qid not in processed]

        cache = _EntityClaimsCache(args.cache_db)
        try:
            session = requests.Session()
            session.trust_env = bool(args.trust_env)
            session.headers.update({"User-Agent": args.user_agent})
            client = _WikidataClient(
                session=session,
                sleep_seconds=float(args.sleep),
                max_retries=int(args.max_retries),
                retry_base_sleep_seconds=float(args.retry_base_sleep),
                retry_max_sleep_seconds=float(args.retry_max_sleep),
                timeout_seconds=float(args.timeout),
            )

            pending_fetch: list[str] = []

            def _flush_pending() -> None:
                nonlocal pending_fetch
                if not pending_fetch:
                    return
                fetched = client.wbgetentities_claims(pending_fetch, lang=args.lang)
                cache.set_many(fetched)
                for qid in pending_fetch:
                    entity = fetched.get(qid) or {}
                    forward = _extract_forward_triples_from_entity(
                        head_qid=qid,
                        entity=entity,
                        allowed_pids=allowed_pids,
                        seed_set=seed_set,
                        keep_neighbors=bool(args.keep_neighbors),
                        max_triples_per_entity=int(args.max_triples_per_entity),
                    )
                    triples = forward + [(t, f"{pid}_inv", h) for (h, pid, t) in forward]
                    state.insert_triples(triples)
                    state.mark_seed_processed(qid, num_forward=len(forward), num_total=len(triples))
                pending_fetch = []

            pbar = tqdm(todo, desc="fetch", unit="ent", total=len(todo))
            for qid in pbar:
                if qid in processed:
                    continue
                cached = cache.get(qid)
                if cached is not None:
                    forward = _extract_forward_triples_from_entity(
                        head_qid=qid,
                        entity=cached,
                        allowed_pids=allowed_pids,
                        seed_set=seed_set,
                        keep_neighbors=bool(args.keep_neighbors),
                        max_triples_per_entity=int(args.max_triples_per_entity),
                    )
                    triples = forward + [(t, f"{pid}_inv", h) for (h, pid, t) in forward]
                    state.insert_triples(triples)
                    state.mark_seed_processed(qid, num_forward=len(forward), num_total=len(triples))
                    continue

                pending_fetch.append(qid)
                if len(pending_fetch) >= int(args.batch_size):
                    _flush_pending()

            _flush_pending()
        finally:
            cache.close()

        kg_triples_path.parent.mkdir(parents=True, exist_ok=True)
        with kg_triples_path.open("w", encoding="utf-8") as f:
            for h, r, t in state.iter_triples():
                f.write(f"{h}\t{r}\t{t}\n")

        if args.keep_neighbors:
            ents = sorted(state.entities_in_triples())
            kg_entities_path.write_text("\n".join(ents) + ("\n" if ents else ""), encoding="utf-8")

        num_seed_with_edges = state.count_seed_with_edges()
        num_seed_processed = len(state.processed_set())
        num_triples = state.count_triples()
        rel_counts = state.relation_counts()
        top_relations = [{"relation": r, "count": int(c)} for r, c in rel_counts.most_common(int(args.top_relations_n))]

        stats = {
            "num_seed_entities": int(total),
            "num_seed_processed": int(num_seed_processed),
            "num_seed_with_edges": int(num_seed_with_edges),
            "seed_edge_coverage": (float(num_seed_with_edges) / float(total)) if total else 0.0,
            "num_relations_kept": int(len(rel_counts)),
            "num_triples": int(num_triples),
            "avg_triples_per_seed": (float(num_triples) / float(total)) if total else 0.0,
            "top_relations": top_relations,
            "keep_neighbors": bool(args.keep_neighbors),
            "max_triples_per_entity": int(args.max_triples_per_entity),
            "allowed_pid_count": int(len(allowed_pids)),
        }
        kg_stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        state.close()


if __name__ == "__main__":
    main()

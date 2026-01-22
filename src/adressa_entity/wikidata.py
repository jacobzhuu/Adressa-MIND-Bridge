from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests
from requests import Response


def _normalize_for_match(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _string_similarity(a: str, b: str) -> float:
    a_n = _normalize_for_match(a)
    b_n = _normalize_for_match(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0
    return SequenceMatcher(a=a_n, b=b_n).ratio()


@dataclass(frozen=True)
class WikidataCandidate:
    qid: str
    label: str
    description: str | None
    match_score: float


class WikidataSearcher:
    def __init__(
        self,
        *,
        cache_db_path: str | Path,
        user_agent: str = "adressa-entity-pipeline/0.1 (https://wikidata.org; contact: local)",
        sleep_seconds: float = 0.0,
        trust_env: bool = True,
        max_retries: int = 6,
        retry_base_sleep_seconds: float = 0.5,
        retry_max_sleep_seconds: float = 20.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._cache_db_path = Path(cache_db_path)
        self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._cache_db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_cache (
              query TEXT NOT NULL,
              lang TEXT NOT NULL,
              limit_n INTEGER NOT NULL,
              response_json TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              PRIMARY KEY (query, lang, limit_n)
            )
            """
        )
        self._conn.commit()

        self._session = requests.Session()
        self._session.trust_env = bool(trust_env)
        self._session.headers.update({"User-Agent": user_agent})
        self._sleep_seconds = float(sleep_seconds)
        self._max_retries = int(max_retries)
        self._retry_base_sleep_seconds = float(retry_base_sleep_seconds)
        self._retry_max_sleep_seconds = float(retry_max_sleep_seconds)
        self._timeout_seconds = float(timeout_seconds)

    def close(self) -> None:
        self._conn.close()
        self._session.close()

    def search(self, query: str, *, lang: str = "nb", limit_n: int = 10) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        cached = self._conn.execute(
            "SELECT response_json FROM search_cache WHERE query=? AND lang=? AND limit_n=?",
            (query, lang, limit_n),
        ).fetchone()
        if cached is not None:
            return json.loads(cached[0])

        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": lang,
            "format": "json",
            "limit": limit_n,
            "uselang": lang,
            "maxlag": "5",
        }

        resp: Response | None = None
        results: list[dict[str, Any]] | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._sleep_seconds > 0:
                    time.sleep(self._sleep_seconds)
                resp = self._session.get(
                    "https://www.wikidata.org/w/api.php",
                    params=params,
                    timeout=self._timeout_seconds,
                )
                # Retry on common transient errors.
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code} from Wikidata API", response=resp
                    )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "error" in data:
                    err = data.get("error") or {}
                    code = str(err.get("code") or "")
                    if code == "maxlag":
                        raise requests.exceptions.HTTPError("Wikidata maxlag", response=resp)
                results = (data.get("search") or []) if isinstance(data, dict) else []
                break
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                if attempt >= self._max_retries:
                    raise
                base = min(self._retry_base_sleep_seconds * (2**attempt), self._retry_max_sleep_seconds)
                jitter = random.uniform(0.0, base / 2)
                time.sleep(base + jitter)
                continue

        if results is None:
            results = []

        self._conn.execute(
            "INSERT OR REPLACE INTO search_cache(query, lang, limit_n, response_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (query, lang, limit_n, json.dumps(results, ensure_ascii=False), int(time.time())),
        )
        self._conn.commit()
        return results

    def best_candidate(self, mention_surface: str, *, lang: str = "nb", limit_n: int = 10) -> WikidataCandidate | None:
        results = self.search(mention_surface, lang=lang, limit_n=limit_n)
        best: WikidataCandidate | None = None
        for r in results:
            qid = str(r.get("id") or "")
            if not qid.startswith("Q"):
                continue
            label = str(r.get("label") or "")
            if not label:
                continue
            score = _string_similarity(mention_surface, label)
            cand = WikidataCandidate(qid=qid, label=label, description=r.get("description"), match_score=score)
            if best is None or cand.match_score > best.match_score:
                best = cand
        return best


def ner_type_to_mind_type(ner_type: str) -> str:
    t = ner_type.upper()
    if t in {"PER", "PERSON"}:
        return "P"
    if t in {"ORG", "ORGANIZATION"}:
        return "O"
    if t in {"LOC", "LOCATION", "GPE"}:
        return "G"
    return "C"

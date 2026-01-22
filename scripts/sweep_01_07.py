#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
MAGENTA = "\033[0;35m"
BOLD = "\033[1m"
NC = "\033[0m"


def _supports_color() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class Console:
    def __init__(self, *, enable_color: bool, echo_subprocess: bool) -> None:
        self.enable_color = bool(enable_color)
        self.echo_subprocess = bool(echo_subprocess)

    def _c(self, s: str) -> str:
        return s if self.enable_color else strip_ansi(s)

    def header(self, title: str) -> None:
        line = "═" * 72
        print(self._c(f"\n{BOLD}{BLUE}╔{line}╗{NC}"))
        print(self._c(f"{BOLD}{BLUE}║{NC}  {BOLD}{title}{NC}"))
        print(self._c(f"{BOLD}{BLUE}╚{line}╝{NC}"))

    def step(self, msg: str) -> None:
        print(self._c(f"{CYAN}▶{NC} {BOLD}{msg}{NC}"))

    def info(self, msg: str) -> None:
        print(self._c(f"  {YELLOW}ℹ{NC} {msg}"))

    def success(self, msg: str) -> None:
        print(self._c(f"  {GREEN}✓{NC} {msg}"))

    def warn(self, msg: str) -> None:
        print(self._c(f"  {YELLOW}!{NC} {msg}"))

    def error(self, msg: str) -> None:
        print(self._c(f"  {RED}✗{NC} {msg}"))

    def config(self, key: str, value: str) -> None:
        print(self._c(f"  {MAGENTA}•{NC} {key:<18} {value}"))

    def kv(self, key: str, value: str) -> None:
        print(self._c(f"  │ {key:<18} {CYAN}{value}{NC}"))


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, _json_dumps(obj) + "\n")


def _parse_float(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _parse_int(s: str) -> int | None:
    try:
        return int(s)
    except Exception:
        return None


def _pct_to_float(s: str) -> float | None:
    s = s.strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    return _parse_float(s)


def _flatten_config_for_csv(config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in config.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def _copy_news_tsv(base_data_root: Path, dest_data_root: Path, splits: list[str]) -> None:
    for sp in splits:
        src = base_data_root / sp / "news.tsv"
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}")
        dst_dir = dest_data_root / sp
        _ensure_dir(dst_dir)
        (dst_dir / "news.tsv").write_bytes(src.read_bytes())


def _run_cmd(
    *,
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    console: Console,
) -> int:
    _ensure_dir(log_path.parent)
    console.step(f"Running: {shlex.join(cmd)}")
    console.info(f"log: {log_path}")
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {shlex.join(cmd)}\n")
        f.write(f"# cwd: {cwd}\n")
        f.write("# env overrides:\n")
        for k in sorted(env.keys()):
            f.write(f"#   {k}={env[k]}\n")
        f.write("\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            if console.echo_subprocess:
                sys.stdout.write(line)
        return int(p.wait())


@dataclass(frozen=True)
class ParsedMetrics:
    ok: bool
    metrics: dict[str, Any]


_RE_SPLIT_HEADER = re.compile(r"^✓\s*(train|val|test)\s+Split:\s*$")
_RE_TOTAL_ROWS = re.compile(r"Total Rows:\s*([0-9]+)")
_RE_NON_EMPTY = re.compile(r"Non-empty Title Entities:\s*([0-9]+)")
_RE_COV_RATIO = re.compile(r"Coverage Ratio:\s*([0-9.]+%)")

_RE_ENTITY_COV = re.compile(r"^✓\s*Entity Coverage Report:\s*$")
_RE_ENTITIES = re.compile(r"Entities:\s*([0-9]+)")
_RE_MIND_HITS = re.compile(r"MIND Hits:\s*([0-9]+)")
_RE_MIND_MISSES = re.compile(r"MIND Misses:\s*([0-9]+)")
_RE_COVERAGE = re.compile(r"Coverage:\s*([0-9.]+%)")
_RE_DIM = re.compile(r"Dimension:\s*([0-9]+)")

_RE_MENTION_COV = re.compile(r"^✓\s*Mention Coverage Report:\s*$")
_RE_TOTAL_MENTIONS = re.compile(r"Total Mentions:\s*([0-9]+)")
_RE_PRETRAINED_HITS = re.compile(r"Pretrained Hits:\s*([0-9]+)")
_RE_WEIGHTED_COV = re.compile(r"Weighted Coverage:\s*([0-9.]+%)")

_RE_RESULTS_HEADER = re.compile(r"^✓\s*(train|val|test)\s+Results\s+\((.+)\):\s*$")
_RE_MENTIONS_LINE = re.compile(r"Mentions:\s*([0-9]+)")
_RE_UNIQ_ENTS_LINE = re.compile(r"Unique Entities:\s*([0-9]+)")
_RE_VOCAB_LINE = re.compile(r"Entity Vocab:\s*([0-9]+)")
_RE_RECALL_LINE = re.compile(r"Recall@([0-9]+):\s*([0-9.]+)")
_RE_MRR_LINE = re.compile(r"MRR:\s*([0-9.]+)")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def parse_run_01_03_stats(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cur_split: str | None = None
    for raw in strip_ansi(text).splitlines():
        line = raw.strip()
        m = _RE_SPLIT_HEADER.match(line)
        if m:
            cur_split = m.group(1)
            out.setdefault("stage01_03", {}).setdefault(cur_split, {})
            continue
        if not cur_split:
            continue
        d = out["stage01_03"][cur_split]
        m = _RE_TOTAL_ROWS.search(line)
        if m:
            d["total_rows"] = int(m.group(1))
        m = _RE_NON_EMPTY.search(line)
        if m:
            d["non_empty_title_entities"] = int(m.group(1))
        m = _RE_COV_RATIO.search(line)
        if m:
            d["coverage_ratio_pct"] = float(m.group(1)[:-1])
    return out


def parse_run_04_07_stats(text: str, tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {tag: {}}
    state: str | None = None
    result_kind: str | None = None  # full/seen/unseen
    result_split: str | None = None  # train/test/val

    for raw in strip_ansi(text).splitlines():
        line = raw.strip()
        if _RE_ENTITY_COV.match(line):
            state = "entity_cov"
            out[tag].setdefault("entity_init", {})
            continue
        if _RE_MENTION_COV.match(line):
            state = "mention_cov"
            out[tag].setdefault("mention_init", {})
            continue

        m = _RE_RESULTS_HEADER.match(line)
        if m:
            result_split = m.group(1)
            label = m.group(2).lower()
            # NOTE: check 'unseen' before 'seen' because 'unseen' contains 'seen' as substring.
            if "unseen" in label:
                result_kind = "unseen"
            elif "seen" in label:
                result_kind = "seen"
            else:
                result_kind = "full"
            out[tag].setdefault("eval", {}).setdefault(result_split, {}).setdefault(result_kind, {})
            state = "eval"
            continue

        if state == "entity_cov":
            d = out[tag]["entity_init"]
            for pat, key, conv in [
                (_RE_ENTITIES, "entities", int),
                (_RE_MIND_HITS, "mind_hits", int),
                (_RE_MIND_MISSES, "mind_misses", int),
                (_RE_COVERAGE, "coverage_pct", lambda s: float(s[:-1])),
                (_RE_DIM, "dim", int),
            ]:
                mm = pat.search(line)
                if mm:
                    d[key] = conv(mm.group(1))
        elif state == "mention_cov":
            d = out[tag]["mention_init"]
            for pat, key, conv in [
                (_RE_TOTAL_MENTIONS, "total_mentions", int),
                (_RE_PRETRAINED_HITS, "pretrained_hits", int),
                (_RE_WEIGHTED_COV, "weighted_coverage_pct", lambda s: float(s[:-1])),
            ]:
                mm = pat.search(line)
                if mm:
                    d[key] = conv(mm.group(1))
        elif state == "eval" and result_split and result_kind:
            d = out[tag]["eval"][result_split][result_kind]
            mm = _RE_MENTIONS_LINE.search(line)
            if mm:
                d["mentions"] = int(mm.group(1))
            mm = _RE_UNIQ_ENTS_LINE.search(line)
            if mm:
                d["unique_entities"] = int(mm.group(1))
            mm = _RE_VOCAB_LINE.search(line)
            if mm:
                d["entity_vocab"] = int(mm.group(1))
            mm = _RE_RECALL_LINE.search(line)
            if mm:
                k = int(mm.group(1))
                d.setdefault("recall", {})[f"@{k}"] = float(mm.group(2))
            mm = _RE_MRR_LINE.search(line)
            if mm:
                d["mrr"] = float(mm.group(1))

    return out


def _metric_get(metrics: dict[str, Any], path: list[str]) -> float:
    cur: Any = metrics
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return float("-inf")
        cur = cur[p]
    if isinstance(cur, (int, float)):
        return float(cur)
    return float("-inf")


def _unseen_mentions(metrics: dict[str, Any], tag: str) -> int:
    cur = metrics.get(tag, {}).get("eval", {}).get("test", {}).get("unseen", {})
    if isinstance(cur, dict):
        val = cur.get("mentions")
        if isinstance(val, int):
            return val
    return 10**9


def choose_best_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, int]:
        m = row.get("metrics") or {}
        # Prefer SEEN first, then FULL. Use MRR as primary; Recall@1 as secondary.
        seen_mrr = _metric_get(m, ["mindsmall", "eval", "test", "seen", "mrr"])
        seen_r1 = _metric_get(m, ["mindsmall", "eval", "test", "seen", "recall", "@1"])
        seen_r5 = _metric_get(m, ["mindsmall", "eval", "test", "seen", "recall", "@5"])
        full_mrr = _metric_get(m, ["mindsmall", "eval", "test", "full", "mrr"])
        full_r1 = _metric_get(m, ["mindsmall", "eval", "test", "full", "recall", "@1"])
        full_r5 = _metric_get(m, ["mindsmall", "eval", "test", "full", "recall", "@5"])
        unseen = _unseen_mentions(m, "mindsmall")
        return (seen_mrr, seen_r1, seen_r5, full_mrr, full_r1, full_r5, -unseen)

    return max(rows, key=score)


def summarize_by_config(rows: list[dict[str, Any]], grid_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[tuple[str, Any], ...], list[dict[str, Any]]] = {}

    def key(cfg: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        items: list[tuple[str, Any]] = []
        for k in grid_keys:
            v = cfg.get(k)
            if isinstance(v, (dict, list)):
                items.append((k, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((k, v))
        return tuple(items)

    for r in rows:
        cfg = r.get("config") or {}
        cfg_only = {k: cfg.get(k) for k in grid_keys}
        groups.setdefault(key(cfg_only), []).append(r)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("-inf")

    summaries: list[dict[str, Any]] = []
    for k, rs in groups.items():
        cfg: dict[str, Any] = {}
        for kk, vv in k:
            if isinstance(vv, str) and (vv.startswith("[") or vv.startswith("{")):
                try:
                    cfg[kk] = json.loads(vv)
                except Exception:
                    cfg[kk] = vv
            else:
                cfg[kk] = vv

        oks = [r for r in rs if r.get("ok")]

        seen_mrrs = [_metric_get(r.get("metrics") or {}, ["mindsmall", "eval", "test", "seen", "mrr"]) for r in oks]
        seen_r1s = [
            _metric_get(r.get("metrics") or {}, ["mindsmall", "eval", "test", "seen", "recall", "@1"]) for r in oks
        ]
        full_mrrs = [_metric_get(r.get("metrics") or {}, ["mindsmall", "eval", "test", "full", "mrr"]) for r in oks]
        full_r1s = [
            _metric_get(r.get("metrics") or {}, ["mindsmall", "eval", "test", "full", "recall", "@1"]) for r in oks
        ]
        unseen_mentions = [_unseen_mentions(r.get("metrics") or {}, "mindsmall") for r in oks]
        unseen_mean = (sum(unseen_mentions) / len(unseen_mentions)) if unseen_mentions else float("inf")

        summaries.append(
            {
                "config": cfg,
                "num_runs": len(rs),
                "num_ok": len(oks),
                "mindsmall": {
                    "seen": {"mrr_mean": mean(seen_mrrs), "recall@1_mean": mean(seen_r1s)},
                    "full": {"mrr_mean": mean(full_mrrs), "recall@1_mean": mean(full_r1s)},
                    "unseen": {"mentions_mean": unseen_mean},
                },
            }
        )

    return summaries


def choose_best_config(config_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not config_summaries:
        return None

    def score(s: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        if int(s.get("num_ok") or 0) <= 0:
            return (float("-inf"),) * 6
        ms = s.get("mindsmall") or {}
        seen = ms.get("seen") or {}
        full = ms.get("full") or {}
        unseen = ms.get("unseen") or {}
        return (
            float(seen.get("mrr_mean", float("-inf"))),
            float(seen.get("recall@1_mean", float("-inf"))),
            float(full.get("mrr_mean", float("-inf"))),
            float(full.get("recall@1_mean", float("-inf"))),
            -float(unseen.get("mentions_mean", float("inf"))),
            float(s.get("num_ok") or 0),
        )

    return max(config_summaries, key=score)


def _validate_eval_consistency(metrics: dict[str, Any], *, tag: str, split: str = "test") -> list[str]:
    """
    Validate basic consistency of parsed eval blocks, catching common parsing bugs.
    Returns a list of human-readable issues (empty if OK).
    """
    issues: list[str] = []
    eval_d = metrics.get(tag, {}).get("eval", {}).get(split, {})
    if not isinstance(eval_d, dict):
        return [f"{tag}.eval.{split} missing"]

    full = eval_d.get("full") if isinstance(eval_d.get("full"), dict) else None
    seen = eval_d.get("seen") if isinstance(eval_d.get("seen"), dict) else None
    unseen = eval_d.get("unseen") if isinstance(eval_d.get("unseen"), dict) else None

    for name, block in [("full", full), ("seen", seen), ("unseen", unseen)]:
        if block is None:
            issues.append(f"{tag}.eval.{split}.{name} missing")
            continue
        if "mrr" not in block:
            issues.append(f"{tag}.eval.{split}.{name}.mrr missing")
        rec = block.get("recall")
        if not isinstance(rec, dict) or "@1" not in rec:
            issues.append(f"{tag}.eval.{split}.{name}.recall@1 missing")
        if "mentions" not in block:
            issues.append(f"{tag}.eval.{split}.{name}.mentions missing")

    if full and seen and unseen:
        mf = full.get("mentions")
        ms = seen.get("mentions")
        mu = unseen.get("mentions")
        if isinstance(mf, int) and isinstance(ms, int) and isinstance(mu, int):
            if ms + mu != mf:
                issues.append(f"{tag}.eval.{split} mentions mismatch: seen({ms})+unseen({mu}) != full({mf})")

    # Soft sanity check: SEEN should typically outperform FULL. Flag extreme inversion.
    if full and seen:
        full_mrr = full.get("mrr")
        seen_mrr = seen.get("mrr")
        if isinstance(full_mrr, (int, float)) and isinstance(seen_mrr, (int, float)):
            if seen_mrr < 0.01 and full_mrr > 0.1:
                issues.append(f"{tag}.eval.{split} suspicious: seen_mrr({seen_mrr}) << full_mrr({full_mrr})")

    return issues


def _fmt_num(x: Any, default: str = "N/A") -> str:
    if isinstance(x, bool):
        return default
    if isinstance(x, int):
        return f"{x}"
    if isinstance(x, float):
        if x == float("inf"):
            return "inf"
        return f"{x:.2f}"
    return default


def _fmt_float(x: Any, digits: int = 6, default: str = "N/A") -> str:
    if isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        return f"{float(x):.{digits}f}"
    return default


def _get(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def print_run_summary(console: Console, metrics: dict[str, Any]) -> None:
    tr_cov = _get(metrics, ["stage01_03", "train", "coverage_ratio_pct"])
    te_cov = _get(metrics, ["stage01_03", "test", "coverage_ratio_pct"])
    console.kv("01-03 train cov", f"{_fmt_float(tr_cov, 2)}%")
    console.kv("01-03 test cov", f"{_fmt_float(te_cov, 2)}%")

    for tag, label in [("mindsmall", "MINDsmall"), ("mindlarge", "MINDlarge")]:
        seen_mrr = _get(metrics, [tag, "eval", "test", "seen", "mrr"])
        seen_r1 = _get(metrics, [tag, "eval", "test", "seen", "recall", "@1"])
        full_mrr = _get(metrics, [tag, "eval", "test", "full", "mrr"])
        full_r1 = _get(metrics, [tag, "eval", "test", "full", "recall", "@1"])
        unseen_m = _get(metrics, [tag, "eval", "test", "unseen", "mentions"])

        console.info(f"{label} (test)")
        console.kv("SEEN MRR", _fmt_float(seen_mrr))
        console.kv("SEEN R@1", _fmt_float(seen_r1))
        console.kv("FULL MRR", _fmt_float(full_mrr))
        console.kv("FULL R@1", _fmt_float(full_r1))
        console.kv("UNSEEN mentions", _fmt_num(unseen_m))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep hyperparameters for Stage 01-03 + 04-07 pipelines, "
            "save all logs/metrics per run, and select best config (SEEN priority)."
        )
    )
    p.add_argument("--base_data_root", type=Path, default=Path("adressa_one_week_mind_final"))
    p.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated splits to include (must exist under base_data_root).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("artifacts/sweeps") / _now_id(),
        help="Root output directory for this sweep (will be created).",
    )
    p.add_argument(
        "--grid_json",
        type=str,
        default=None,
        help=(
            "JSON object mapping env var name -> list of values. "
            "Example: '{\"WIKIDATA_MIN_MATCH_HEUR\":[0.85,0.9]}'"
        ),
    )
    p.add_argument(
        "--grid_file",
        type=Path,
        default=None,
        help="Path to JSON file mapping env var name -> list of values (same schema as --grid_json).",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each config N times (useful to average randomness).",
    )
    p.add_argument("--echo", action="store_true", help="Echo subprocess output to console while running.")
    p.add_argument("--dry_run", action="store_true", help="Print planned runs without executing.")
    p.add_argument("--no_color", action="store_true", help="Disable ANSI colors in console output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console(enable_color=(not args.no_color) and _supports_color(), echo_subprocess=bool(args.echo))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("--splits is empty")

    grid: dict[str, list[Any]] = {}
    if args.grid_file is not None:
        grid = json.loads(_read_text(args.grid_file))
    elif args.grid_json is not None:
        grid = json.loads(args.grid_json)
    else:
        # Conservative default grid for SEEN-first optimization.
        grid = {
            "WIKIDATA_MIN_MATCH_HEUR": [0.85, 0.9, 0.92],
            "NER_HEURISTIC_MODE": ["fallback"],
            "NER_HEURISTIC_MAX_MENTIONS": [4, 6],
            "WIKIDATA_MIN_MATCH": [0.6],
        }

    for k, v in list(grid.items()):
        if not isinstance(v, list) or not v:
            raise SystemExit(f"grid[{k}] must be a non-empty list")

    out_dir: Path = args.out_dir
    _ensure_dir(out_dir)

    console.header("SWEEP: Stage 01-03 + 04-07")
    console.info("Pipelines: run_01_03.sh + run_04_07_mindsmall.sh + run_04_07_mindlarge.sh")
    console.config("out_dir:", str(out_dir))
    console.config("base_data_root:", str(args.base_data_root))
    console.config("splits:", ",".join(splits))
    console.config("repeat:", str(args.repeat))

    sweep_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_data_root": str(args.base_data_root),
        "splits": splits,
        "grid": grid,
        "repeat": int(args.repeat),
        "repo_root": str(REPO_ROOT),
    }
    _write_json(out_dir / "sweep_meta.json", sweep_meta)

    keys = sorted(grid.keys())
    values_list = [grid[k] for k in keys]
    configs: list[dict[str, Any]] = []
    for vals in itertools.product(*values_list):
        cfg = {k: v for k, v in zip(keys, vals)}
        configs.append(cfg)
    expanded: list[dict[str, Any]] = []
    for cfg in configs:
        for r in range(args.repeat):
            expanded.append({**cfg, "_repeat_idx": r})

    console.info(f"grid keys: {', '.join(keys)}")
    console.info(f"runs: {len(expanded)} (configs={len(configs)} × repeat={args.repeat})")

    if args.dry_run:
        console.header("DRY RUN")
        for i, cfg in enumerate(expanded, start=1):
            console.info(f"{i:03d}: {cfg}")
        return

    rows: list[dict[str, Any]] = []

    for idx, cfg in enumerate(expanded, start=1):
        run_id = f"{idx:03d}"
        console.header(f"RUN {run_id}/{len(expanded):03d}")
        console.info("Hyperparameters:")
        for k in keys:
            if k in cfg:
                console.config(f"{k}:", str(cfg.get(k)))
        console.config("repeat_idx:", str(cfg.get("_repeat_idx", 0)))

        run_dir = out_dir / "runs" / run_id
        data_root = run_dir / "data_root"
        artifacts_dir = run_dir / "artifacts"
        out_init_small = run_dir / "out" / "entities_mindsmall"
        out_train_small = run_dir / "out" / "train_mindsmall"
        out_init_large = run_dir / "out" / "entities_mindlarge"
        out_train_large = run_dir / "out" / "train_mindlarge"

        _ensure_dir(run_dir)
        _copy_news_tsv(args.base_data_root, data_root, splits)
        console.info(f"run_dir:  {run_dir}")
        console.info(f"data_root: {data_root}")
        console.info(f"artifacts: {artifacts_dir}")

        env: dict[str, str] = {
            "DATA_ROOT": str(data_root),
            "ARTIFACTS_DIR": str(artifacts_dir),
            # keep shared cache for speed (wikidata + huggingface).
            "CACHE_DIR": "cache",
            # isolate 04-07 outputs
            "OUT_INIT_DIR": str(out_init_small),
            "OUT_TRAIN_DIR": str(out_train_small),
        }
        for k, v in cfg.items():
            if k == "_repeat_idx":
                continue
            env[k] = str(v)

        _write_json(run_dir / "config.json", {"run_id": run_id, "env": env, "raw": cfg})

        logs_dir = run_dir / "logs"
        t0 = time.time()
        console.header("Step 01-03: NER + Entity Linking")
        rc1 = _run_cmd(
            cmd=["bash", "scripts/run_01_03.sh"],
            env=env,
            cwd=REPO_ROOT,
            log_path=logs_dir / "run_01_03.log",
            console=console,
        )
        console.success(f"Step 01-03 finished (rc={rc1})")

        # MINDsmall
        env_small = {**env, "OUT_INIT_DIR": str(out_init_small), "OUT_TRAIN_DIR": str(out_train_small)}
        console.header("Step 04-07: MINDsmall")
        rc2 = _run_cmd(
            cmd=["bash", "scripts/run_04_07_mindsmall.sh"],
            env=env_small,
            cwd=REPO_ROOT,
            log_path=logs_dir / "run_04_07_mindsmall.log",
            console=console,
        )
        console.success(f"Step 04-07 MINDsmall finished (rc={rc2})")

        # MINDlarge
        env_large = {**env, "OUT_INIT_DIR": str(out_init_large), "OUT_TRAIN_DIR": str(out_train_large)}
        console.header("Step 04-07: MINDlarge")
        rc3 = _run_cmd(
            cmd=["bash", "scripts/run_04_07_mindlarge.sh"],
            env=env_large,
            cwd=REPO_ROOT,
            log_path=logs_dir / "run_04_07_mindlarge.log",
            console=console,
        )
        console.success(f"Step 04-07 MINDlarge finished (rc={rc3})")
        dt = time.time() - t0

        ok_exec = (rc1 == 0) and (rc2 == 0) and (rc3 == 0)
        metrics: dict[str, Any] = {
            "ok": ok_exec,
            "ok_exec": ok_exec,
            "return_codes": {"run_01_03": rc1, "mindsmall": rc2, "mindlarge": rc3},
            "wall_time_sec": dt,
        }

        # Parse metrics from logs (even if partial).
        try:
            metrics.update(parse_run_01_03_stats(_read_text(logs_dir / "run_01_03.log")))
        except Exception as e:
            metrics.setdefault("parse_errors", []).append(f"run_01_03: {e}")
        try:
            metrics.update(parse_run_04_07_stats(_read_text(logs_dir / "run_04_07_mindsmall.log"), "mindsmall"))
        except Exception as e:
            metrics.setdefault("parse_errors", []).append(f"mindsmall: {e}")
        try:
            metrics.update(parse_run_04_07_stats(_read_text(logs_dir / "run_04_07_mindlarge.log"), "mindlarge"))
        except Exception as e:
            metrics.setdefault("parse_errors", []).append(f"mindlarge: {e}")

        parse_issues: list[str] = []
        parse_issues.extend(_validate_eval_consistency(metrics, tag="mindsmall", split="test"))
        parse_issues.extend(_validate_eval_consistency(metrics, tag="mindlarge", split="test"))
        if parse_issues:
            metrics["parse_issues"] = parse_issues

        ok_parse = ok_exec and not metrics.get("parse_errors") and not parse_issues
        metrics["ok_parse"] = ok_parse
        metrics["ok"] = ok_parse

        _write_json(run_dir / "metrics.json", metrics)

        row = {"run_id": run_id, "ok": metrics.get("ok"), "config": cfg, "metrics": metrics}
        rows.append(row)

        console.header("Result")
        if metrics.get("ok"):
            console.success(f"run ok (time={dt:.1f}s)")
        else:
            console.error(f"run failed (time={dt:.1f}s)")
        print_run_summary(console, metrics)

    _write_json(out_dir / "results.json", {"rows": rows})
    _write_text(out_dir / "results.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    best_run = choose_best_run([r for r in rows if r.get("ok")])
    _write_json(out_dir / "best_run.json", best_run or {})

    config_summaries = summarize_by_config(rows, sorted(grid.keys()))
    _write_json(out_dir / "config_summaries.json", {"rows": config_summaries})
    best_cfg = choose_best_config(config_summaries)
    _write_json(out_dir / "best_config.json", best_cfg or {})

    # Write a flat CSV summary (config + selected metrics).
    csv_path = out_dir / "summary.csv"
    fieldnames: list[str] = ["run_id", "ok"]
    for k in sorted(grid.keys()):
        fieldnames.append(f"cfg.{k}")
    fieldnames += [
        "mindsmall.seen.mrr",
        "mindsmall.seen.recall@1",
        "mindsmall.full.mrr",
        "mindsmall.full.recall@1",
        "mindsmall.unseen.mentions",
        "mindsmall.entity_init.entities",
        "mindsmall.entity_init.coverage_pct",
        "mindsmall.mention_init.total_mentions",
        "mindsmall.mention_init.weighted_coverage_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            metrics = r.get("metrics") or {}
            row_out: dict[str, Any] = {"run_id": r.get("run_id"), "ok": r.get("ok")}
            cfg = r.get("config") or {}
            for k in sorted(grid.keys()):
                row_out[f"cfg.{k}"] = cfg.get(k)
            row_out["mindsmall.seen.mrr"] = _metric_get(metrics, ["mindsmall", "eval", "test", "seen", "mrr"])
            row_out["mindsmall.seen.recall@1"] = _metric_get(
                metrics, ["mindsmall", "eval", "test", "seen", "recall", "@1"]
            )
            row_out["mindsmall.full.mrr"] = _metric_get(metrics, ["mindsmall", "eval", "test", "full", "mrr"])
            row_out["mindsmall.full.recall@1"] = _metric_get(
                metrics, ["mindsmall", "eval", "test", "full", "recall", "@1"]
            )
            row_out["mindsmall.unseen.mentions"] = _unseen_mentions(metrics, "mindsmall")
            row_out["mindsmall.entity_init.entities"] = metrics.get("mindsmall", {}).get("entity_init", {}).get("entities")
            row_out["mindsmall.entity_init.coverage_pct"] = metrics.get("mindsmall", {}).get("entity_init", {}).get(
                "coverage_pct"
            )
            row_out["mindsmall.mention_init.total_mentions"] = metrics.get("mindsmall", {}).get("mention_init", {}).get(
                "total_mentions"
            )
            row_out["mindsmall.mention_init.weighted_coverage_pct"] = metrics.get("mindsmall", {}).get("mention_init", {}).get(
                "weighted_coverage_pct"
            )
            w.writerow(row_out)

    cfg_csv = out_dir / "config_summary.csv"
    cfg_fields = ["num_runs", "num_ok"] + [f"cfg.{k}" for k in sorted(grid.keys())] + [
        "mindsmall.seen.mrr_mean",
        "mindsmall.seen.recall@1_mean",
        "mindsmall.full.mrr_mean",
        "mindsmall.full.recall@1_mean",
        "mindsmall.unseen.mentions_mean",
    ]
    with cfg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cfg_fields)
        w.writeheader()
        for s in config_summaries:
            row_out: dict[str, Any] = {"num_runs": s.get("num_runs"), "num_ok": s.get("num_ok")}
            cfg = s.get("config") or {}
            for k in sorted(grid.keys()):
                row_out[f"cfg.{k}"] = cfg.get(k)
            ms = s.get("mindsmall") or {}
            row_out["mindsmall.seen.mrr_mean"] = (ms.get("seen") or {}).get("mrr_mean")
            row_out["mindsmall.seen.recall@1_mean"] = (ms.get("seen") or {}).get("recall@1_mean")
            row_out["mindsmall.full.mrr_mean"] = (ms.get("full") or {}).get("mrr_mean")
            row_out["mindsmall.full.recall@1_mean"] = (ms.get("full") or {}).get("recall@1_mean")
            row_out["mindsmall.unseen.mentions_mean"] = (ms.get("unseen") or {}).get("mentions_mean")
            w.writerow(row_out)

    if best_cfg:
        console.header("BEST CONFIG (SEEN priority)")
        cfg = best_cfg.get("config") or {}
        for k in sorted(grid.keys()):
            if k in cfg:
                console.config(f"{k}:", str(cfg.get(k)))
        ms = best_cfg.get("mindsmall") or {}
        console.kv("SEEN MRR (mean)", _fmt_float((ms.get("seen") or {}).get("mrr_mean")))
        console.kv("SEEN R@1 (mean)", _fmt_float((ms.get("seen") or {}).get("recall@1_mean")))
        console.kv("FULL MRR (mean)", _fmt_float((ms.get("full") or {}).get("mrr_mean")))
        console.kv("FULL R@1 (mean)", _fmt_float((ms.get("full") or {}).get("recall@1_mean")))
        console.kv("UNSEEN mentions", _fmt_num((ms.get("unseen") or {}).get("mentions_mean")))
        console.info(f"saved: {out_dir/'best_config.json'}")
    else:
        console.header("BEST CONFIG (SEEN priority)")
        console.warn("None (no successful configs)")

    if best_run:
        console.header("BEST RUN")
        console.info(f"run_id: {best_run.get('run_id')}")
        cfg = best_run.get("config") or {}
        for k in sorted(grid.keys()):
            if k in cfg:
                console.config(f"{k}:", str(cfg.get(k)))
        console.info(f"saved: {out_dir/'best_run.json'}")
    else:
        console.header("BEST RUN")
        console.warn("None (no successful runs)")

    console.header("Artifacts")
    console.info(f"per-run outputs: {out_dir/'runs'}")
    console.info(f"run summary CSV: {csv_path}")
    console.info(f"config summary CSV: {cfg_csv}")
    console.info(f"best config: {out_dir/'best_config.json'}")
    console.info(f"best run: {out_dir/'best_run.json'}")


if __name__ == "__main__":
    main()

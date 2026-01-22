#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _metric_get(metrics: dict[str, Any], path: list[str]) -> float:
    cur: Any = metrics
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return float("-inf")
        cur = cur[p]
    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
        return float(cur)
    return float("-inf")


def _unseen_mentions(metrics: dict[str, Any], tag: str) -> int:
    cur = metrics.get(tag, {}).get("eval", {}).get("test", {}).get("unseen", {})
    if isinstance(cur, dict):
        val = cur.get("mentions")
        if isinstance(val, int):
            return val
    return 10**9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Re-analyze an existing sweep directory by re-parsing per-run log files, "
            "then regenerate summary CSVs and best_config/best_run without re-running training."
        )
    )
    p.add_argument("sweep_dir", type=Path, help="e.g. artifacts/sweeps/20260119_231633")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <sweep_dir>/reanalyzed)",
    )
    p.add_argument("--echo", action="store_true", help="Print per-run progress.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir: Path = args.sweep_dir
    runs_dir = sweep_dir / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"Missing {runs_dir}")

    out_dir = args.out_dir or (sweep_dir / "reanalyzed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import parsing helpers from sweep_01_07.py (same folder).
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sweep_01_07 as sw  # type: ignore

    meta_path = sweep_dir / "sweep_meta.json"
    grid_keys: list[str] = []
    if meta_path.exists():
        try:
            meta = json.loads(_read_text(meta_path))
            grid = meta.get("grid") or {}
            if isinstance(grid, dict):
                grid_keys = sorted(grid.keys())
        except Exception:
            grid_keys = []

    rows: list[dict[str, Any]] = []
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    for rd in run_dirs:
        run_id = rd.name
        cfg_path = rd / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(_read_text(cfg_path))
        raw_cfg = cfg.get("raw") or {}

        logs = rd / "logs"
        m: dict[str, Any] = {"ok": True}
        parse_errors: list[str] = []
        try:
            m.update(sw.parse_run_01_03_stats(_read_text(logs / "run_01_03.log")))
        except Exception as e:
            parse_errors.append(f"run_01_03: {e}")
        try:
            m.update(sw.parse_run_04_07_stats(_read_text(logs / "run_04_07_mindsmall.log"), "mindsmall"))
        except Exception as e:
            parse_errors.append(f"mindsmall: {e}")
        try:
            m.update(sw.parse_run_04_07_stats(_read_text(logs / "run_04_07_mindlarge.log"), "mindlarge"))
        except Exception as e:
            parse_errors.append(f"mindlarge: {e}")

        parse_issues: list[str] = []
        parse_issues.extend(sw._validate_eval_consistency(m, tag="mindsmall", split="test"))  # type: ignore[attr-defined]
        parse_issues.extend(sw._validate_eval_consistency(m, tag="mindlarge", split="test"))  # type: ignore[attr-defined]

        ok = (not parse_errors) and (not parse_issues)
        m["ok"] = ok
        if parse_errors:
            m["parse_errors"] = parse_errors
        if parse_issues:
            m["parse_issues"] = parse_issues

        # Save per-run re-parsed metrics for provenance.
        _write_json(rd / "metrics.reanalyzed.json", m)

        rows.append({"run_id": run_id, "ok": ok, "config": raw_cfg, "metrics": m})
        if args.echo:
            seen_mrr = _metric_get(m, ["mindsmall", "eval", "test", "seen", "mrr"])
            full_mrr = _metric_get(m, ["mindsmall", "eval", "test", "full", "mrr"])
            print(f"[reanalyze] {run_id} ok={ok} seen_mrr={seen_mrr:.6f} full_mrr={full_mrr:.6f}")

    _write_json(out_dir / "results.json", {"rows": rows})
    _write_text(out_dir / "results.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    # Group/summarize configs using sweep helper (so objective matches sweep).
    if not grid_keys:
        # Infer grid keys from configs (exclude _repeat_idx).
        ks = set()
        for r in rows:
            for k in (r.get("config") or {}).keys():
                if k != "_repeat_idx":
                    ks.add(k)
        grid_keys = sorted(ks)

    config_summaries = sw.summarize_by_config(rows, grid_keys)  # type: ignore[attr-defined]
    best_cfg = sw.choose_best_config(config_summaries)  # type: ignore[attr-defined]
    best_run = sw.choose_best_run([r for r in rows if r.get("ok")])  # type: ignore[attr-defined]

    _write_json(out_dir / "config_summaries.json", {"rows": config_summaries})
    _write_json(out_dir / "best_config.json", best_cfg or {})
    _write_json(out_dir / "best_run.json", best_run or {})

    # Flat per-run CSV
    summary_csv = out_dir / "summary.csv"
    fieldnames = ["run_id", "ok"] + [f"cfg.{k}" for k in grid_keys] + [
        "mindsmall.seen.mrr",
        "mindsmall.seen.recall@1",
        "mindsmall.full.mrr",
        "mindsmall.full.recall@1",
        "mindsmall.unseen.mentions",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            m = r.get("metrics") or {}
            cfg = r.get("config") or {}
            row_out: dict[str, Any] = {"run_id": r.get("run_id"), "ok": r.get("ok")}
            for k in grid_keys:
                row_out[f"cfg.{k}"] = cfg.get(k)
            row_out["mindsmall.seen.mrr"] = _metric_get(m, ["mindsmall", "eval", "test", "seen", "mrr"])
            row_out["mindsmall.seen.recall@1"] = _metric_get(m, ["mindsmall", "eval", "test", "seen", "recall", "@1"])
            row_out["mindsmall.full.mrr"] = _metric_get(m, ["mindsmall", "eval", "test", "full", "mrr"])
            row_out["mindsmall.full.recall@1"] = _metric_get(m, ["mindsmall", "eval", "test", "full", "recall", "@1"])
            row_out["mindsmall.unseen.mentions"] = _unseen_mentions(m, "mindsmall")
            w.writerow(row_out)

    # Flat per-config CSV (means)
    cfg_csv = out_dir / "config_summary.csv"
    cfg_fields = ["num_runs", "num_ok"] + [f"cfg.{k}" for k in grid_keys] + [
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
            for k in grid_keys:
                row_out[f"cfg.{k}"] = cfg.get(k)
            ms = s.get("mindsmall") or {}
            row_out["mindsmall.seen.mrr_mean"] = (ms.get("seen") or {}).get("mrr_mean")
            row_out["mindsmall.seen.recall@1_mean"] = (ms.get("seen") or {}).get("recall@1_mean")
            row_out["mindsmall.full.mrr_mean"] = (ms.get("full") or {}).get("mrr_mean")
            row_out["mindsmall.full.recall@1_mean"] = (ms.get("full") or {}).get("recall@1_mean")
            row_out["mindsmall.unseen.mentions_mean"] = (ms.get("unseen") or {}).get("mentions_mean")
            w.writerow(row_out)

    print(f"[reanalyze] wrote: {out_dir}")
    print(f"[reanalyze] best_config: {out_dir/'best_config.json'}")
    print(f"[reanalyze] best_run: {out_dir/'best_run.json'}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur.parent] + list(cur.parents):
        if (parent / "src").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"Could not find repo root from {cur}")


REPO_ROOT = _repo_root()

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
        tqdm.write(self._c(f"\n{BOLD}{BLUE}╔{line}╗{NC}"))
        tqdm.write(self._c(f"{BOLD}{BLUE}║{NC}  {BOLD}{title}{NC}"))
        tqdm.write(self._c(f"{BOLD}{BLUE}╚{line}╝{NC}"))

    def step(self, msg: str) -> None:
        tqdm.write(self._c(f"{CYAN}▶{NC} {BOLD}{msg}{NC}"))

    def info(self, msg: str) -> None:
        tqdm.write(self._c(f"  {YELLOW}ℹ{NC} {msg}"))

    def success(self, msg: str) -> None:
        tqdm.write(self._c(f"  {GREEN}✓{NC} {msg}"))

    def warn(self, msg: str) -> None:
        tqdm.write(self._c(f"  {YELLOW}!{NC} {msg}"))

    def error(self, msg: str) -> None:
        tqdm.write(self._c(f"  {RED}✗{NC} {msg}"))

    def config(self, key: str, value: str) -> None:
        tqdm.write(self._c(f"  {MAGENTA}•{NC} {key:<22} {value}"))



_ANSI_RE = __import__("re").compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _default_mind_rel_vec() -> Path:
    preferred = Path("data/mind/MINDlarge/train/relation_embedding.vec")
    legacy = Path("MINDlarge/train/relation_embedding.vec")
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _coerce_list(v: Any) -> list[Any]:
    if isinstance(v, list):
        return v
    return [v]


def _grid_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out: list[dict[str, Any]] = []
    for prod in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, prod)})
    return out


def _run_cmd(*, cmd: list[str], cwd: Path, env_overrides: dict[str, str], log_path: Path, console: Console) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console.step(f"Running: {shlex.join(cmd)}")
    console.info(f"log: {log_path}")
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {shlex.join(cmd)}\n")
        f.write(f"# cwd: {cwd}\n")
        f.write("# env overrides:\n")
        for k in sorted(env_overrides.keys()):
            f.write(f"#   {k}={env_overrides[k]}\n")
        f.write("\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env={**os.environ, **env_overrides},
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


def _warn_missing_candidate_coverage(*, data_root: Path, eval_split: str, console: Console) -> None:
    # If many candidates are missing from news.tsv, AUC/MRR becomes harder to interpret.
    # We compute a quick estimate on the eval split only.
    news_ids: set[str] = set()
    for p in _discover_news_tsv(data_root):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                nid = line.split("\t", 1)[0].strip()
                if nid:
                    news_ids.add(nid)

    beh = _discover_behaviors_tsv(data_root, eval_split)
    cand = 0
    missing = 0
    with beh.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            for tok in parts[4].split():
                nid = tok.rsplit("-", 1)[0]
                cand += 1
                if nid not in news_ids:
                    missing += 1
    if cand <= 0:
        return
    frac = float(missing) / float(cand)
    if frac >= 0.2:
        console.warn(
            f"High missing candidate coverage on {eval_split}: {missing}/{cand} ({frac:.1%}) candidates not in news.tsv. "
            "Consider regenerating dataset with convert_one_week.py --news-scope all."
        )


@dataclass(frozen=True)
class RunResult:
    run_id: str
    ok: bool
    config: dict[str, Any]
    metrics: dict[str, Any] | None
    primary: float
    secondary: float
    train_dir: Path
    eval_json: Path | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep TransE hyperparams and select best by downstream AUC/MRR (entity-avg proxy).")
    p.add_argument("--data_root", type=Path, default=_default_data_root())
    p.add_argument("--artifacts_dir", type=Path, default=_default_artifacts_dir())
    p.add_argument("--init_dir", type=Path, default=None, help="Step 04 output dir (entity_vocab.txt + entity_init.npy).")
    p.add_argument("--kg_triples", type=Path, default=None, help="kg_triples.txt path (from step 09).")
    p.add_argument(
        "--mind_entity_vec",
        type=Path,
        nargs="+",
        default=None,
        help="Optional MIND entity_embedding.vec paths (used by TransE to initialize/freeze neighbor anchors).",
    )
    p.add_argument("--mind_relation_vec", type=Path, default=_default_mind_rel_vec())
    p.add_argument("--device", type=str, default=os.environ.get("TRAIN_DEVICE") or "mps")
    p.add_argument("--eval_split", choices=["train", "val", "test"], default="test")
    p.add_argument("--entity_weight", choices=["uniform", "confidence"], default="confidence")
    p.add_argument("--score", choices=["cosine", "dot"], default="cosine")
    p.add_argument("--tie_mode", choices=["expected", "optimistic", "pessimistic"], default="expected")
    p.add_argument("--objective", type=str, default="AUC", help="Metric key under metrics['transe'][...], primary sort desc.")
    p.add_argument("--secondary", type=str, default="MRR", help="Secondary tiebreak metric key under metrics['transe'][...].")
    p.add_argument("--repeat", type=int, default=1, help="Repeat each config with different seeds.")
    p.add_argument("--seed", type=int, default=42, help="Base seed for repeats.")
    p.add_argument("--max_runs", type=int, default=None, help="Optional limit number of runs (after grid expansion).")
    p.add_argument("--output_root", type=Path, default=None, help="Output directory (default: artifacts_dir/sweeps_transe_reco/<ts>).")
    p.add_argument("--grid_json", type=str, default=None, help="JSON dict: param -> list of values.")
    p.add_argument("--grid_file", type=Path, default=None, help="Path to JSON file: param -> list of values.")
    p.add_argument("--dry_run", action="store_true", help="Print planned runs without executing.")
    p.add_argument("--echo", action="store_true", help="Echo subprocess output to stdout.")
    p.add_argument("--no_color", action="store_true")
    p.add_argument(
        "--promote_best_to",
        type=Path,
        default=None,
        help="If set, copy best entity_trained.npy into this directory (and write best_config.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console(enable_color=_supports_color() and (not args.no_color), echo_subprocess=bool(args.echo))

    init_dir = args.init_dir or (Path(args.artifacts_dir) / "entities_mindlarge")
    kg_triples = args.kg_triples or (Path(args.artifacts_dir) / "wikidata_subgraph" / "kg_triples.txt")
    output_root = args.output_root or (Path(args.artifacts_dir) / "sweeps_transe_reco" / _now_id())

    entity_vocab = init_dir / "entity_vocab.txt"
    entity_init = init_dir / "entity_init.npy"
    entity_init_mask = init_dir / "entity_init_mask.npy"
    if not entity_vocab.exists():
        raise FileNotFoundError(f"Missing {entity_vocab}")
    if not entity_init.exists() or not entity_init_mask.exists():
        raise FileNotFoundError(f"Missing entity_init.npy or entity_init_mask.npy under {init_dir}")
    if not kg_triples.exists():
        raise FileNotFoundError(f"Missing {kg_triples}")
    if args.mind_entity_vec is not None:
        for p in args.mind_entity_vec:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing {p}")
    if not args.mind_relation_vec.exists():
        raise FileNotFoundError(f"Missing {args.mind_relation_vec}")

    news_tsv = _discover_news_tsv(args.data_root)
    if not news_tsv:
        raise FileNotFoundError(f"No news.tsv found under {args.data_root}/(train|val|test)")
    behaviors = _discover_behaviors_tsv(args.data_root, str(args.eval_split))

    console.header("SWEEP: TransE → downstream AUC/MRR")
    console.config("data_root", str(args.data_root))
    console.config("artifacts_dir", str(args.artifacts_dir))
    console.config("init_dir", str(init_dir))
    console.config("kg_triples", str(kg_triples))
    if args.mind_entity_vec:
        console.config("mind_entity_vec", f"{len(args.mind_entity_vec)} files")
    console.config("mind_relation_vec", str(args.mind_relation_vec))
    console.config("device", str(args.device))
    console.config("eval_split", str(args.eval_split))
    console.config("objective", f"{args.objective} then {args.secondary}")
    console.config("repeat", str(int(args.repeat)))

    _warn_missing_candidate_coverage(data_root=args.data_root, eval_split=str(args.eval_split), console=console)

    default_grid = {
        "epochs": [10],
        "lr": [1e-3, 5e-4, 2e-3],
        "margin": [0.5, 1.0, 2.0],
        "batch_size": [1024],
        "neg_ratio": [1, 4, 8],
        "weight_decay": [1e-4],
        "max_entity_norm": [1.0],
        "init_from_anchors": [1],
        "relation_weighting": ["sqrt_inv"],
    }

    grid: dict[str, list[Any]] = {k: _coerce_list(v) for k, v in default_grid.items()}
    if args.grid_file is not None:
        grid = {k: _coerce_list(v) for k, v in _read_json(args.grid_file).items()}
    if args.grid_json is not None:
        grid = {k: _coerce_list(v) for k, v in json.loads(args.grid_json).items()}

    configs = _grid_product(grid)
    if args.max_runs is not None:
        configs = configs[: int(args.max_runs)]

    console.info(f"grid_size={len(configs)} (expanded)")
    if args.dry_run:
        for i, cfg in enumerate(configs, start=1):
            console.info(f"run[{i:03d}]: {cfg}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "sweep_config.json",
        {
            "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": str(args.data_root),
            "init_dir": str(init_dir),
            "kg_triples": str(kg_triples),
            "mind_entity_vec": [str(p) for p in (args.mind_entity_vec or [])],
            "mind_relation_vec": str(args.mind_relation_vec),
            "device": str(args.device),
            "eval_split": str(args.eval_split),
            "entity_weight": str(args.entity_weight),
            "score": str(args.score),
            "tie_mode": str(args.tie_mode),
            "objective": str(args.objective),
            "secondary": str(args.secondary),
            "repeat": int(args.repeat),
            "seed": int(args.seed),
            "grid": grid,
            "grid_size": len(configs),
        },
    )

    results: list[RunResult] = []
    run_counter = 0

    results: list[RunResult] = []
    run_counter = 0

    pbar = tqdm(configs, desc="Sweeping", unit="cfg")
    for cfg_idx, cfg in enumerate(pbar, start=1):
        for rep in range(int(args.repeat)):
            run_counter += 1
            run_id = f"run_{run_counter:04d}"
            run_seed = int(args.seed) + rep

            run_dir = output_root / "runs" / run_id
            train_dir = run_dir / "transe_train"
            eval_dir = run_dir / "eval"
            logs_dir = run_dir / "logs"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_cfg = {"run_id": run_id, "cfg_idx": cfg_idx, "repeat_idx": rep, "seed": run_seed, "config": cfg}
            _write_json(run_dir / "config.json", run_cfg)

            # Train
            train_cmd = [
                sys.executable,
                "scripts/transe/10_train_transe.py",
                "--kg_triples",
                str(kg_triples),
                "--seed_entity_vocab",
                str(entity_vocab),
                "--entity_init",
                str(entity_init),
                "--entity_init_mask",
                str(entity_init_mask),
            ]
            if args.mind_entity_vec:
                train_cmd += ["--mind_entity_vec", *[str(p) for p in args.mind_entity_vec]]
            train_cmd += [
                "--mind_relation_vec",
                str(args.mind_relation_vec),
                "--output_dir",
                str(train_dir),
                "--device",
                str(args.device),
                "--seed",
                str(run_seed),
            ]
            # Numeric params.
            for k in ("epochs", "lr", "margin", "batch_size", "neg_ratio", "neg_resample_max", "weight_decay", "max_entity_norm"):
                if k in cfg:
                    train_cmd += [f"--{k}", str(cfg[k])]
            # Bool: init_from_anchors
            if int(cfg.get("init_from_anchors", 0)) == 1:
                train_cmd.append("--init_from_anchors")
            else:
                train_cmd.append("--no-init_from_anchors")
            # Enum: relation_weighting
            if "relation_weighting" in cfg:
                train_cmd += ["--relation_weighting", str(cfg["relation_weighting"])]

            rc = _run_cmd(
                cmd=train_cmd,
                cwd=REPO_ROOT,
                env_overrides={},
                log_path=logs_dir / "train.log",
                console=console,
            )
            if rc != 0:
                console.error(f"{run_id}: train failed (rc={rc})")
                results.append(
                    RunResult(
                        run_id=run_id,
                        ok=False,
                        config=run_cfg,
                        metrics=None,
                        primary=float("nan"),
                        secondary=float("nan"),
                        train_dir=train_dir,
                        eval_json=None,
                    )
                )
                continue

            trained_npy = train_dir / "entity_trained.npy"
            if not trained_npy.exists():
                console.error(f"{run_id}: missing {trained_npy}")
                results.append(
                    RunResult(
                        run_id=run_id,
                        ok=False,
                        config=run_cfg,
                        metrics=None,
                        primary=float("nan"),
                        secondary=float("nan"),
                        train_dir=train_dir,
                        eval_json=None,
                    )
                )
                continue

            # Eval
            eval_json = eval_dir / f"{args.eval_split}.metrics.json"
            eval_cmd = [
                sys.executable,
                "scripts/transe/12_eval_reco_entityavg.py",
                "--news_tsv",
                *[str(p) for p in news_tsv],
                "--behaviors_tsv",
                str(behaviors),
                "--entity_vocab",
                str(entity_vocab),
                "--entity_matrix",
                "init",
                str(entity_init),
                "--entity_matrix",
                "transe",
                str(trained_npy),
                "--entity_weight",
                str(args.entity_weight),
                "--score",
                str(args.score),
                "--tie_mode",
                str(args.tie_mode),
                "--output_json",
                str(eval_json),
            ]
            rc2 = _run_cmd(
                cmd=eval_cmd,
                cwd=REPO_ROOT,
                env_overrides={},
                log_path=logs_dir / f"eval_{args.eval_split}.log",
                console=console,
            )
            if rc2 != 0 or not eval_json.exists():
                console.error(f"{run_id}: eval failed (rc={rc2})")
                results.append(
                    RunResult(
                        run_id=run_id,
                        ok=False,
                        config=run_cfg,
                        metrics=None,
                        primary=float("nan"),
                        secondary=float("nan"),
                        train_dir=train_dir,
                        eval_json=eval_json if eval_json.exists() else None,
                    )
                )
                continue

            metrics_all = _read_json(eval_json).get("metrics") or {}
            transe_metrics = metrics_all.get("transe") or {}
            try:
                primary = float(transe_metrics.get(str(args.objective)))
                secondary = float(transe_metrics.get(str(args.secondary)))
            except Exception:
                primary = float("nan")
                secondary = float("nan")

            console.success(f"{run_id}: {args.objective}={primary:.6f} {args.secondary}={secondary:.6f}")
            results.append(
                RunResult(
                    run_id=run_id,
                    ok=True,
                    config=run_cfg,
                    metrics=metrics_all,
                    primary=primary,
                    secondary=secondary,
                    train_dir=train_dir,
                    eval_json=eval_json,
                )
            )

    # Pick best.
    ok_runs = [r for r in results if r.ok and (r.primary == r.primary) and (r.secondary == r.secondary)]  # NaN check
    if not ok_runs:
        console.error("No successful runs.")
        return

    ok_runs.sort(key=lambda r: (r.primary, r.secondary), reverse=True)
    best = ok_runs[0]

    # Summary CSV.
    summary_path = output_root / "summary.csv"
    keys = sorted({k for rr in results for k in (rr.config.get("config") or {}).keys()})
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_id",
                "ok",
                f"transe_{args.objective}",
                f"transe_{args.secondary}",
                "eval_json",
                "train_dir",
                *keys,
            ]
        )
        for r in results:
            cfg = r.config.get("config") or {}
            w.writerow(
                [
                    r.run_id,
                    int(r.ok),
                    f"{r.primary:.8f}" if r.primary == r.primary else "",
                    f"{r.secondary:.8f}" if r.secondary == r.secondary else "",
                    str(r.eval_json) if r.eval_json else "",
                    str(r.train_dir),
                    *[cfg.get(k, "") for k in keys],
                ]
            )

    best_obj = {
        "run_id": best.run_id,
        "objective": str(args.objective),
        "secondary": str(args.secondary),
        "primary": float(best.primary),
        "secondary_value": float(best.secondary),
        "config": best.config,
        "metrics": best.metrics,
        "train_dir": str(best.train_dir),
        "eval_json": str(best.eval_json) if best.eval_json else None,
        "summary_csv": str(summary_path),
    }
    _write_json(output_root / "best_run.json", best_obj)
    _write_json(output_root / "best_config.json", best.config)

    console.header("BEST RUN")
    console.info(f"run_id:    {best.run_id}")
    console.info(f"{args.objective}: {best.primary:.6f}")
    console.info(f"{args.secondary}: {best.secondary:.6f}")
    console.info(f"train_dir: {best.train_dir}")
    console.info(f"eval_json: {best.eval_json}")
    console.info(f"summary:   {summary_path}")

    # Optional promote.
    if args.promote_best_to is not None:
        dst = Path(args.promote_best_to)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "entity_trained.npy").write_bytes((best.train_dir / "entity_trained.npy").read_bytes())
        _write_json(dst / "best_config.json", best.config)
        console.success(f"Promoted best entity_trained.npy to {dst}")


if __name__ == "__main__":
    main()

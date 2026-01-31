#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# 颜色定义
# ═══════════════════════════════════════════════════════════════════════════════
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 计时功能
START_TIME=$(date +%s)
show_timer() {
  local NOW
  NOW=$(date +%s)
  local ELAPSED=$((NOW - START_TIME))
  local H=$((ELAPSED / 3600))
  local M=$(( (ELAPSED % 3600) / 60 ))
  local S=$((ELAPSED % 60))
  echo -e "  ⏱  ${YELLOW}Current Run Time: ${H}h ${M}m ${S}s${NC}"
}

log_header() {
  echo ""
  echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${BLUE}║${NC}  ${BOLD}$1${NC}"
  echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
}

log_info() {
  echo -e "  ${YELLOW}ℹ${NC} $1"
}

log_success() {
  echo -e "  ${GREEN}✓${NC} $1"
}

log_warn() {
  echo -e "  ${YELLOW}!${NC} $1"
}

# 路径兼容（新结构优先，旧结构回退）
resolve_dir_default() {
  local preferred="$1"
  local legacy="$2"
  if [[ -d "$preferred" ]]; then
    echo "$preferred"
  elif [[ -d "$legacy" ]]; then
    echo "$legacy"
  else
    echo "$preferred"
  fi
}

resolve_file_default() {
  local preferred="$1"
  local legacy="$2"
  if [[ -f "$preferred" ]]; then
    echo "$preferred"
  elif [[ -f "$legacy" ]]; then
    echo "$legacy"
  else
    echo "$preferred"
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# 配置（可用环境变量覆盖；默认偏“卷”，但仍建议按需收敛网格/外层循环）
# ═══════════════════════════════════════════════════════════════════════════════
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/artifacts/sota_transe_reco/$RUN_TAG}"

# Data roots (safe by default: base → copy to a fresh work dir)
RAW_INPUT="${RAW_INPUT:-data/raw/adressa_one_week}"
BASE_DATA_ROOT="${BASE_DATA_ROOT:-data/base/adressa_one_week_mind_base}"
MAKE_DATASET="${MAKE_DATASET:-1}"   # 1: auto convert+copy if needed
SKIP_01_03="${SKIP_01_03:-0}"       # 1: assume news.tsv already has title_entities filled

# DATA_ROOT default depends on MAKE_DATASET:
# - MAKE_DATASET=1: create a fresh work dir (timestamped)
# - MAKE_DATASET=0: reuse existing default work dir (avoid surprising "missing path")
DEFAULT_EXISTING_DATA_ROOT="$(resolve_dir_default "data/work/adressa_one_week_mind_final" "adressa_one_week_mind_final")"
DEFAULT_SOTA_DATA_ROOT="data/work/adressa_one_week_mind_sota_$RUN_TAG"
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ "$MAKE_DATASET" == "1" ]]; then
    DATA_ROOT="$DEFAULT_SOTA_DATA_ROOT"
  else
    DATA_ROOT="$DEFAULT_EXISTING_DATA_ROOT"
  fi
fi

# Output dirs
# - ARTIFACTS_DIR is per-run by default (keeps logs/artifacts reproducible)
# - CACHE_DIR is global by default (avoids re-downloading HF models and re-fetching Wikidata across runs)
DEFAULT_GLOBAL_CACHE_DIR="$(resolve_dir_default "outputs/cache" "cache")"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$OUT_ROOT/artifacts}"
CACHE_DIR="${CACHE_DIR:-$DEFAULT_GLOBAL_CACHE_DIR}"

DEFAULT_MIND_VEC_TRAIN="$(resolve_file_default "data/mind/MINDlarge/train/entity_embedding.vec" "MINDlarge/train/entity_embedding.vec")"
DEFAULT_MIND_VEC_VAL="$(resolve_file_default "data/mind/MINDlarge/val/entity_embedding.vec" "MINDlarge/val/entity_embedding.vec")"
DEFAULT_MIND_VEC_TEST="$(resolve_file_default "data/mind/MINDlarge/test/entity_embedding.vec" "MINDlarge/test/entity_embedding.vec")"
DEFAULT_MIND_REL_VEC="$(resolve_file_default "data/mind/MINDlarge/train/relation_embedding.vec" "MINDlarge/train/relation_embedding.vec")"

MIND_VEC_TRAIN="${MIND_VEC_TRAIN:-$DEFAULT_MIND_VEC_TRAIN}"
MIND_VEC_VAL="${MIND_VEC_VAL:-$DEFAULT_MIND_VEC_VAL}"
MIND_VEC_TEST="${MIND_VEC_TEST:-$DEFAULT_MIND_VEC_TEST}"
MIND_REL_VEC="${MIND_REL_VEC:-$DEFAULT_MIND_REL_VEC}"

# Core knobs
TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
ENTITY_WEIGHT="${ENTITY_WEIGHT:-confidence}"  # uniform|confidence
SCORE_FN="${SCORE_FN:-cosine}"                # cosine|dot
TIE_MODE="${TIE_MODE:-expected}"              # expected|optimistic|pessimistic

# Which split to select best config (sweep/polish) and which to report final metrics.
# Defaults are auto-chosen based on which behaviors.tsv exist under DATA_ROOT.
COARSE_EVAL_SPLIT="${COARSE_EVAL_SPLIT:-}"
FINAL_EVAL_SPLIT="${FINAL_EVAL_SPLIT:-}"

# SOTA guardrail: downstream AUC/MRR becomes meaningless if behaviors references news_ids not present in news.tsv.
# Default: fail fast if coverage is incomplete (can override).
REQUIRE_FULL_COVERAGE="${REQUIRE_FULL_COVERAGE:-1}"  # 1 recommended

# Outer loop: KG density
KEEP_NEIGHBORS_LIST="${KEEP_NEIGHBORS_LIST:-1}"       # e.g. "0 1"
MAX_TRIPLES_LIST="${MAX_TRIPLES_LIST:-200 600 1000}"       # e.g. "200 400 800"

# Inner loop: TransE sweep grid (val-based selection)
SWEEP_GRID_FILE="${SWEEP_GRID_FILE:-configs/sweep_grid.transe_reco_sota.json}"
COARSE_REPEAT="${COARSE_REPEAT:-3}"
COARSE_MAX_RUNS="${COARSE_MAX_RUNS:-}"                # optional cap (after grid expansion)

# “终局更卷”：固定最优超参，再多试不同 seed 取 val 最优
POLISH_REPEAT="${POLISH_REPEAT:-10}"

# Wikidata fetch settings
WIKIDATA_SLEEP="${WIKIDATA_SLEEP:-1.0}"
WIKIDATA_TIMEOUT="${WIKIDATA_TIMEOUT:-120}"
WIKIDATA_BATCH_SIZE="${WIKIDATA_BATCH_SIZE:-20}"
WIKIDATA_MAX_RETRIES="${WIKIDATA_MAX_RETRIES:-20}"
WIKIDATA_STEP09_RESTARTS="${WIKIDATA_STEP09_RESTARTS:-20}"            # rerun step09 on fatal error
WIKIDATA_STEP09_RESTART_BASE_SLEEP="${WIKIDATA_STEP09_RESTART_BASE_SLEEP:-30}" # seconds
WIKIDATA_STEP09_RESTART_MAX_SLEEP="${WIKIDATA_STEP09_RESTART_MAX_SLEEP:-600}"  # seconds
WIKIDATA_TRUST_ENV="${WIKIDATA_TRUST_ENV:-0}" # 0 recommended

trust_env_flag="--no-trust-env"
if [[ "$WIKIDATA_TRUST_ENV" == "1" ]]; then
  trust_env_flag="--trust-env"
fi

# Cache
export HF_HOME="${HF_HOME:-$PWD/$CACHE_DIR/huggingface}"
unset TRANSFORMERS_CACHE || true

INIT_DIR="$OUT_ROOT/init_entities_mindlarge"
META_JSONL="$OUT_ROOT/meta_summary.jsonl"
META_CSV="$OUT_ROOT/meta_summary.csv"

OUT_ROOT_NONEMPTY=0
if [[ -d "$OUT_ROOT" ]]; then
  if [[ -n "$(ls -A "$OUT_ROOT" 2>/dev/null || true)" ]]; then
    OUT_ROOT_NONEMPTY=1
  fi
fi

mkdir -p "$OUT_ROOT" "$ARTIFACTS_DIR" "$CACHE_DIR"

log_header "PIPELINE: SOTA Loop (KG × TransE sweep → select by val → polish → test) [MINDlarge]"
log_info "run_tag:     $RUN_TAG"
log_info "out_root:    $OUT_ROOT"
log_info "data_root:   $DATA_ROOT"
log_info "artifacts:   $ARTIFACTS_DIR"
log_info "cache:       $CACHE_DIR"
log_info "device:      $TRAIN_DEVICE"
log_info "eval_metric: weight=$ENTITY_WEIGHT score=$SCORE_FN tie=$TIE_MODE"
log_info "kg_loop:     keep_neighbors=[$KEEP_NEIGHBORS_LIST] max_triples_per_entity=[$MAX_TRIPLES_LIST]"
log_info "sweep_grid:  $SWEEP_GRID_FILE (repeat=$COARSE_REPEAT)"
log_info "polish:      repeat=$POLISH_REPEAT (single best config)"
log_info "wikidata:    sleep=$WIKIDATA_SLEEP timeout=$WIKIDATA_TIMEOUT batch=$WIKIDATA_BATCH_SIZE max_retries=$WIKIDATA_MAX_RETRIES"

if [[ "$MAKE_DATASET" == "1" && "$OUT_ROOT_NONEMPTY" == "1" ]]; then
  log_warn "OUT_ROOT already has files; if this is a fresh run, set a new RUN_TAG/OUT_ROOT to avoid reusing old artifacts."
fi

if [[ ! -f "$SWEEP_GRID_FILE" ]]; then
  echo -e "${RED}ERROR:${NC} Missing sweep grid file: $SWEEP_GRID_FILE"
  exit 1
fi
if [[ ! -f "$MIND_VEC_TRAIN" || ! -f "$MIND_VEC_VAL" || ! -f "$MIND_VEC_TEST" ]]; then
  echo -e "${RED}ERROR:${NC} Missing MIND entity vec(s). Check data/mind/MINDlarge/*/entity_embedding.vec"
  exit 1
fi
if [[ ! -f "$MIND_REL_VEC" ]]; then
  echo -e "${RED}ERROR:${NC} Missing MIND relation vec: $MIND_REL_VEC"
  exit 1
fi

if [[ "$MAKE_DATASET" == "0" && ! -d "$DATA_ROOT" ]]; then
  echo -e "${RED}ERROR:${NC} Missing DATA_ROOT: $DATA_ROOT"
  echo -e "  Fix: set DATA_ROOT=... OR set MAKE_DATASET=1 to auto-create."
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step A: Prepare dataset (convert + copy to a fresh work dir)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$MAKE_DATASET" == "1" ]]; then
  if [[ ! -d "$BASE_DATA_ROOT" ]]; then
    log_header "Step A0: Convert one_week → MIND (base)"
    log_info "raw_input:  $RAW_INPUT"
    log_info "base_out:   $BASE_DATA_ROOT"
    python scripts/steps/00_convert_one_week.py \
      --input "$RAW_INPUT" \
      --output "$BASE_DATA_ROOT" \
      --news-scope split \
      --carry-history 2>&1 | awk 'NF'
    show_timer
    log_success "Converted base dataset"
  else
    log_info "base exists: $BASE_DATA_ROOT"
  fi

  if [[ ! -d "$DATA_ROOT" ]]; then
    log_header "Step A1: Create fresh work DATA_ROOT (copy base)"
    mkdir -p "$(dirname "$DATA_ROOT")"
    cp -R "$BASE_DATA_ROOT" "$DATA_ROOT"
    show_timer
    log_success "Created work dir: $DATA_ROOT"
  else
    log_warn "DATA_ROOT exists, reuse as-is: $DATA_ROOT"
  fi
fi

# Auto-pick eval splits based on available behaviors.tsv (must run after dataset is ready)
pick_first_split_with_behaviors() {
  local a="$1"
  local b="$2"
  local c="$3"
  if [[ -f "$DATA_ROOT/$a/behaviors.tsv" ]]; then
    echo "$a"
    return
  fi
  if [[ -f "$DATA_ROOT/$b/behaviors.tsv" ]]; then
    echo "$b"
    return
  fi
  if [[ -f "$DATA_ROOT/$c/behaviors.tsv" ]]; then
    echo "$c"
    return
  fi
  echo ""
}

if [[ -z "$COARSE_EVAL_SPLIT" ]]; then
  COARSE_EVAL_SPLIT="$(pick_first_split_with_behaviors val train test)"
fi
if [[ -z "$FINAL_EVAL_SPLIT" ]]; then
  FINAL_EVAL_SPLIT="$(pick_first_split_with_behaviors test val train)"
fi

if [[ -z "$COARSE_EVAL_SPLIT" || -z "$FINAL_EVAL_SPLIT" ]]; then
  echo -e "${RED}ERROR:${NC} No behaviors.tsv found under $DATA_ROOT/{train,val,test}"
  exit 1
fi

log_info "select_split: $COARSE_EVAL_SPLIT (sweep/polish)"
log_info "final_split:  $FINAL_EVAL_SPLIT (final report)"
if [[ "$COARSE_EVAL_SPLIT" == "$FINAL_EVAL_SPLIT" ]]; then
  log_warn "select_split == final_split ($FINAL_EVAL_SPLIT): metrics are not holdout. Consider adding a val split."
fi

log_header "Sanity: Coverage (behaviors.tsv → news.tsv)"
python tools/check_mind_coverage.py --data_root "$DATA_ROOT" --eval_split "$FINAL_EVAL_SPLIT" 2>&1 | awk 'NF'
if [[ "$COARSE_EVAL_SPLIT" != "$FINAL_EVAL_SPLIT" ]]; then
  echo ""
  python tools/check_mind_coverage.py --data_root "$DATA_ROOT" --eval_split "$COARSE_EVAL_SPLIT" 2>&1 | awk 'NF'
fi

if [[ "$REQUIRE_FULL_COVERAGE" == "1" ]]; then
  python - "$DATA_ROOT" "$FINAL_EVAL_SPLIT" "$COARSE_EVAL_SPLIT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path


def load_news_ids(data_root: Path) -> set[str]:
    ids: set[str] = set()
    for sp in ("train", "val", "test"):
        p = data_root / sp / "news.tsv"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                nid = line.split("\t", 1)[0].strip()
                if nid:
                    ids.add(nid)
    return ids


def check_split(data_root: Path, split: str, news_ids: set[str]) -> tuple[int, int, int, int]:
    beh = data_root / split / "behaviors.tsv"
    if not beh.exists():
        return 0, 0, 0, 0
    cand_total = cand_missing = hist_total = hist_missing = 0
    with beh.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            hist = parts[3].strip()
            impr = parts[4].strip()
            if hist:
                for nid in hist.split():
                    hist_total += 1
                    if nid not in news_ids:
                        hist_missing += 1
            if impr:
                for tok in impr.split():
                    nid = tok.rsplit("-", 1)[0]
                    cand_total += 1
                    if nid not in news_ids:
                        cand_missing += 1
    return cand_missing, cand_total, hist_missing, hist_total


data_root = Path(sys.argv[1])
splits = [sys.argv[2]]
if len(sys.argv) > 3 and sys.argv[3] and sys.argv[3] not in splits:
    splits.append(sys.argv[3])

news_ids = load_news_ids(data_root)
bad = []
for sp in splits:
    cm, ct, hm, ht = check_split(data_root, sp, news_ids)
    if cm > 0 or hm > 0:
        bad.append((sp, cm, ct, hm, ht))

if bad:
    print("ERROR: Incomplete behaviors→news coverage detected (SOTA requires full coverage).", file=sys.stderr)
    for sp, cm, ct, hm, ht in bad:
        cand_pct = (cm / ct * 100.0) if ct else 0.0
        hist_pct = (hm / ht * 100.0) if ht else 0.0
        print(f"  split={sp}: candidates_missing={cm}/{ct} ({cand_pct:.1f}%), history_missing={hm}/{ht} ({hist_pct:.1f}%)", file=sys.stderr)
    print("Fix: regenerate dataset with updated scripts/steps/00_convert_one_week.py (coverage fix) and rerun.", file=sys.stderr)
    print("Override (not recommended): REQUIRE_FULL_COVERAGE=0", file=sys.stderr)
    raise SystemExit(2)
PY
fi
show_timer

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 01-03: NER + Wikidata linking → write back news.tsv
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_01_03" == "1" ]]; then
  log_warn "SKIP_01_03=1: Skip NER/EL; assume title_entities already present in news.tsv"
else
  log_header "Stage 01-03: NER → Wikidata → write back news.tsv"
  DATA_ROOT="$DATA_ROOT" ARTIFACTS_DIR="$ARTIFACTS_DIR" CACHE_DIR="$CACHE_DIR" TRAIN_DEVICE="$TRAIN_DEVICE" \
    WIKIDATA_TRUST_ENV="$WIKIDATA_TRUST_ENV" WIKIDATA_SLEEP="$WIKIDATA_SLEEP" \
    bash scripts/pipeline/run_01_03.sh
  show_timer
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 04: Build entity vocab + init (MINDlarge)
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 04: Build Entity Vocab + Init (MINDlarge)"

splits=()
for sp in train val test; do
  if [[ -f "$DATA_ROOT/$sp/news.tsv" ]]; then
    splits+=("$sp")
  fi
done
if [[ ${#splits[@]} -eq 0 ]]; then
  echo -e "${RED}ERROR:${NC} No splits found under $DATA_ROOT (expected at least train/test with news.tsv)"
  exit 1
fi

news_args=()
for sp in "${splits[@]}"; do
  news_args+=("$DATA_ROOT/$sp/news.tsv")
done

python scripts/steps/04_build_entity_vocab_and_init.py \
  --news_tsv "${news_args[@]}" \
  --mind_entity_vec "$MIND_VEC_TRAIN" "$MIND_VEC_VAL" "$MIND_VEC_TEST" \
  --output_dir "$INIT_DIR" 2>&1 | awk 'NF'
show_timer
log_success "Built init dir: $INIT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Sweep loops: (KG config) × (TransE grid)
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Sweep Loops (selection split=$COARSE_EVAL_SPLIT)"
echo -n > "$META_JSONL"
echo "keep_neighbors,max_triples_per_entity,select_split,select_AUC,select_MRR,sweep_dir" > "$META_CSV"

extra_sweep_flags=()
if [[ -n "$COARSE_MAX_RUNS" ]]; then
  extra_sweep_flags+=(--max_runs "$COARSE_MAX_RUNS")
fi

fetch_step09() {
  local kg_dir="$1"
  local max_triples="$2"
  shift 2
  local keep_neighbors_flag=("$@")

  local attempt=0
  while true; do
	    if python scripts/transe/09_fetch_wikidata_triples.py \
	      --seed_entity_vocab "$INIT_DIR/entity_vocab.txt" \
	      --mind_relation_vec "$MIND_REL_VEC" \
	      --output_dir "$kg_dir" \
	      --cache_db "$CACHE_DIR/wikidata_entities.sqlite" \
	      --sleep "$WIKIDATA_SLEEP" \
	      --max_triples_per_entity "$max_triples" \
	      ${keep_neighbors_flag[@]+"${keep_neighbors_flag[@]}"} \
	      --max-retries "$WIKIDATA_MAX_RETRIES" \
	      --retry-base-sleep "${WIKIDATA_RETRY_BASE_SLEEP:-2}" \
	      --retry-max-sleep "${WIKIDATA_RETRY_MAX_SLEEP:-120}" \
	      --timeout "$WIKIDATA_TIMEOUT" \
	      --batch_size "$WIKIDATA_BATCH_SIZE" \
	      $trust_env_flag \
	      --resume 2>&1 | awk 'NF'; then
      return 0
    fi

    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$WIKIDATA_STEP09_RESTARTS" ]]; then
      echo -e "${RED}ERROR:${NC} Step 09 failed after $WIKIDATA_STEP09_RESTARTS restarts. Abort."
      return 1
    fi

    local sleep_s=$((WIKIDATA_STEP09_RESTART_BASE_SLEEP * attempt))
    if [[ "$sleep_s" -gt "$WIKIDATA_STEP09_RESTART_MAX_SLEEP" ]]; then
      sleep_s="$WIKIDATA_STEP09_RESTART_MAX_SLEEP"
    fi
    log_warn "Step 09 failed; will resume (restart ${attempt}/${WIKIDATA_STEP09_RESTARTS}) after ${sleep_s}s..."
    sleep "$sleep_s"
  done
}

for keep_neighbors in $KEEP_NEIGHBORS_LIST; do
  keep_neighbors_flag=()
  if [[ "$keep_neighbors" == "1" ]]; then
    keep_neighbors_flag=(--keep_neighbors)
  fi

  for max_triples in $MAX_TRIPLES_LIST; do
    KG_DIR="$OUT_ROOT/kg_nei${keep_neighbors}_k${max_triples}"
    SWEEP_DIR="$OUT_ROOT/sweeps/nei${keep_neighbors}_k${max_triples}"
    PROMOTE_DIR="$OUT_ROOT/promoted/nei${keep_neighbors}_k${max_triples}"

    mkdir -p "$KG_DIR" "$SWEEP_DIR" "$PROMOTE_DIR"

    # Step 09: Fetch triples (resume)
    log_header "Step 09: Fetch Wikidata Triples (nei=${keep_neighbors}, k=${max_triples})"
    fetch_step09 "$KG_DIR" "$max_triples" "${keep_neighbors_flag[@]+"${keep_neighbors_flag[@]}"}"
    show_timer

    if [[ ! -f "$KG_DIR/kg_triples.txt" ]]; then
      echo -e "${RED}ERROR:${NC} Missing $KG_DIR/kg_triples.txt"
      exit 1
    fi

    # Step 10+12 sweep on val
    if [[ -f "$SWEEP_DIR/best_run.json" ]]; then
      log_warn "Skip sweep (exists): $SWEEP_DIR/best_run.json"
    else
      log_header "Sweep TransE → downstream AUC/MRR (val) (nei=${keep_neighbors}, k=${max_triples})"
      python scripts/transe/13_sweep_transe_reco.py \
        --data_root "$DATA_ROOT" \
        --artifacts_dir "$ARTIFACTS_DIR" \
        --init_dir "$INIT_DIR" \
        --kg_triples "$KG_DIR/kg_triples.txt" \
        --mind_entity_vec "$MIND_VEC_TRAIN" "$MIND_VEC_VAL" "$MIND_VEC_TEST" \
        --mind_relation_vec "$MIND_REL_VEC" \
        --device "$TRAIN_DEVICE" \
        --eval_split "$COARSE_EVAL_SPLIT" \
        --entity_weight "$ENTITY_WEIGHT" \
        --score "$SCORE_FN" \
        --tie_mode "$TIE_MODE" \
        --repeat "$COARSE_REPEAT" \
        --grid_file "$SWEEP_GRID_FILE" \
        --output_root "$SWEEP_DIR" \
        --promote_best_to "$PROMOTE_DIR" \
        ${extra_sweep_flags[@]+"${extra_sweep_flags[@]}"}
      show_timer
    fi

    if [[ ! -f "$SWEEP_DIR/best_run.json" ]]; then
      echo -e "${RED}ERROR:${NC} Missing $SWEEP_DIR/best_run.json"
      exit 1
    fi

    python - "$keep_neighbors" "$max_triples" "$COARSE_EVAL_SPLIT" "$KG_DIR" "$SWEEP_DIR" "$PROMOTE_DIR" >> "$META_JSONL" <<'PY'
import json
import sys

keep_neighbors = int(sys.argv[1])
max_triples = int(sys.argv[2])
select_split = sys.argv[3]
kg_dir = sys.argv[4]
sweep_dir = sys.argv[5]
promote_dir = sys.argv[6]

best = json.load(open(f"{sweep_dir}/best_run.json"))
rec = {
    "keep_neighbors": keep_neighbors,
    "max_triples_per_entity": max_triples,
    "select_split": select_split,
    "kg_dir": kg_dir,
    "sweep_dir": sweep_dir,
    "promote_dir": promote_dir,
    "select_AUC": float(best["primary"]),
    "select_MRR": float(best["secondary_value"]),
    "best_run_json": f"{sweep_dir}/best_run.json",
}
print(json.dumps(rec, ensure_ascii=False))
PY

    python - "$keep_neighbors" "$max_triples" "$COARSE_EVAL_SPLIT" "$SWEEP_DIR" >> "$META_CSV" <<'PY'
import json
import sys

keep_neighbors = int(sys.argv[1])
max_triples = int(sys.argv[2])
select_split = sys.argv[3]
sweep_dir = sys.argv[4]
best = json.load(open(f"{sweep_dir}/best_run.json"))
auc = float(best["primary"])
mrr = float(best["secondary_value"])
print(f"{keep_neighbors},{max_triples},{select_split},{auc:.6f},{mrr:.6f},{sweep_dir}")
PY

    log_success "Recorded: nei=${keep_neighbors} k=${max_triples} (${COARSE_EVAL_SPLIT} AUC/MRR) → $SWEEP_DIR/best_run.json"
  done
done

log_header "Select global best by ${COARSE_EVAL_SPLIT} (AUC then MRR)"
BEST_REC_JSON="$(python - "$META_JSONL" <<'PY'
import json
import sys

path = sys.argv[1]
best = None
for line in open(path):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    key = (float(rec["select_AUC"]), float(rec["select_MRR"]))
    if best is None:
        best = rec
        continue
    best_key = (float(best["select_AUC"]), float(best["select_MRR"]))
    if key > best_key:
        best = rec

if best is None:
    raise SystemExit("No records found in meta_summary.jsonl")

print(json.dumps(best))
PY)"

python - "$BEST_REC_JSON" > "$OUT_ROOT/best_global_choice.json" <<'PY'
import json
import sys

print(json.dumps(json.loads(sys.argv[1]), indent=2, ensure_ascii=False))
PY

BEST_KG_DIR="$(python - "$BEST_REC_JSON" <<'PY'
import json,sys
print(json.loads(sys.argv[1])["kg_dir"])
PY)"
BEST_SWEEP_DIR="$(python - "$BEST_REC_JSON" <<'PY'
import json,sys
print(json.loads(sys.argv[1])["sweep_dir"])
PY)"
BEST_PROMOTE_DIR="$(python - "$BEST_REC_JSON" <<'PY'
import json,sys
print(json.loads(sys.argv[1])["promote_dir"])
PY)"
BEST_RUN_JSON="$(python - "$BEST_REC_JSON" <<'PY'
import json,sys
print(json.loads(sys.argv[1])["best_run_json"])
PY)"
BEST_VAL_AUC="$(python - "$BEST_REC_JSON" <<'PY'
import json,sys
print(f'{float(json.loads(sys.argv[1])[\"select_AUC\"]):.6f}')
PY)"
BEST_VAL_MRR="$(python - "$BEST_REC_JSON" <<'PY'
import json,sys
print(f'{float(json.loads(sys.argv[1])[\"select_MRR\"]):.6f}')
PY)"

log_info "best_sweep:   $BEST_SWEEP_DIR"
log_info "best_kg:      $BEST_KG_DIR"
log_info "best_select:  AUC=$BEST_VAL_AUC MRR=$BEST_VAL_MRR"

# ═══════════════════════════════════════════════════════════════════════════════
# Polish: fix best hyperparams, increase seed repeats on val
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Polish (fixed best config × many seeds on val)"
POLISH_DIR="$OUT_ROOT/polish"
BEST_DIR="$OUT_ROOT/best_global"
mkdir -p "$POLISH_DIR" "$BEST_DIR"

BEST_GRID_JSON="$(python - "$BEST_RUN_JSON" <<'PY'
import json
import sys

best = json.load(open(sys.argv[1]))
cfg = best["config"]["config"]
grid = {k: [v] for k, v in cfg.items()}
print(json.dumps(grid))
PY)"

python scripts/transe/13_sweep_transe_reco.py \
  --data_root "$DATA_ROOT" \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --init_dir "$INIT_DIR" \
  --kg_triples "$BEST_KG_DIR/kg_triples.txt" \
  --mind_entity_vec "$MIND_VEC_TRAIN" "$MIND_VEC_VAL" "$MIND_VEC_TEST" \
  --mind_relation_vec "$MIND_REL_VEC" \
  --device "$TRAIN_DEVICE" \
  --eval_split "$COARSE_EVAL_SPLIT" \
  --entity_weight "$ENTITY_WEIGHT" \
  --score "$SCORE_FN" \
  --tie_mode "$TIE_MODE" \
  --repeat "$POLISH_REPEAT" \
  --grid_json "$BEST_GRID_JSON" \
  --output_root "$POLISH_DIR" \
  --promote_best_to "$BEST_DIR"
show_timer

if [[ ! -f "$BEST_DIR/entity_trained.npy" ]]; then
  echo -e "${RED}ERROR:${NC} Missing polished best: $BEST_DIR/entity_trained.npy"
  exit 1
fi

POLISH_BEST_JSON="$POLISH_DIR/best_run.json"
if [[ -f "$POLISH_BEST_JSON" ]]; then
  POLISH_VAL_AUC="$(python - "$POLISH_BEST_JSON" <<'PY'
import json,sys
best=json.load(open(sys.argv[1]))
print(f'{float(best[\"primary\"]):.6f}')
PY)"
  POLISH_VAL_MRR="$(python - "$POLISH_BEST_JSON" <<'PY'
import json,sys
best=json.load(open(sys.argv[1]))
print(f'{float(best[\"secondary_value\"]):.6f}')
PY)"
  log_info "polish_val: AUC=$POLISH_VAL_AUC MRR=$POLISH_VAL_MRR"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Final: evaluate on test + export vec
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Final: Downstream reco eval (${FINAL_EVAL_SPLIT})"
OUT_INIT_DIR="$INIT_DIR" OUT_TRAIN_DIR="$BEST_DIR" OUT_EVAL_DIR="$OUT_ROOT/final_reco_eval" \
  DATA_ROOT="$DATA_ROOT" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
  ENTITY_WEIGHT="$ENTITY_WEIGHT" SCORE_FN="$SCORE_FN" TIE_MODE="$TIE_MODE" SPLITS="$FINAL_EVAL_SPLIT" \
  bash scripts/pipeline/run_12_reco_eval_entityavg_mindlarge.sh

log_header "Final: Export entity_embedding.vec"
python scripts/steps/06_export_entity_embedding_vec.py \
  --entity_vocab "$INIT_DIR/entity_vocab.txt" \
  --entity_matrix "$BEST_DIR/entity_trained.npy" \
  --output_vec "$BEST_DIR/entity_embedding.vec"

for sp in "${splits[@]}"; do
  cp "$BEST_DIR/entity_embedding.vec" "$DATA_ROOT/$sp/entity_embedding.vec"
done
log_success "Copied entity_embedding.vec to splits: ${splits[*]}"

echo ""
show_timer
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ SOTA LOOP COMPLETE${NC}"
echo -e "${GREEN}${BOLD}  • best_choice:  $OUT_ROOT/best_global_choice.json${NC}"
echo -e "${GREEN}${BOLD}  • meta_summary: $META_CSV${NC}"
echo -e "${GREEN}${BOLD}  • final_vec:    $BEST_DIR/entity_embedding.vec${NC}"
echo -e "${GREEN}${BOLD}  • final_metrics: $OUT_ROOT/final_reco_eval/${FINAL_EVAL_SPLIT}.metrics.json${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

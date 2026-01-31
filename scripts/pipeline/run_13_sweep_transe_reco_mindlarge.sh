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
  local NOW=$(date +%s)
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
# 配置
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_DATA_ROOT="$(resolve_dir_default "data/work/adressa_one_week_mind_final" "adressa_one_week_mind_final")"
DEFAULT_ARTIFACTS_DIR="$(resolve_dir_default "outputs/artifacts" "artifacts")"
DEFAULT_CACHE_DIR="$(resolve_dir_default "outputs/cache" "cache")"

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$DEFAULT_ARTIFACTS_DIR}"
CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE_DIR}"

DEFAULT_MIND_VEC_TRAIN="$(resolve_file_default "data/mind/MINDlarge/train/entity_embedding.vec" "MINDlarge/train/entity_embedding.vec")"
DEFAULT_MIND_VEC_VAL="$(resolve_file_default "data/mind/MINDlarge/val/entity_embedding.vec" "MINDlarge/val/entity_embedding.vec")"
DEFAULT_MIND_VEC_TEST="$(resolve_file_default "data/mind/MINDlarge/test/entity_embedding.vec" "MINDlarge/test/entity_embedding.vec")"
DEFAULT_MIND_REL_VEC="$(resolve_file_default "data/mind/MINDlarge/train/relation_embedding.vec" "MINDlarge/train/relation_embedding.vec")"

MIND_VEC_TRAIN="${MIND_VEC_TRAIN:-$DEFAULT_MIND_VEC_TRAIN}"
MIND_VEC_VAL="${MIND_VEC_VAL:-$DEFAULT_MIND_VEC_VAL}"
MIND_VEC_TEST="${MIND_VEC_TEST:-$DEFAULT_MIND_VEC_TEST}"
MIND_REL_VEC="${MIND_REL_VEC:-$DEFAULT_MIND_REL_VEC}"

OUT_INIT_DIR="${OUT_INIT_DIR:-$ARTIFACTS_DIR/entities_mindlarge}"

TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
SWEEP_OUT_DIR="${SWEEP_OUT_DIR:-$ARTIFACTS_DIR/sweeps_transe_reco}"

# KG build knobs (recommended for stronger anchor propagation)
WIKIDATA_KEEP_NEIGHBORS="${WIKIDATA_KEEP_NEIGHBORS:-1}"          # 1 recommended
WIKIDATA_MAX_TRIPLES_PER_ENTITY="${WIKIDATA_MAX_TRIPLES_PER_ENTITY:-200}"
DEFAULT_OUT_KG_DIR="$ARTIFACTS_DIR/wikidata_subgraph_nei${WIKIDATA_KEEP_NEIGHBORS}_k${WIKIDATA_MAX_TRIPLES_PER_ENTITY}"
OUT_KG_DIR="${OUT_KG_DIR:-$DEFAULT_OUT_KG_DIR}"

# Sweep knobs (optional)
SWEEP_EVAL_SPLIT="${SWEEP_EVAL_SPLIT:-test}"
SWEEP_REPEAT="${SWEEP_REPEAT:-1}"
SWEEP_MAX_RUNS="${SWEEP_MAX_RUNS:-}"
SWEEP_ENTITY_WEIGHT="${SWEEP_ENTITY_WEIGHT:-confidence}"  # uniform|confidence
SWEEP_SCORE_FN="${SWEEP_SCORE_FN:-cosine}"                # cosine|dot
SWEEP_TIE_MODE="${SWEEP_TIE_MODE:-expected}"              # expected|optimistic|pessimistic
# Default grid: focused epoch sweep with best params from run_0061
DEFAULT_GRID='{"epochs":[20,40,60,80,100],"lr":[0.0005],"margin":[0.25],"neg_ratio":[1,4,8],"batch_size":[1024],"weight_decay":[0.0001],"max_entity_norm":[1.0],"init_from_anchors":[1],"relation_weighting":["sqrt_inv"]}'
SWEEP_GRID_JSON="${SWEEP_GRID_JSON:-$DEFAULT_GRID}"

WIKIDATA_TRUST_ENV="${WIKIDATA_TRUST_ENV:-0}" # 0 recommended
trust_env_flag="--no-trust-env"
if [[ "$WIKIDATA_TRUST_ENV" == "1" ]]; then
  trust_env_flag="--trust-env"
fi

keep_neighbors_flag=()
if [[ "$WIKIDATA_KEEP_NEIGHBORS" == "1" ]]; then
  keep_neighbors_flag=(--keep_neighbors)
fi

export HF_HOME="${HF_HOME:-$PWD/$CACHE_DIR/huggingface}"
unset TRANSFORMERS_CACHE || true

mkdir -p "$ARTIFACTS_DIR" "$CACHE_DIR" "$SWEEP_OUT_DIR"

log_header "PIPELINE: Sweep TransE → downstream AUC/MRR [MINDlarge]"
log_info "data_root:    $DATA_ROOT"
log_info "artifacts:    $ARTIFACTS_DIR"
log_info "cache:        $CACHE_DIR"
log_info "init_dir:     $OUT_INIT_DIR"
log_info "kg_dir:       $OUT_KG_DIR"
log_info "mind_rel_vec: $MIND_REL_VEC"
log_info "wikidata:     keep_neighbors=$WIKIDATA_KEEP_NEIGHBORS max_triples_per_entity=$WIKIDATA_MAX_TRIPLES_PER_ENTITY"
log_info "device:       $TRAIN_DEVICE"
log_info "eval_split:   $SWEEP_EVAL_SPLIT"
log_info "repeat:       $SWEEP_REPEAT"
log_info "eval_metric:  weight=$SWEEP_ENTITY_WEIGHT score=$SWEEP_SCORE_FN tie=$SWEEP_TIE_MODE"
log_info "out_root:     $SWEEP_OUT_DIR"

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

# Step 04: Build vocab + init (cheap, deterministic)
log_header "Step 04: Build Entity Vocab + Init (MINDlarge)"
python scripts/steps/04_build_entity_vocab_and_init.py \
  --news_tsv "${news_args[@]}" \
  --mind_entity_vec "$MIND_VEC_TRAIN" "$MIND_VEC_VAL" "$MIND_VEC_TEST" \
  --output_dir "$OUT_INIT_DIR"
show_timer
log_success "Built entity_vocab.txt + entity_init.npy"

# Step 09: Fetch triples (resume; will be fast if already done)
log_header "Step 09: Fetch Wikidata Triples (resume)"
python scripts/transe/09_fetch_wikidata_triples.py \
  --seed_entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --mind_relation_vec "$MIND_REL_VEC" \
  --output_dir "$OUT_KG_DIR" \
  --cache_db "$CACHE_DIR/wikidata_entities.sqlite" \
  --sleep "${WIKIDATA_SLEEP:-0.2}" \
  --max_triples_per_entity "$WIKIDATA_MAX_TRIPLES_PER_ENTITY" \
  ${keep_neighbors_flag[@]+"${keep_neighbors_flag[@]}"} \
  --max-retries "${WIKIDATA_MAX_RETRIES:-12}" \
  --retry-base-sleep "${WIKIDATA_RETRY_BASE_SLEEP:-2}" \
  --retry-max-sleep "${WIKIDATA_RETRY_MAX_SLEEP:-120}" \
  --timeout "${WIKIDATA_TIMEOUT:-30}" \
  --batch_size "${WIKIDATA_BATCH_SIZE:-50}" \
  $trust_env_flag \
  --resume
show_timer
log_success "kg_triples ready"

# Sweep (Step 10 + Step 12)
log_header "Sweep: Step 10 (TransE) → Step 12 (AUC/MRR)"

extra=()
if [[ -n "$SWEEP_MAX_RUNS" ]]; then
  extra+=(--max_runs "$SWEEP_MAX_RUNS")
fi
if [[ -n "$SWEEP_GRID_JSON" ]]; then
  extra+=(--grid_json "$SWEEP_GRID_JSON")
fi

python scripts/transe/13_sweep_transe_reco.py \
  --data_root "$DATA_ROOT" \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --init_dir "$OUT_INIT_DIR" \
  --kg_triples "$OUT_KG_DIR/kg_triples.txt" \
  --mind_entity_vec "$MIND_VEC_TRAIN" "$MIND_VEC_VAL" "$MIND_VEC_TEST" \
  --mind_relation_vec "$MIND_REL_VEC" \
  --device "$TRAIN_DEVICE" \
  --eval_split "$SWEEP_EVAL_SPLIT" \
  --entity_weight "$SWEEP_ENTITY_WEIGHT" \
  --score "$SWEEP_SCORE_FN" \
  --tie_mode "$SWEEP_TIE_MODE" \
  --repeat "$SWEEP_REPEAT" \
  --output_root "$SWEEP_OUT_DIR/$(date +%Y%m%d_%H%M%S)" \
  ${extra[@]+"${extra[@]}"}
show_timer

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ SWEEP COMPLETE: see $SWEEP_OUT_DIR/*/best_run.json and summary.csv${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

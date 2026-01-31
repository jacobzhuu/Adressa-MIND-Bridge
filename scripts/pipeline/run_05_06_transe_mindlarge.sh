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

log_header() {
  echo ""
  echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${BLUE}║${NC}  ${BOLD}$1${NC}"
  echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
}

log_step() {
  echo -e "${CYAN}▶${NC} ${BOLD}$1${NC}"
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
OUT_TRAIN_DIR="${OUT_TRAIN_DIR:-$ARTIFACTS_DIR/transe_train}"

TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"

TRANSE_INIT_FROM_ANCHORS="${TRANSE_INIT_FROM_ANCHORS:-1}"   # 1 recommended for downstream AUC/MRR
TRANSE_REL_WEIGHTING="${TRANSE_REL_WEIGHTING:-sqrt_inv}"    # none|sqrt_inv
transe_init_flag="--no-init_from_anchors"
if [[ "$TRANSE_INIT_FROM_ANCHORS" == "1" ]]; then
  transe_init_flag="--init_from_anchors"
fi

# Train hyperparams (optional overrides)
TRANSE_EPOCHS="${TRANSE_EPOCHS:-10}"
TRANSE_LR="${TRANSE_LR:-1e-3}"
TRANSE_MARGIN="${TRANSE_MARGIN:-1.0}"
TRANSE_BATCH_SIZE="${TRANSE_BATCH_SIZE:-1024}"
TRANSE_NEG_RATIO="${TRANSE_NEG_RATIO:-4}"
TRANSE_NEG_RESAMPLE_MAX="${TRANSE_NEG_RESAMPLE_MAX:-50}"
TRANSE_WEIGHT_DECAY="${TRANSE_WEIGHT_DECAY:-1e-4}"
TRANSE_MAX_ENTITY_NORM="${TRANSE_MAX_ENTITY_NORM:-1.0}"
TRANSE_SEED="${TRANSE_SEED:-42}"

# KG build knobs (recommended for stronger anchor propagation)
WIKIDATA_KEEP_NEIGHBORS="${WIKIDATA_KEEP_NEIGHBORS:-1}"          # 1 recommended
WIKIDATA_MAX_TRIPLES_PER_ENTITY="${WIKIDATA_MAX_TRIPLES_PER_ENTITY:-200}"
DEFAULT_OUT_KG_DIR="$ARTIFACTS_DIR/wikidata_subgraph_nei${WIKIDATA_KEEP_NEIGHBORS}_k${WIKIDATA_MAX_TRIPLES_PER_ENTITY}"
OUT_KG_DIR="${OUT_KG_DIR:-$DEFAULT_OUT_KG_DIR}"

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

mkdir -p "$ARTIFACTS_DIR" "$CACHE_DIR"

log_header "PIPELINE: Stage 04 → 09 → 10 → 06 [TransE + MINDlarge]"
log_info "data_root: $DATA_ROOT"
log_info "artifacts: $ARTIFACTS_DIR"
log_info "cache:     $CACHE_DIR"
log_info "mind_vecs:  $MIND_VEC_TRAIN, $MIND_VEC_VAL, $MIND_VEC_TEST"
log_info "mind_rel:   $MIND_REL_VEC"
log_info "wikidata:   keep_neighbors=$WIKIDATA_KEEP_NEIGHBORS max_triples_per_entity=$WIKIDATA_MAX_TRIPLES_PER_ENTITY"
log_info "transe:     init_from_anchors=$TRANSE_INIT_FROM_ANCHORS relation_weighting=$TRANSE_REL_WEIGHTING"
log_info "transe_hp:  epochs=$TRANSE_EPOCHS lr=$TRANSE_LR margin=$TRANSE_MARGIN neg=$TRANSE_NEG_RATIO bs=$TRANSE_BATCH_SIZE wd=$TRANSE_WEIGHT_DECAY max_norm=$TRANSE_MAX_ENTITY_NORM seed=$TRANSE_SEED neg_resample_max=$TRANSE_NEG_RESAMPLE_MAX"

splits=()
for sp in train val test; do
  if [[ -f "$DATA_ROOT/$sp/news.tsv" ]]; then
    splits+=("$sp")
  fi
done
if [[ ${#splits[@]} -eq 0 ]]; then
  echo -e "${RED}ERROR:${NC} No splits found under $DATA_ROOT (expected at least train/val/test with news.tsv)"
  exit 1
fi
log_info "splits: ${splits[*]}"

news_args=()
for sp in "${splits[@]}"; do
  news_args+=("$DATA_ROOT/$sp/news.tsv")
done

# ═══════════════════════════════════════════════════════════════════════════════
# Step 04: Build Entity Vocab + Init (MINDlarge)
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 04: Build Entity Vocab + Init (MINDlarge)"
log_info "output_dir: $OUT_INIT_DIR"

python scripts/steps/04_build_entity_vocab_and_init.py \
  --news_tsv "${news_args[@]}" \
  --mind_entity_vec \
    "$MIND_VEC_TRAIN" \
    "$MIND_VEC_VAL" \
    "$MIND_VEC_TEST" \
  --output_dir "$OUT_INIT_DIR" 2>&1 | awk 'NF'

log_success "Built entity_vocab.txt + entity_init.npy"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 09: Fetch Wikidata triples
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 09: Fetch Wikidata Triples (wbgetentities)"
log_info "output_dir: $OUT_KG_DIR"

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
  --resume 2>&1 | awk 'NF'

log_success "Built kg_triples.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 10: Train TransE (anchor-locked)
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 10: Train TransE (anchor-locked)"
log_info "output_dir: $OUT_TRAIN_DIR"
log_info "device: $TRAIN_DEVICE"

python scripts/transe/10_train_transe.py \
  --kg_triples "$OUT_KG_DIR/kg_triples.txt" \
  --seed_entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --entity_init "$OUT_INIT_DIR/entity_init.npy" \
  --entity_init_mask "$OUT_INIT_DIR/entity_init_mask.npy" \
  --mind_entity_vec "$MIND_VEC_TRAIN" "$MIND_VEC_VAL" "$MIND_VEC_TEST" \
  --mind_relation_vec "$MIND_REL_VEC" \
  --output_dir "$OUT_TRAIN_DIR" \
  --epochs "$TRANSE_EPOCHS" \
  --lr "$TRANSE_LR" \
  --margin "$TRANSE_MARGIN" \
  --batch_size "$TRANSE_BATCH_SIZE" \
  --neg_ratio "$TRANSE_NEG_RATIO" \
  --neg_resample_max "$TRANSE_NEG_RESAMPLE_MAX" \
  --weight_decay "$TRANSE_WEIGHT_DECAY" \
  --max_entity_norm "$TRANSE_MAX_ENTITY_NORM" \
  --seed "$TRANSE_SEED" \
  $transe_init_flag \
  --relation_weighting "$TRANSE_REL_WEIGHTING" \
  --device "$TRAIN_DEVICE" 2>&1 | awk 'NF'

log_success "Wrote entity_trained.npy"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 06: Export entity_embedding.vec + copy to splits
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 06: Export entity_embedding.vec"

python scripts/steps/06_export_entity_embedding_vec.py \
  --entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --entity_matrix "$OUT_TRAIN_DIR/entity_trained.npy" \
  --output_vec "$OUT_TRAIN_DIR/entity_embedding.vec"

for sp in "${splits[@]}"; do
  cp "$OUT_TRAIN_DIR/entity_embedding.vec" "$DATA_ROOT/$sp/entity_embedding.vec"
done
log_success "Copied entity_embedding.vec to splits: ${splits[*]}"

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ PIPELINE COMPLETE: Stage 04 → 09 → 10 → 06 [TransE + MINDlarge]${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

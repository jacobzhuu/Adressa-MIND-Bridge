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
DEFAULT_ARTIFACTS_DIR="$(resolve_dir_default "outputs/artifacts" "artifacts")"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$DEFAULT_ARTIFACTS_DIR}"

DEFAULT_MIND_VEC_TRAIN="$(resolve_file_default "data/mind/MINDlarge/train/entity_embedding.vec" "MINDlarge/train/entity_embedding.vec")"
DEFAULT_MIND_VEC_VAL="$(resolve_file_default "data/mind/MINDlarge/val/entity_embedding.vec" "MINDlarge/val/entity_embedding.vec")"
DEFAULT_MIND_VEC_TEST="$(resolve_file_default "data/mind/MINDlarge/test/entity_embedding.vec" "MINDlarge/test/entity_embedding.vec")"

DEFAULT_MIND_REL_VEC="$(resolve_file_default "data/mind/MINDlarge/train/relation_embedding.vec" "MINDlarge/train/relation_embedding.vec")"
MIND_VEC_TRAIN="${MIND_VEC_TRAIN:-$DEFAULT_MIND_VEC_TRAIN}"
MIND_VEC_VAL="${MIND_VEC_VAL:-$DEFAULT_MIND_VEC_VAL}"
MIND_VEC_TEST="${MIND_VEC_TEST:-$DEFAULT_MIND_VEC_TEST}"
MIND_REL_VEC="${MIND_REL_VEC:-$DEFAULT_MIND_REL_VEC}"

OUT_INIT_DIR="${OUT_INIT_DIR:-$ARTIFACTS_DIR/entities_mindlarge}"

# Match the default KG settings used by run_05_06_transe_mindlarge.sh
WIKIDATA_KEEP_NEIGHBORS="${WIKIDATA_KEEP_NEIGHBORS:-1}"
WIKIDATA_MAX_TRIPLES_PER_ENTITY="${WIKIDATA_MAX_TRIPLES_PER_ENTITY:-200}"
DEFAULT_OUT_KG_DIR="$ARTIFACTS_DIR/wikidata_subgraph_nei${WIKIDATA_KEEP_NEIGHBORS}_k${WIKIDATA_MAX_TRIPLES_PER_ENTITY}"
OUT_KG_DIR="${OUT_KG_DIR:-$DEFAULT_OUT_KG_DIR}"

OUT_HOLDOUT_DIR="${OUT_HOLDOUT_DIR:-$ARTIFACTS_DIR/wikidata_subgraph_holdout}"
OUT_TRAIN_DIR="${OUT_TRAIN_DIR:-$ARTIFACTS_DIR/transe_train_holdout}"

TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"

TRANSE_INIT_FROM_ANCHORS="${TRANSE_INIT_FROM_ANCHORS:-1}"   # 1 recommended
TRANSE_REL_WEIGHTING="${TRANSE_REL_WEIGHTING:-sqrt_inv}"    # none|sqrt_inv
transe_init_flag="--no-init_from_anchors"
if [[ "$TRANSE_INIT_FROM_ANCHORS" == "1" ]]; then
  transe_init_flag="--init_from_anchors"
fi

log_header "PIPELINE: KG holdout (split → train → eval) [TransE + MINDlarge]"
log_info "artifacts:   $ARTIFACTS_DIR"
log_info "init_dir:    $OUT_INIT_DIR"
log_info "kg_dir:      $OUT_KG_DIR"
log_info "holdout_dir: $OUT_HOLDOUT_DIR"
log_info "train_out:   $OUT_TRAIN_DIR"
log_info "mind_vecs:   $MIND_VEC_TRAIN, $MIND_VEC_VAL, $MIND_VEC_TEST"
log_info "mind_rel:    $MIND_REL_VEC"
log_info "device:      $TRAIN_DEVICE"
log_info "transe:      init_from_anchors=$TRANSE_INIT_FROM_ANCHORS relation_weighting=$TRANSE_REL_WEIGHTING"

if [[ ! -f "$OUT_INIT_DIR/entity_vocab.txt" ]]; then
  echo -e "${RED}ERROR:${NC} Missing $OUT_INIT_DIR/entity_vocab.txt (run Step 04 first)"
  exit 1
fi
if [[ ! -f "$OUT_INIT_DIR/entity_init.npy" || ! -f "$OUT_INIT_DIR/entity_init_mask.npy" ]]; then
  echo -e "${RED}ERROR:${NC} Missing entity_init.npy or entity_init_mask.npy under $OUT_INIT_DIR"
  exit 1
fi
if [[ ! -f "$OUT_KG_DIR/kg_triples.txt" ]]; then
  echo -e "${RED}ERROR:${NC} Missing $OUT_KG_DIR/kg_triples.txt (run Step 09 first)"
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step A: Split KG (forward holdout, no inverse leakage)
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step A: Split KG into train/val/test"

python scripts/transe/11_eval_transe_link_prediction.py split \
  --kg_triples "$OUT_KG_DIR/kg_triples.txt" \
  --seed_entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --mind_relation_vec "$MIND_REL_VEC" \
  --output_dir "$OUT_HOLDOUT_DIR" \
  --split 0.8 0.1 0.1 \
  --seed 42 \
  --overwrite 2>&1 | awk 'NF'

log_success "Wrote holdout splits"

# ═══════════════════════════════════════════════════════════════════════════════
# Step B: Train on holdout-train
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step B: Train TransE on holdout-train"

python scripts/transe/10_train_transe.py \
  --kg_triples "$OUT_HOLDOUT_DIR/train.txt" \
  --seed_entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --entity_init "$OUT_INIT_DIR/entity_init.npy" \
  --entity_init_mask "$OUT_INIT_DIR/entity_init_mask.npy" \
  --mind_entity_vec "$MIND_VEC_TRAIN" "$MIND_VEC_VAL" "$MIND_VEC_TEST" \
  --mind_relation_vec "$MIND_REL_VEC" \
  --output_dir "$OUT_TRAIN_DIR" \
  $transe_init_flag \
  --relation_weighting "$TRANSE_REL_WEIGHTING" \
  --device "$TRAIN_DEVICE" 2>&1 | awk 'NF'

log_success "Trained holdout model"

# ═══════════════════════════════════════════════════════════════════════════════
# Step C: Evaluate (init vs trained)
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step C: Eval link prediction (filtered MRR/Hits@K)"

python scripts/transe/11_eval_transe_link_prediction.py eval \
  --split_dir "$OUT_HOLDOUT_DIR" \
  --seed_entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --mind_relation_vec "$MIND_REL_VEC" \
  --entity_matrix init "$OUT_INIT_DIR/entity_init.npy" \
  --entity_matrix trained "$OUT_TRAIN_DIR/entity_trained.npy" \
  --mode both \
  --ks 1 3 10 2>&1 | awk 'NF'

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ HOLDOUT EVAL COMPLETE: see $OUT_HOLDOUT_DIR/metrics.json${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

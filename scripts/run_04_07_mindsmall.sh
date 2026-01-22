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

# 输出函数
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

log_result() {
  echo -e "  ${GREEN}│${NC} $1"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════
DATA_ROOT="${DATA_ROOT:-adressa_one_week_mind_final}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
CACHE_DIR="${CACHE_DIR:-cache}"

MIND_VEC="${MIND_VEC:-MINDsmall/train/entity_embedding.vec}"
OUT_INIT_DIR="${OUT_INIT_DIR:-$ARTIFACTS_DIR/entities_mindsmall}"
OUT_TRAIN_DIR="${OUT_TRAIN_DIR:-$ARTIFACTS_DIR/no_entity_train_mindsmall}"

TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"

export HF_HOME="${HF_HOME:-$PWD/$CACHE_DIR/huggingface}"
unset TRANSFORMERS_CACHE || true

mkdir -p "$ARTIFACTS_DIR" "$CACHE_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 开始
# ═══════════════════════════════════════════════════════════════════════════════
log_header "PIPELINE: Stage 04-07 [MINDsmall]"
log_info "data_root: $DATA_ROOT"
log_info "mind_vec: $MIND_VEC"

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
if [[ ! -f "$DATA_ROOT/train/news.tsv" ]]; then
  echo -e "${RED}ERROR:${NC} Missing $DATA_ROOT/train/news.tsv (training requires train split)"
  exit 1
fi
log_info "splits: ${splits[*]}"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 04: Build Entity Vocab
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 04: Build Entity Vocab & Init"

news_tsv_args=()
for sp in "${splits[@]}"; do
  news_tsv_args+=("$DATA_ROOT/$sp/news.tsv")
done

python scripts/04_build_entity_vocab_and_init.py \
  --news_tsv \
    "${news_tsv_args[@]}" \
  --mind_entity_vec "$MIND_VEC" \
  --output_dir "$OUT_INIT_DIR" 2>&1 | awk 'NF'

python - <<PY
import json
from pathlib import Path

stats=json.loads(Path("$OUT_INIT_DIR/entity_init_stats.json").read_text(encoding="utf-8"))
print(f"\033[0;32m✓\033[0m Entity Coverage Report:")
print(f"  │ Entities:    {stats['num_entities']:>6}")
print(f"  │ MIND Hits:   {stats['mind_hits']:>6}")
print(f"  │ MIND Misses: {stats['mind_misses']:>6}")
print(f"  │ Coverage:    {stats['coverage']*100:>6.2f}%")
print(f"  │ Dimension:   {stats['dim']:>6}")
PY

# ═══════════════════════════════════════════════════════════════════════════════
# Step 05: Train Entity Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 05: Train Entity Embeddings"
log_info "device: $TRAIN_DEVICE"

python scripts/05_train_entity_embeddings_no.py \
  --news_tsv "$DATA_ROOT/train/news.tsv" \
  --entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --entity_init "$OUT_INIT_DIR/entity_init.npy" \
  --output_dir "$OUT_TRAIN_DIR" \
  --device "$TRAIN_DEVICE" \
  --reuse_precomputed 2>&1 | awk '!/(UserWarning|gradient_checkpointing|warnings\\.warn)/ && NF'

python - <<PY
import numpy as np

mask=np.load("$OUT_INIT_DIR/entity_init_mask.npy")
y=np.load("$OUT_TRAIN_DIR/nbbert.qid_idx.npy")
if y.size == 0:
    hits = 0
    invalid = 0
else:
    y_valid = y[(y >= 0) & (y < len(mask))]
    invalid = int(y.size - y_valid.size)
    hits = int(mask[y_valid].sum())
print(f"\033[0;32m✓\033[0m Mention Coverage Report:")
print(f"  │ Total Mentions:    {len(y):>6}")
print(f"  │ Pretrained Hits:   {hits:>6}")
if invalid:
    print(f"  │ Invalid Mentions:  {invalid:>6}  (stale cache; rerun without --reuse_precomputed)")
print(f"  │ Weighted Coverage: {hits/len(y)*100:>6.2f}%")
PY

# ═══════════════════════════════════════════════════════════════════════════════
# Step 06: Export Entity Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 06: Export Entity Embeddings"

python scripts/06_export_entity_embedding_vec.py \
  --entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
  --entity_matrix "$OUT_TRAIN_DIR/entity_trained.npy" \
  --output_vec "$OUT_TRAIN_DIR/entity_embedding.vec"

for sp in "${splits[@]}"; do
  cp "$OUT_TRAIN_DIR/entity_embedding.vec" "$DATA_ROOT/$sp/entity_embedding.vec"
done
log_success "Copied entity_embedding.vec to splits: ${splits[*]}"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 07: Evaluate Retrieval
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 07: Evaluate Entity Retrieval"
log_info "device: $EVAL_DEVICE"

eval_split() {
  local split="$1"
  log_step "Evaluating $split split..."
  
  local out
  out="$(
    python scripts/07_eval_entity_embedding_retrieval.py \
      --news_tsv "$DATA_ROOT/$split/news.tsv" \
      --train_news_tsv "$DATA_ROOT/train/news.tsv" \
      --entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
      --entity_matrix "$OUT_TRAIN_DIR/entity_trained.npy" \
      --projection "$OUT_TRAIN_DIR/projection.pt" \
      --device "$EVAL_DEVICE" 2>&1 | awk '!/(UserWarning|gradient_checkpointing|warnings\\.warn)/ && NF'
  )"

  get_val() {
    local key="$1"
    echo "$out" | awk -F= -v k="$key" '$1==k {print $2; exit}'
  }

  print_block() {
    local label="$1"
    local prefix="$2"

    local mentions ue ev r1 r5 r10 mrr
    mentions="$(get_val "${prefix}mentions")"
    ue="$(get_val "${prefix}unique_entities_in_eval")"
    ev="$(get_val "${prefix}entity_vocab")"
    r1="$(get_val "${prefix}recall@1")"
    r5="$(get_val "${prefix}recall@5")"
    r10="$(get_val "${prefix}recall@10")"
    mrr="$(get_val "${prefix}mrr")"

    echo -e "  ${GREEN}✓${NC} ${BOLD}$split${NC} Results (${label}):"
    echo -e "  │ Mentions:         ${mentions:-N/A}"
    echo -e "  │ Unique Entities:  ${ue:-N/A}"
    echo -e "  │ Entity Vocab:     ${ev:-N/A}"
    echo -e "  ├─────────────────────────────"
    echo -e "  │ ${BOLD}Recall@1:${NC}   ${CYAN}${r1:-N/A}${NC}"
    echo -e "  │ ${BOLD}Recall@5:${NC}   ${CYAN}${r5:-N/A}${NC}"
    echo -e "  │ ${BOLD}Recall@10:${NC}  ${CYAN}${r10:-N/A}${NC}"
    echo -e "  │ ${BOLD}MRR:${NC}        ${CYAN}${mrr:-N/A}${NC}"
    echo ""
  }

  print_block "FULL $split (all mentions)" "full_"
  print_block "SEEN $split (entity in train)" "seen_"
  print_block "UNSEEN $split (entity not in train)" "unseen_"
}

for sp in "${splits[@]}"; do
  if [[ "$sp" == "train" ]]; then
    continue
  fi
  eval_split "$sp"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ PIPELINE COMPLETE: Stage 04-07 [MINDsmall]${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

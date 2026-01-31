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

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_DATA_ROOT="$(resolve_dir_default "data/work/adressa_one_week_mind_final" "adressa_one_week_mind_final")"
DEFAULT_ARTIFACTS_DIR="$(resolve_dir_default "outputs/artifacts" "artifacts")"

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$DEFAULT_ARTIFACTS_DIR}"

OUT_INIT_DIR="${OUT_INIT_DIR:-$ARTIFACTS_DIR/entities_mindlarge}"
OUT_TRAIN_DIR="${OUT_TRAIN_DIR:-$ARTIFACTS_DIR/transe_train}"
OUT_EVAL_DIR="${OUT_EVAL_DIR:-$ARTIFACTS_DIR/reco_eval_entityavg_mindlarge}"

ENTITY_WEIGHT="${ENTITY_WEIGHT:-confidence}"   # uniform|confidence
SCORE_FN="${SCORE_FN:-cosine}"                 # cosine|dot
TIE_MODE="${TIE_MODE:-expected}"               # expected|optimistic|pessimistic
MAX_IMPRESSIONS="${MAX_IMPRESSIONS:-}"         # optional
SPLITS="${SPLITS:-}"                           # optional: e.g. "test" or "train test"

log_header "PIPELINE: Downstream reco baseline (entity-avg) [AUC/MRR]"
log_info "data_root:  $DATA_ROOT"
log_info "artifacts:  $ARTIFACTS_DIR"
log_info "init_dir:   $OUT_INIT_DIR"
log_info "train_dir:  $OUT_TRAIN_DIR"
log_info "out_dir:    $OUT_EVAL_DIR"
log_info "score:      $SCORE_FN (tie=$TIE_MODE, weight=$ENTITY_WEIGHT)"

if [[ ! -f "$OUT_INIT_DIR/entity_vocab.txt" ]]; then
  echo -e "${RED}ERROR:${NC} Missing $OUT_INIT_DIR/entity_vocab.txt"
  exit 1
fi
if [[ ! -f "$OUT_INIT_DIR/entity_init.npy" ]]; then
  echo -e "${RED}ERROR:${NC} Missing $OUT_INIT_DIR/entity_init.npy"
  exit 1
fi
if [[ ! -f "$OUT_TRAIN_DIR/entity_trained.npy" ]]; then
  echo -e "${RED}ERROR:${NC} Missing $OUT_TRAIN_DIR/entity_trained.npy"
  exit 1
fi

news_args=()
splits=()
for sp in train val test; do
  if [[ -f "$DATA_ROOT/$sp/news.tsv" ]]; then
    news_args+=("$DATA_ROOT/$sp/news.tsv")
    if [[ -f "$DATA_ROOT/$sp/behaviors.tsv" ]]; then
      splits+=("$sp")
    fi
  fi
done

if [[ ${#splits[@]} -eq 0 ]]; then
  echo -e "${RED}ERROR:${NC} No splits with both news.tsv and behaviors.tsv under $DATA_ROOT"
  exit 1
fi

if [[ -n "$SPLITS" ]]; then
  want=()
  for sp in $SPLITS; do
    want+=("$sp")
  done
  filtered=()
  for sp in "${splits[@]}"; do
    for w in "${want[@]}"; do
      if [[ "$sp" == "$w" ]]; then
        filtered+=("$sp")
      fi
    done
  done
  splits=("${filtered[@]}")
fi

if [[ ${#splits[@]} -eq 0 ]]; then
  echo -e "${RED}ERROR:${NC} No requested splits found (SPLITS=$SPLITS)"
  exit 1
fi
log_info "splits:     ${splits[*]}"

mkdir -p "$OUT_EVAL_DIR"

extra_flags=()
if [[ -n "$MAX_IMPRESSIONS" ]]; then
  extra_flags+=(--max_impressions "$MAX_IMPRESSIONS")
fi

for sp in "${splits[@]}"; do
  log_header "Eval split: $sp"
  out_json="$OUT_EVAL_DIR/$sp.metrics.json"
  python scripts/transe/12_eval_reco_entityavg.py \
    --news_tsv "${news_args[@]}" \
    --behaviors_tsv "$DATA_ROOT/$sp/behaviors.tsv" \
    --entity_vocab "$OUT_INIT_DIR/entity_vocab.txt" \
    --entity_matrix init "$OUT_INIT_DIR/entity_init.npy" \
    --entity_matrix transe "$OUT_TRAIN_DIR/entity_trained.npy" \
    --entity_weight "$ENTITY_WEIGHT" \
    --score "$SCORE_FN" \
    --tie_mode "$TIE_MODE" \
    --output_json "$out_json" \
    ${extra_flags[@]+"${extra_flags[@]}"} 2>&1 | awk 'NF'
  log_success "Wrote $out_json"
done

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ RECO EVAL COMPLETE: see $OUT_EVAL_DIR/*.metrics.json${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

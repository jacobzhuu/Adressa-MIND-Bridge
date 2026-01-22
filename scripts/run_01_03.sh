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
MAGENTA='\033[0;35m'
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

log_config() {
  echo -e "  ${MAGENTA}•${NC} $1"
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
DEFAULT_CACHE_DIR="$(resolve_dir_default "outputs/cache" "cache")"

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$DEFAULT_ARTIFACTS_DIR}"
CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE_DIR}"

export DATA_ROOT

NER_MODEL="${NER_MODEL:-NbAiLab/nb-bert-base-ner}"
NER_BATCH_SIZE="${NER_BATCH_SIZE:-32}"
NER_MAX_LENGTH="${NER_MAX_LENGTH:-128}"
NER_AGGREGATION_STRATEGY="${NER_AGGREGATION_STRATEGY:-simple}"
NER_DEVICE="${NER_DEVICE:--1}" # -1 CPU
NER_HEURISTIC_MODE="${NER_HEURISTIC_MODE:-fallback}"  # off|fallback|merge
NER_HEURISTIC_MAX_MENTIONS="${NER_HEURISTIC_MAX_MENTIONS:-4}"
NER_HEURISTIC_SCORE="${NER_HEURISTIC_SCORE:-0.35}"
NER_HEURISTIC_MAX_SPAN_CHARS="${NER_HEURISTIC_MAX_SPAN_CHARS:-60}"
NER_HEURISTIC_MAX_SPAN_TOKENS="${NER_HEURISTIC_MAX_SPAN_TOKENS:-6}"
NER_HEURISTIC_MIN_FIRST_TOKEN_LEN="${NER_HEURISTIC_MIN_FIRST_TOKEN_LEN:-0}"
NER_MAX_MENTIONS_PER_TITLE="${NER_MAX_MENTIONS_PER_TITLE:-10}"

WIKIDATA_LANG="${WIKIDATA_LANG:-nb}"
WIKIDATA_CACHE_DB="${WIKIDATA_CACHE_DB:-$CACHE_DIR/wikidata_search.sqlite}"
WIKIDATA_SLEEP="${WIKIDATA_SLEEP:-0.2}"
WIKIDATA_MAX_RETRIES="${WIKIDATA_MAX_RETRIES:-12}"
WIKIDATA_RETRY_BASE_SLEEP="${WIKIDATA_RETRY_BASE_SLEEP:-2}"
WIKIDATA_RETRY_MAX_SLEEP="${WIKIDATA_RETRY_MAX_SLEEP:-120}"
WIKIDATA_MAX_CONSEC_ERRORS="${WIKIDATA_MAX_CONSEC_ERRORS:-1000}"
WIKIDATA_MIN_MATCH="${WIKIDATA_MIN_MATCH:-0.6}"
WIKIDATA_MIN_MATCH_HEUR="${WIKIDATA_MIN_MATCH_HEUR:-0.92}"
WIKIDATA_TRUST_ENV="${WIKIDATA_TRUST_ENV:-0}" # 0 -> --no-trust-env (recommended if you have system proxy issues)

LIMIT="${LIMIT:-}"

export HF_HOME="${HF_HOME:-$PWD/$CACHE_DIR/huggingface}"
unset TRANSFORMERS_CACHE || true

mkdir -p "$ARTIFACTS_DIR" "$CACHE_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 开始
# ═══════════════════════════════════════════════════════════════════════════════
log_header "PIPELINE: Stage 01-03 [NER + Entity Linking]"
echo -e "  ${YELLOW}Configuration:${NC}"
log_config "data_root:     $DATA_ROOT"
log_config "artifacts:     $ARTIFACTS_DIR"
log_config "cache:         $CACHE_DIR"
echo ""
echo -e "  ${YELLOW}NER Settings:${NC}"
log_config "model:         $NER_MODEL"
log_config "device:        $NER_DEVICE"
log_config "batch_size:    $NER_BATCH_SIZE"
log_config "max_length:    $NER_MAX_LENGTH"
log_config "aggregation:   $NER_AGGREGATION_STRATEGY"
log_config "heuristic:     $NER_HEURISTIC_MODE (max=$NER_HEURISTIC_MAX_MENTIONS score=$NER_HEURISTIC_SCORE)"
log_config "heur_span:     max_chars=$NER_HEURISTIC_MAX_SPAN_CHARS max_tokens=$NER_HEURISTIC_MAX_SPAN_TOKENS"
log_config "heur_first:    min_first_token_len=$NER_HEURISTIC_MIN_FIRST_TOKEN_LEN"
log_config "max_mentions:  $NER_MAX_MENTIONS_PER_TITLE"
echo ""
echo -e "  ${YELLOW}Wikidata Settings:${NC}"
log_config "lang:          $WIKIDATA_LANG"
log_config "sleep:         $WIKIDATA_SLEEP"
log_config "max_retries:   $WIKIDATA_MAX_RETRIES"
log_config "min_match:     $WIKIDATA_MIN_MATCH"
log_config "min_match_heur:$WIKIDATA_MIN_MATCH_HEUR"
log_config "trust_env:     $WIKIDATA_TRUST_ENV"
log_config "cache_db:      $WIKIDATA_CACHE_DB"

trust_env_flag="--no-trust-env"
if [[ "$WIKIDATA_TRUST_ENV" == "1" ]]; then
  trust_env_flag="--trust-env"
fi

limit_flag=()
if [[ -n "$LIMIT" ]]; then
  limit_flag=(--limit "$LIMIT")
  echo ""
  echo -e "  ${RED}⚠${NC} Debug mode: LIMIT=$LIMIT"
fi

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
log_config "splits:        ${splits[*]}"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 01: NER Extraction
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 01: Named Entity Recognition (NER)"

for split in "${splits[@]}"; do
  news_tsv="$DATA_ROOT/$split/news.tsv"
  mentions_jsonl="$ARTIFACTS_DIR/$split.mentions.jsonl"

  log_step "Processing $split split..."
  log_info "input:  $news_tsv"
  log_info "output: $mentions_jsonl"
  
		  python scripts/01_ner_titles_nbbert.py \
		    --news_tsv "$news_tsv" \
		    --output_jsonl "$mentions_jsonl" \
		    --model "$NER_MODEL" \
		    --batch_size "$NER_BATCH_SIZE" \
		    --max_length "$NER_MAX_LENGTH" \
		    --aggregation_strategy "$NER_AGGREGATION_STRATEGY" \
		    --device "$NER_DEVICE" \
		    --heuristic_mode "$NER_HEURISTIC_MODE" \
		    --heuristic_max_mentions "$NER_HEURISTIC_MAX_MENTIONS" \
		    --heuristic_score "$NER_HEURISTIC_SCORE" \
		    --heuristic_max_span_chars "$NER_HEURISTIC_MAX_SPAN_CHARS" \
		    --heuristic_max_span_tokens "$NER_HEURISTIC_MAX_SPAN_TOKENS" \
		    --heuristic_min_first_token_len "$NER_HEURISTIC_MIN_FIRST_TOKEN_LEN" \
		    --max_mentions_per_title "$NER_MAX_MENTIONS_PER_TITLE" \
		    ${limit_flag[@]+"${limit_flag[@]}"} 2>&1 | awk 'NF'
  
  log_success "Completed $split NER extraction"
  echo ""
done

# ═══════════════════════════════════════════════════════════════════════════════
# Step 02: Entity Linking to Wikidata
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 02: Entity Linking to Wikidata"

for split in "${splits[@]}"; do
  mentions_jsonl="$ARTIFACTS_DIR/$split.mentions.jsonl"
  linked_jsonl="$ARTIFACTS_DIR/$split.linked.jsonl"

  log_step "Linking $split split..."
  log_info "input:  $mentions_jsonl"
  log_info "output: $linked_jsonl"
  
	  python scripts/02_link_to_wikidata.py \
	    --mentions_jsonl "$mentions_jsonl" \
	    --output_jsonl "$linked_jsonl" \
	    --cache_db "$WIKIDATA_CACHE_DB" \
	    --lang "$WIKIDATA_LANG" \
	    --sleep "$WIKIDATA_SLEEP" \
	    --max-retries "$WIKIDATA_MAX_RETRIES" \
	    --retry-base-sleep "$WIKIDATA_RETRY_BASE_SLEEP" \
	    --retry-max-sleep "$WIKIDATA_RETRY_MAX_SLEEP" \
	    --max-consecutive-errors "$WIKIDATA_MAX_CONSEC_ERRORS" \
	    --min_match "$WIKIDATA_MIN_MATCH" \
	    --min_match_heur "$WIKIDATA_MIN_MATCH_HEUR" \
	    $trust_env_flag \
	    --overwrite \
	    ${limit_flag[@]+"${limit_flag[@]}"} 2>&1 | awk 'NF'
  
  log_success "Completed $split entity linking"
  echo ""
done

# ═══════════════════════════════════════════════════════════════════════════════
# Step 03: Write Entities to TSV
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Step 03: Write Title Entities to TSV"

timestamp="$(date +%Y%m%d_%H%M%S)"
for split in "${splits[@]}"; do
  news_tsv="$DATA_ROOT/$split/news.tsv"
  linked_jsonl="$ARTIFACTS_DIR/$split.linked.jsonl"
  out_tsv="$DATA_ROOT/$split/news.with_entities.tsv"

  log_step "Writing $split split..."
  log_info "news_tsv:    $news_tsv"
  log_info "linked:      $linked_jsonl"
  log_info "output:      $out_tsv"
  
  python scripts/03_write_title_entities_to_tsv.py \
    --news_tsv "$news_tsv" \
    --linked_jsonl "$linked_jsonl" \
    --output_tsv "$out_tsv" 2>&1 | awk 'NF'

  backup="$news_tsv.bak.$timestamp"
  cp "$news_tsv" "$backup"
  mv "$out_tsv" "$news_tsv"
  log_success "Replaced news.tsv (backup: $backup)"
  echo ""
done

# ═══════════════════════════════════════════════════════════════════════════════
# Statistics Summary
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Statistics Summary"

python - <<'PY'
import json
import re
from pathlib import Path

import pandas as pd

qid_re = re.compile(r"^Q\d+$")

import os

data_root = Path(os.environ.get("DATA_ROOT", "data/work/adressa_one_week_mind_final"))
legacy_root = Path("adressa_one_week_mind_final")
if not data_root.exists() and legacy_root.exists():
    data_root = legacy_root
for sp in ["train", "val", "test"]:
    p = data_root / sp / "news.tsv"
    if not p.exists():
        continue
    df = pd.read_csv(p, sep="\t", header=None)
    te = df.iloc[:, 6].astype(str)
    te_non_empty = te[te != "[]"]
    non_empty = int((te != "[]").sum())
    head_ok = 0
    tail_ok = 0
    sample = None

    def count_ok(series):
        ok = 0
        sample = None
        for s in series:
            try:
                arr = json.loads(s)
                if arr and isinstance(arr, list):
                    q = arr[0].get("WikidataId")
                    if q and qid_re.match(str(q)):
                        ok += 1
                        if sample is None:
                            sample = arr[0]
            except Exception:
                continue
        return ok, sample

    head_ok, head_sample = count_ok(te_non_empty.head(2000))
    tail_ok, tail_sample = count_ok(te_non_empty.tail(2000))
    sample = head_sample or tail_sample

    ratio = non_empty / len(df) if len(df) else 0.0
    print(f"\033[0;32m✓\033[0m \033[1m{sp}\033[0m Split:")
    print(f"  │ Total Rows:              {len(df):>6}")
    print(f"  │ Non-empty Title Entities: {non_empty:>6}")
    print(f"  │ Coverage Ratio:           {ratio*100:>6.2f}%")
    print(f"  │ Valid QID in Head(2k):    {head_ok:>6}")
    print(f"  │ Valid QID in Tail(2k):    {tail_ok:>6}")
    if sample:
        print(f"  │ Sample Entity:            {sample.get('WikidataId', 'N/A')} ({sample.get('Label', 'N/A')})")
    print()
PY

# ═══════════════════════════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ PIPELINE COMPLETE: Stage 01-03 [NER + Entity Linking]${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

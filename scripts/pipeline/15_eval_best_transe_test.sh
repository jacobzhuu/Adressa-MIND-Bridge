#!/usr/bin/env bash
set -euo pipefail

# Best model from sota loop (nei=1, k=200 had best val AUC=0.5449)
DEFAULT_MODEL="outputs/artifacts/sota_transe_reco/20260127_124952/promoted/nei1_k200/entity_trained.npy"
MODEL="${1:-$DEFAULT_MODEL}"

if [[ ! -f "$MODEL" ]]; then
    # Try alternate location (sometimes promote might fail or be different)
    ALT_MODEL="outputs/artifacts/sota_transe_reco/20260127_124952/promoted/nei1_k200/entity_trained.npy"
    if [[ -f "$ALT_MODEL" ]]; then
        MODEL="$ALT_MODEL"
    else
        echo "Error: Model file not found at $MODEL"
        exit 1
    fi
fi

DATA_ROOT="data/work/adressa_one_week_mind_final"
INIT_DIR="outputs/artifacts/sota_transe_reco/20260127_124952/init_entities_mindlarge"
ENTITY_VOCAB="$INIT_DIR/entity_vocab.txt"
ENTITY_INIT="$INIT_DIR/entity_init.npy"

if [[ ! -f "$ENTITY_VOCAB" ]]; then
    echo "Error: Entity vocab not found at $ENTITY_VOCAB"
    exit 1
fi

echo "Evaluating Model: $MODEL"
echo "Entity Vocab:   $ENTITY_VOCAB"
echo "Data Root:      $DATA_ROOT"
echo "Output:         outputs/eval_reco_best_transe_test.json"

python scripts/transe/12_eval_reco_entityavg.py \
  --news_tsv \
    "$DATA_ROOT/train/news.tsv" \
    "$DATA_ROOT/test/news.tsv" \
  --behaviors_tsv "$DATA_ROOT/test/behaviors.tsv" \
  --entity_vocab "$ENTITY_VOCAB" \
  --entity_matrix "init" "$ENTITY_INIT" \
  --entity_matrix "transe" "$MODEL" \
  --entity_weight confidence \
  --score cosine \
  --tie_mode expected \
  --output_json "outputs/eval_reco_best_transe_test.json"

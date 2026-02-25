#!/bin/bash
# Submit all Bible training jobs for all 6 tokenizers
# 
# Usage:
#   cd /home/parsaidp/nanochat
#   export NANOCHAT_BASE_DIR="$PWD/bible_run"
#   bash scripts/submit_all_bible_train.sh

set -e

# Set base directory if not already set
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/large_storage/goodarzilab/parsaidp/nanochat/bible_run}"
echo "Using NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo ""

# Array of tokenizer configurations
declare -a TOKENIZERS=(
    "tokenizer-256"
    "tokenizer-512"
    "tokenizer-1k"
    "tokenizer-2k"
    "tokenizer-4k"
    "tokenizer-8k"
)

echo "Submitting training jobs for all 6 tokenizers..."
echo ""

for TOKENIZER_NAME in "${TOKENIZERS[@]}"; do
    TOK_DIR="$NANOCHAT_BASE_DIR/$TOKENIZER_NAME"
    if [ ! -f "$TOK_DIR/tokenizer.pkl" ]; then
        echo "WARNING: Tokenizer $TOKENIZER_NAME not found at $TOK_DIR"
        echo "  Train it first using: TOKENIZER_DIR=$TOKENIZER_NAME VOCAB_SIZE=<size> sbatch scripts/bible_tok_train.slurm"
        continue
    fi
    
    echo "Submitting training job for: $TOKENIZER_NAME"
    sbatch \
        --job-name="bible_${TOKENIZER_NAME}" \
        --export=ALL,NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",TOKENIZER_NAME="$TOKENIZER_NAME" \
        scripts/bible_train.slurm
    sleep 1  # Small delay between submissions
done

echo ""
echo "All training jobs submitted!"

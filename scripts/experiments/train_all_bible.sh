#!/bin/bash
# Complete pipeline: Train all tokenizers, then train all models
# 
# Usage:
#   cd /home/parsaidp/nanochat
#   export NANOCHAT_BASE_DIR="$PWD/bible_run"
#   bash scripts/train_all_bible.sh

set -e

# Set base directory if not already set
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/bible_run}"
echo "Using NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo ""

# Array of tokenizer configurations: (TOKENIZER_DIR VOCAB_SIZE)
declare -a TOKENIZERS=(
    "tokenizer-256 256"
    "tokenizer-512 512"
    "tokenizer-1k 1024"
    "tokenizer-2k 2048"
    "tokenizer-4k 4096"
    "tokenizer-8k 8192"
)

echo "=========================================="
echo "Step 1: Training all tokenizers..."
echo "=========================================="
echo ""

for tok_config in "${TOKENIZERS[@]}"; do
    read -r TOK_DIR VOCAB_SIZE <<< "$tok_config"
    TOK_PATH="$NANOCHAT_BASE_DIR/$TOK_DIR"
    
    if [ -f "$TOK_PATH/tokenizer.pkl" ]; then
        echo "Tokenizer $TOK_DIR already exists, skipping..."
        continue
    fi
    
    echo "Submitting tokenizer training: $TOK_DIR (vocab_size=$VOCAB_SIZE)"
    sbatch \
        --job-name="bible_tok_${TOK_DIR}" \
        --export=ALL,NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",TOKENIZER_DIR="$TOK_DIR",VOCAB_SIZE="$VOCAB_SIZE" \
        scripts/bible_tok_train.slurm
    sleep 1
done

echo ""
echo "Tokenizer training jobs submitted!"
echo ""
echo "=========================================="
echo "Step 2: Submit training jobs (wait for tokenizers first)"
echo "=========================================="
echo ""
echo "After tokenizers are trained, run:"
echo "  bash scripts/submit_all_bible_train.sh"
echo ""
echo "Or manually submit each:"
for tok_config in "${TOKENIZERS[@]}"; do
    read -r TOK_DIR VOCAB_SIZE <<< "$tok_config"
    echo "  TOKENIZER_NAME=$TOK_DIR sbatch scripts/bible_train.slurm"
done

#!/bin/bash
# Helper script to submit multiple Bible training experiments with different tokenizers
# 
# Usage:
#   cd /home/parsaidp/nanochat
#   export NANOCHAT_BASE_DIR="$PWD/bible_run"
#   bash scripts/submit_bible_experiments.sh

set -e

# Set base directory if not already set
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/bible_run}"
echo "Using NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"

# Array of tokenizer configurations: (TOKENIZER_DIR VOCAB_SIZE)
declare -a TOKENIZERS=(
    "tokenizer-256 256"
    "tokenizer-512 512"
    "tokenizer-1k 1024"
    "tokenizer-2k 2048"
    "tokenizer-4k 4096"
    "tokenizer-8k 8192"
)

echo "Step 1: Training tokenizers..."
for tok_config in "${TOKENIZERS[@]}"; do
    read -r TOK_DIR VOCAB_SIZE <<< "$tok_config"
    echo ""
    echo "Submitting tokenizer training job: $TOK_DIR (vocab_size=$VOCAB_SIZE)"
    sbatch \
        --job-name="bible_tok_${TOK_DIR}" \
        --export=ALL,NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",TOKENIZER_DIR="$TOK_DIR",VOCAB_SIZE="$VOCAB_SIZE" \
        scripts/bible_tok_train.slurm
    sleep 1  # Small delay between submissions
done

echo ""
echo "Tokenizer training jobs submitted!"
echo ""
echo "Step 2: After tokenizers are trained, submit model training jobs:"
echo ""

for tok_config in "${TOKENIZERS[@]}"; do
    read -r TOK_DIR VOCAB_SIZE <<< "$tok_config"
    echo "# Train with $TOK_DIR:"
    echo "sbatch --export=ALL,NANOCHAT_BASE_DIR=\"$NANOCHAT_BASE_DIR\",TOKENIZER_NAME=\"$TOK_DIR\" scripts/bible_train.slurm"
done

#!/bin/bash
# Rerun Bible training experiments with longer duration and verified tokenizers
# 
# Usage:
#   cd /home/parsaidp/nanochat
#   export NANOCHAT_BASE_DIR="$PWD/bible_run"
#   bash scripts/rerun_bible_training.sh

set -e

# Set base directory if not already set
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/bible_run}"
echo "Using NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"

# Verify tokenizers exist and have correct vocab sizes
echo "Verifying tokenizers..."
cd "$(dirname "$0")/.."
source ~/miniconda/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wave 2>/dev/null || true

python3 << 'PYTHON_VERIFY'
import os
from nanochat.tokenizer import get_tokenizer
import torch

base = os.environ.get('NANOCHAT_BASE_DIR', '/home/parsaidp/nanochat/bible_run')
tokenizers = ['tokenizer-512', 'tokenizer-1k', 'tokenizer-2k', 'tokenizer-4k', 'tokenizer-8k']
expected_vocabs = {'tokenizer-512': 512, 'tokenizer-1k': 1024, 'tokenizer-2k': 2048, 'tokenizer-4k': 4096, 'tokenizer-8k': 8192}

print("Checking tokenizers...")
all_good = True
for tok_name in tokenizers:
    tok_dir = os.path.join(base, tok_name)
    if not os.path.exists(os.path.join(tok_dir, 'tokenizer.pkl')):
        print(f"ERROR: {tok_name} tokenizer.pkl not found!")
        all_good = False
        continue
    if not os.path.exists(os.path.join(tok_dir, 'token_bytes.pt')):
        print(f"ERROR: {tok_name} token_bytes.pt not found!")
        all_good = False
        continue
    
    try:
        tok = get_tokenizer(tokenizer_path=tok_dir)
        vocab_size = tok.get_vocab_size()
        expected = expected_vocabs[tok_name]
        if vocab_size != expected:
            print(f"ERROR: {tok_name} vocab_size={vocab_size}, expected={expected}")
            all_good = False
        else:
            # Check token_bytes size
            token_bytes = torch.load(os.path.join(tok_dir, 'token_bytes.pt'))
            if token_bytes.shape[0] != vocab_size:
                print(f"ERROR: {tok_name} token_bytes shape={token_bytes.shape[0]}, vocab_size={vocab_size}")
                all_good = False
            else:
                print(f"OK: {tok_name} vocab_size={vocab_size}, token_bytes.shape={token_bytes.shape[0]}")
    except Exception as e:
        print(f"ERROR: {tok_name} failed to load: {e}")
        all_good = False

if not all_good:
    print("\nSome tokenizers have issues. Please fix them before running training.")
    exit(1)
else:
    print("\nAll tokenizers verified successfully!")
PYTHON_VERIFY

if [ $? -ne 0 ]; then
    echo "Tokenizer verification failed. Exiting."
    exit 1
fi

# Training configuration - make it longer
# Increase TARGET_PARAM_DATA_RATIO from 20 to 40 for longer training
# Or set NUM_ITERATIONS explicitly if you prefer
TARGET_PARAM_DATA_RATIO=${TARGET_PARAM_DATA_RATIO:-4000}  # Double the default (was 20)
NUM_ITERATIONS=${NUM_ITERATIONS:--1}  # -1 = use target_param_data_ratio

echo ""
echo "Training configuration:"
echo "  TARGET_PARAM_DATA_RATIO: $TARGET_PARAM_DATA_RATIO"
echo "  NUM_ITERATIONS: $NUM_ITERATIONS (will use TARGET_PARAM_DATA_RATIO if -1)"
echo ""

# Array of tokenizer names to train with
declare -a TOKENIZERS=(
    "tokenizer-512"
    "tokenizer-1k"
    "tokenizer-2k"
    "tokenizer-4k"
    "tokenizer-8k"
)

echo "Submitting training jobs for all tokenizers..."
echo ""

for TOK_NAME in "${TOKENIZERS[@]}"; do
    echo "Submitting training job for $TOK_NAME..."
    sbatch \
        --job-name="bible_train_${TOK_NAME}" \
        --export=ALL,NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR",TOKENIZER_NAME="$TOK_NAME",TARGET_PARAM_DATA_RATIO="$TARGET_PARAM_DATA_RATIO",NUM_ITERATIONS="$NUM_ITERATIONS" \
        scripts/bible_train.slurm
    sleep 1  # Small delay between submissions
done

echo ""
echo "All training jobs submitted!"
echo "Monitor with: squeue -u $USER"

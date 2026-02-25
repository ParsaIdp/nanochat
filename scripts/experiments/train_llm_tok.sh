#!/bin/bash
#SBATCH --job-name=llm_tok
#SBATCH --output=llm_tok_%j.out
#SBATCH --error=llm_tok_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --partition=preemptible

# Train a small LLM using a specified tokenizer directory.
# Choose tokenizer via TOKENIZER_NAME (relative to NANOCHAT_BASE_DIR).
# Examples:
#   TOKENIZER_NAME=tokenizer sbatch scripts/train_llm_tok.sh
#   TOKENIZER_NAME=tokenizer-big sbatch scripts/train_llm_tok.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/large_storage/goodarzilab/parsaidp/nanochat}"
# Fix memory fragmentation for large vocab models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
mkdir -p "$NANOCHAT_BASE_DIR"

# Resolve project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# Conda env
CONDA_INIT=""
for c in "$HOME/miniconda/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "$c" ] && CONDA_INIT="$c" && break
done
if [ -n "$CONDA_INIT" ]; then
  source "$CONDA_INIT"
else
  echo "conda not found"; exit 1
fi
conda activate wave

# Clear Python cache to avoid stale bytecode
find "$PROJECT_ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true

# Deps (best-effort editable install)
pip install -e . || true

# Config (small model, a few hundred M params; tuned for 1-2 H100 80GB)
DEPTH=${DEPTH:-12}                         # ~200M params with default vocab
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}  # per GPU
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-131072} # global tokens
NUM_GPUS=${NUM_GPUS:-2}
TOKENIZER_NAME=${TOKENIZER_NAME:-tokenizer}
RUN_NAME=${RUN_NAME:-llm_${TOKENIZER_NAME}}
# Allow much longer runs: default ~50x previous (was ~13k steps)
# 50 * 13,200 ≈ 660,000
NUM_ITERATIONS=${NUM_ITERATIONS:-660000}

# Check tokenizer exists
TOK_DIR="$NANOCHAT_BASE_DIR/$TOKENIZER_NAME"
if [ ! -f "$TOK_DIR/tokenizer.pkl" ]; then
  echo "Tokenizer not found at $TOK_DIR. Train it first (tok_train.sh)."
  exit 1
fi

echo "Training with tokenizer: $TOKENIZER_NAME at $TOK_DIR"

torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_train -- \
  --run="$RUN_NAME" \
  --depth=$DEPTH \
  --max_seq_len=$MAX_SEQ_LEN \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --total_batch_size=$TOTAL_BATCH_SIZE \
  --num_iterations=$NUM_ITERATIONS \
  --tokenizer_name=$TOKENIZER_NAME \
  --eval_every=500 \
  --core_metric_every=-1 \
  --save_every=-1

echo "Done. Run name: $RUN_NAME"


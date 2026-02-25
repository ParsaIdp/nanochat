#!/bin/bash
#SBATCH --job-name=llm_tok_big
#SBATCH --output=llm_tok_big_%j.out
#SBATCH --error=llm_tok_big_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --partition=gpu

# Train a small LLM using the big tokenizer (tokenizer-big by default).

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/large_storage/goodarzilab/parsaidp/nanochat}"
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

# Deps (best-effort editable install)
pip install -e . || true

# Config (small model, a few hundred M params)
DEPTH=${DEPTH:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-524288}
NUM_GPUS=${NUM_GPUS:-8}
TOKENIZER_NAME=${TOKENIZER_NAME:-tokenizer-big}
RUN_NAME=${RUN_NAME:-llm_${TOKENIZER_NAME}}

# Check tokenizer exists
TOK_DIR="$NANOCHAT_BASE_DIR/$TOKENIZER_NAME"
if [ ! -f "$TOK_DIR/tokenizer.pkl" ]; then
  echo "Tokenizer not found at $TOK_DIR. Train it first (tok_train.sh with TOKENIZER_DIR=tokenizer-big)."
  exit 1
fi

echo "Training with tokenizer: $TOKENIZER_NAME at $TOK_DIR"

torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_train -- \
  --run="$RUN_NAME" \
  --depth=$DEPTH \
  --max_seq_len=$MAX_SEQ_LEN \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --total_batch_size=$TOTAL_BATCH_SIZE \
  --tokenizer_name=$TOKENIZER_NAME \
  --eval_every=500 \
  --core_metric_every=2000 \
  --save_every=-1

echo "Done. Run name: $RUN_NAME"


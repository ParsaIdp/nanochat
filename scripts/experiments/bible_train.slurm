#!/bin/bash
#SBATCH --job-name=bible_train
#SBATCH --output=bible_train_%j.out
#SBATCH --error=bible_train_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --partition=preemptible

# Train a small LLM on the Bible dataset using a specified tokenizer.
# Specify tokenizer via TOKENIZER_NAME (relative to NANOCHAT_BASE_DIR).
# Examples:
#   TOKENIZER_NAME=tokenizer-2k sbatch scripts/bible_train.slurm
#   TOKENIZER_NAME=tokenizer-4k sbatch scripts/bible_train.slurm
#   TOKENIZER_NAME=tokenizer-8k sbatch scripts/bible_train.slurm

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/large_storage/goodarzilab/parsaidp/nanochat/bible_run}"
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

# Config - ~20M parameter model with max VRAM usage and 10x longer training
DEPTH=${DEPTH:-7}                           # ~20M params: depth=7 gives model_dim=448 (~24M params with vocab ~8K)
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-64}  # per GPU - increased to use all VRAM (was 16)
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-262144} # global tokens - 64 * 2048 * 2 GPUs = 262144
NUM_GPUS=${NUM_GPUS:-2}
TOKENIZER_NAME=${TOKENIZER_NAME:-tokenizer}
RUN_NAME=${RUN_NAME:-bible_${TOKENIZER_NAME}}
NUM_ITERATIONS=${NUM_ITERATIONS:--1}        # -1 = disable, use target_param_data_ratio instead
TARGET_PARAM_DATA_RATIO=${TARGET_PARAM_DATA_RATIO:-200}  # 10x longer: was 20, now 200 for many epochs

# Check tokenizer exists
TOK_DIR="$NANOCHAT_BASE_DIR/$TOKENIZER_NAME"
if [ ! -f "$TOK_DIR/tokenizer.pkl" ]; then
  echo "ERROR: Tokenizer not found at $TOK_DIR"
  echo "Train it first using:"
  echo "  TOKENIZER_DIR=$TOKENIZER_NAME VOCAB_SIZE=<size> sbatch scripts/bible_tok_train.slurm"
  exit 1
fi

echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "Training with tokenizer: $TOKENIZER_NAME at $TOK_DIR"
echo "Run name: $RUN_NAME"
echo "Depth: $DEPTH (~20M params)"
echo "Num GPUs: $NUM_GPUS"
echo "Device batch size: $DEVICE_BATCH_SIZE (max VRAM)"
echo "Total batch size: $TOTAL_BATCH_SIZE"
echo "Target param:data ratio: $TARGET_PARAM_DATA_RATIO (10x longer training)"
echo "Num iterations: $NUM_ITERATIONS (auto-calculated from ratio)"
echo ""

torchrun --standalone --nproc_per_node=$NUM_GPUS -m scripts.base_train -- \
  --run="$RUN_NAME" \
  --depth=$DEPTH \
  --max_seq_len=$MAX_SEQ_LEN \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --total_batch_size=$TOTAL_BATCH_SIZE \
  --num_iterations=$NUM_ITERATIONS \
  --target_param_data_ratio=$TARGET_PARAM_DATA_RATIO \
  --tokenizer_name=$TOKENIZER_NAME \
  --eval_every=200 \
  --core_metric_every=-1 \
  --save_every=-1 \
  --sample_every=200

echo "Done. Run name: $RUN_NAME"

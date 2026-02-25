#!/bin/bash
#SBATCH --partition=preemptible
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Usage: sbatch --export=RUN_NAME=...,TOK_DIR=...,EMB_MODE=...,DATA_DIR=... scripts/lz78_train.slurm

set -e
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

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
pip install -e . || true

echo "Run: $RUN_NAME"
echo "Tokenizer dir: $TOK_DIR"
echo "Embedding mode: $EMB_MODE"
echo "Data dir: $DATA_DIR"

torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --tokenizer_type=lz78 \
    --tokenizer_dir="$TOK_DIR" \
    --embedding_mode="$EMB_MODE" \
    --data_mode=pretokenized \
    --pretokenized_dir="$DATA_DIR" \
    --run="$RUN_NAME" \
    --depth=12 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --save_every=1000 \
    --core_metric_every=-1

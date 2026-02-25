#!/bin/bash
#SBATCH --partition=preemptible
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=trie2x-chunked-ext
#SBATCH --output=logs/trie2x-chunked-ext-full-%j.out

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
pip install -e . 2>/dev/null || true

# Verify we have 2 GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs available: $NUM_GPUS"
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "ERROR: Need 2 GPUs but only $NUM_GPUS available"
    exit 1
fi

BASE_DIR=/large_storage/goodarzilab/parsaidp/weezl/lz78_ablations
TOK_DIR=$BASE_DIR/tokenizers/trie2x_44k
DATA_DIR=$BASE_DIR/data/trie2x_44k_chunked

echo "Run: trie2x-44k-chunked-extended"
echo "Tokenizer dir: $TOK_DIR"
echo "Data dir: $DATA_DIR"
echo "Resuming from step 5846, training to step 10846 (+5000 steps)"

torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --tokenizer_type=lz78 \
    --tokenizer_dir="$TOK_DIR" \
    --embedding_mode=flat \
    --data_mode=pretokenized \
    --pretokenized_dir="$DATA_DIR" \
    --run=trie2x-44k-chunked-extended \
    --depth=12 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --num_iterations=10846 \
    --model_tag=trie2x-chunked-ext \
    --resume_from_step=5846 \
    --save_every=1000 \
    --core_metric_every=-1

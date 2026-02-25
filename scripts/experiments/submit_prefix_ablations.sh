#!/bin/bash
# Submit prefix label loss experiments on C4 using LZ78-32K tokenizer.
#
# These experiments compare different prefix label loss configurations
# against the standard CE baseline (lz78-32k-flat-c4-d12, already submitted).
#
# All use: LZ78 32K tokenizer, flat embedding, depth=12, C4 data.
# The pre-tokenized data from the main ablation run is reused.
#
# Usage: bash scripts/submit_prefix_ablations.sh

set -e

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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

BASE_DIR=/large_storage/goodarzilab/parsaidp/weezl/lz78_ablations
TOK_DIR=$BASE_DIR/tokenizers/lz78_32k
DATA_DIR=$BASE_DIR/data/lz78_32k

mkdir -p logs

# Verify data exists (should have been created by the main ablation run)
if [ ! -d "$DATA_DIR/train" ]; then
    echo "ERROR: Pre-tokenized data not found at $DATA_DIR/train"
    echo "Run submit_lz78_ablations.sh first and wait for pretokenization to complete."
    exit 1
fi

# Verify ancestor data exists
if [ ! -f "$TOK_DIR/token_ancestors.pt" ]; then
    echo "ERROR: Ancestor data not found at $TOK_DIR/token_ancestors.pt"
    echo "Re-run lz78_setup_tokenizer.py to generate it."
    exit 1
fi

echo "=== Submitting prefix label loss experiments ==="
echo "Tokenizer: $TOK_DIR"
echo "Data: $DATA_DIR"

# Helper function to submit a prefix training job
submit_prefix_job() {
    local RUN_NAME=$1
    local LOSS_MODE=$2
    local DECAY=$3
    local ALPHA=$4

    JOB=$(sbatch --parsable \
        --job-name=$RUN_NAME \
        --output=logs/${RUN_NAME}-%j.out \
        --partition=preemptible \
        --gres=gpu:2 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        --wrap="$(cat <<WRAP_EOF
set -e
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
CONDA_INIT=""
for c in "\$HOME/miniconda/etc/profile.d/conda.sh" "\$HOME/anaconda3/etc/profile.d/conda.sh" "\$HOME/miniconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "\$c" ] && CONDA_INIT="\$c" && break
done
[ -n "\$CONDA_INIT" ] && . "\$CONDA_INIT"
conda activate wave
cd $PROJECT_ROOT
pip install -e . 2>/dev/null || true
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --tokenizer_type=lz78 \
    --tokenizer_dir=$TOK_DIR \
    --embedding_mode=flat \
    --data_mode=pretokenized \
    --pretokenized_dir=$DATA_DIR \
    --loss_mode=$LOSS_MODE \
    --prefix_weight=$DECAY \
    --run=$RUN_NAME \
    --depth=12 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --core_metric_every=-1
WRAP_EOF
    )")
    echo "Submitted: $RUN_NAME (job $JOB)"
}

# Run 0: Standard CE baseline (same config, standard loss — for direct comparison)
submit_prefix_job "lz78-32k-standard-c4-d12" "standard" "0.5" "0.0"

# Run 1: Prefix loss, decay=0.5
submit_prefix_job "lz78-32k-prefix-d0.5-c4-d12" "prefix" "0.5" "0.0"

# Run 2: Prefix loss, decay=0.3 (more weight on exact match)
submit_prefix_job "lz78-32k-prefix-d0.3-c4-d12" "prefix" "0.3" "0.0"

# Run 3: Prefix loss, decay=0.7 (more weight on prefixes)
submit_prefix_job "lz78-32k-prefix-d0.7-c4-d12" "prefix" "0.7" "0.0"

# Run 4: Interpolated loss, alpha=0.2, decay=0.5
submit_prefix_job "lz78-32k-prefix-interp0.2-c4-d12" "prefix_interp" "0.5" "0.2"

echo ""
echo "=== All 4 prefix experiments submitted ==="
echo "All runs log to wandb project 'nanochat' (entity: goodarzilab)"
echo "Baseline comparison: lz78-32k-flat-c4-d12 (already submitted)"
echo "Monitor with: squeue -u \$USER"

#!/bin/bash
# Submit prefix label loss experiments using BPE tokenizer.
#
# These experiments test whether prefix-based losses (originally designed
# for LZ78's tree structure) also help with BPE, where prefix relationships
# are derived from byte-level prefix matching between BPE tokens.
#
# All use: default BPE tokenizer, flat embedding, depth=12, C4 data.
# Online tokenization (no pretokenization needed).
#
# Usage: bash scripts/submit_bpe_prefix_ablations.sh

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

# Default BPE tokenizer path
TOK_DIR=$(python -c "from nanochat.common import get_base_dir; import os; print(os.path.join(get_base_dir(), 'tokenizer'))")
echo "BPE tokenizer dir: $TOK_DIR"

# Generate ancestor data if not already present
if [ ! -f "$TOK_DIR/token_ancestors.pt" ]; then
    echo "Generating ancestor data for BPE tokenizer..."
    python -m scripts.bpe_generate_ancestors --tokenizer_dir "$TOK_DIR"
fi

# Verify ancestor data exists
if [ ! -f "$TOK_DIR/token_ancestors.pt" ]; then
    echo "ERROR: Failed to generate ancestor data"
    exit 1
fi

mkdir -p logs

echo "=== Submitting BPE prefix label loss experiments ==="
echo "Tokenizer: $TOK_DIR"

# Helper function to submit a BPE prefix training job
submit_bpe_prefix_job() {
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
    --tokenizer_dir=$TOK_DIR \
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

# Run 1: Prefix loss with BPE, decay=0.5
submit_bpe_prefix_job "bpe-prefix-d0.5-c4-d12" "prefix" "0.5" "0.0"

# Run 2: Prefix interp loss with BPE, alpha=0.2
submit_bpe_prefix_job "bpe-prefix-interp0.2-c4-d12" "prefix_interp" "0.5" "0.2"

# Run 3: Prefix BCE loss with BPE
submit_bpe_prefix_job "bpe-prefix-bce-c4-d12" "prefix_bce" "0.5" "0.0"

echo ""
echo "=== All 3 BPE prefix experiments submitted ==="
echo "All runs log to wandb project 'nanochat' (entity: goodarzilab)"
echo "Baseline comparison: bpe-50k-flat-c4-d12 (already submitted)"
echo "Monitor with: squeue -u \$USER"

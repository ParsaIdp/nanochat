#!/bin/bash
# Submit BPE baseline and prefix experiments using a 32K BPE tokenizer.
#
# This script expects the 32K BPE tokenizer to already be trained at:
#   /large_storage/goodarzilab/parsaidp/nanochat/tokenizer-32k/
#
# If the tokenizer hasn't been trained yet, run:
#   python -m scripts.tok_train --vocab_size=32768 --tokenizer_dir=tokenizer-32k
#
# Usage: bash scripts/submit_bpe_ablations.sh [TOK_TRAIN_JOBID]
#   Optional: pass the SLURM job ID of a running tok_train job to set dependencies.

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

TOK_DIR=/large_storage/goodarzilab/parsaidp/nanochat/tokenizer-32k
TOK_TRAIN_JOBID=${1:-""}

mkdir -p logs

# =============================================================================
# Step 1: Generate ancestor data (needed for prefix losses)
# =============================================================================
generate_ancestors() {
    echo "=== Generating BPE ancestor data ==="
    if [ ! -f "$TOK_DIR/tokenizer.pkl" ]; then
        echo "ERROR: tokenizer not found at $TOK_DIR. Train it first."
        exit 1
    fi
    if [ ! -f "$TOK_DIR/token_ancestors.pt" ]; then
        python -m scripts.bpe_generate_ancestors --tokenizer_dir "$TOK_DIR"
    else
        echo "Ancestor data already exists, skipping."
    fi
}

# If we have a dependency job, we need to generate ancestors after it completes
# Otherwise, try generating now
if [ -z "$TOK_TRAIN_JOBID" ]; then
    generate_ancestors
    DEP_FLAG=""
else
    echo "Will wait for tokenizer training job $TOK_TRAIN_JOBID to finish."
    DEP_FLAG="--dependency=afterok:$TOK_TRAIN_JOBID"
fi

echo "=== Submitting BPE experiments (32K vocab, matching LZ78) ==="
echo "Tokenizer: $TOK_DIR"

# =============================================================================
# Helper function
# =============================================================================
submit_bpe_job() {
    local RUN_NAME=$1
    local LOSS_MODE=$2
    local DECAY=$3
    local ALPHA=$4
    local EXTRA_DEP=$5  # optional extra dependency

    local DEP=""
    if [ -n "$EXTRA_DEP" ]; then
        DEP="--dependency=afterok:$EXTRA_DEP"
    fi

    JOB=$(sbatch --parsable \
        $DEP \
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

# Generate ancestors on first BPE prefix run if not done
if [ "$LOSS_MODE" != "standard" ] && [ ! -f "$TOK_DIR/token_ancestors.pt" ]; then
    python -m scripts.bpe_generate_ancestors --tokenizer_dir=$TOK_DIR
fi

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

# =============================================================================
# Step 2: Submit BPE baseline (standard loss, flat embedding)
# =============================================================================
submit_bpe_job "bpe-32k-flat-c4-d12" "standard" "0.0" "0.0" "$TOK_TRAIN_JOBID"

# =============================================================================
# Step 3: Submit BPE prefix experiments
# =============================================================================
submit_bpe_job "bpe-32k-prefix-d0.5-c4-d12" "prefix" "0.5" "0.0" "$TOK_TRAIN_JOBID"
submit_bpe_job "bpe-32k-prefix-interp0.2-c4-d12" "prefix_interp" "0.5" "0.2" "$TOK_TRAIN_JOBID"
submit_bpe_job "bpe-32k-prefix-bce-c4-d12" "prefix_bce" "0.5" "0.0" "$TOK_TRAIN_JOBID"

echo ""
echo "=== All 4 BPE experiments submitted ==="
echo "All runs log to wandb project 'nanochat' (entity: goodarzilab)"
echo "Monitor with: squeue -u \$USER"

#!/bin/bash
# Resubmit 9 LZ78-family experiments to run to full convergence (~5133 steps).
# Previous runs were preempted at step ~2000 on the preemptible partition.
# This script adds --save_every=1000 and --model_tag so each run:
#   1. Checkpoints every 1000 steps (survives preemption)
#   2. Uses a unique checkpoint directory (no collisions)
#
# After preemption, resubmit with --resume_from_step=<last_saved_step>
#
# Usage: bash scripts/submit_full_convergence.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_DIR=/large_storage/goodarzilab/parsaidp/weezl/lz78_ablations
mkdir -p logs

echo "=== Submitting 9 LZ78-family full-convergence runs ==="
echo "Each run saves checkpoints every 1000 steps for preemption recovery."
echo ""

# Helper function
submit_lz78_run() {
    local RUN_NAME=$1
    local TOK_DIR=$2
    local EMB_MODE=$3
    local DATA_DIR=$4
    local MODEL_TAG=$5
    local RESUME=${6:-"-1"}  # default: no resume

    local RESUME_FLAG=""
    if [ "$RESUME" != "-1" ]; then
        RESUME_FLAG="--resume_from_step=$RESUME"
    fi

    JOB=$(sbatch --parsable \
        --job-name="$RUN_NAME" \
        --output="logs/${RUN_NAME}-%j.out" \
        --partition=preemptible \
        --gres=gpu:2 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        --wrap="set -e
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
CONDA_INIT=''
for c in \"\$HOME/miniconda/etc/profile.d/conda.sh\" \"\$HOME/anaconda3/etc/profile.d/conda.sh\" \"\$HOME/miniconda3/etc/profile.d/conda.sh\" \"/opt/conda/etc/profile.d/conda.sh\"; do
  [ -f \"\$c\" ] && CONDA_INIT=\"\$c\" && break
done
[ -n \"\$CONDA_INIT\" ] && . \"\$CONDA_INIT\"
conda activate wave
cd $(pwd)
pip install -e . 2>/dev/null || true

torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --tokenizer_type=lz78 \
    --tokenizer_dir=$TOK_DIR \
    --embedding_mode=$EMB_MODE \
    --data_mode=pretokenized \
    --pretokenized_dir=$DATA_DIR \
    --run=$RUN_NAME \
    --model_tag=$MODEL_TAG \
    --depth=12 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --save_every=1000 \
    --core_metric_every=-1 \
    $RESUME_FLAG")

    echo "Submitted: $RUN_NAME (job $JOB, tag=$MODEL_TAG)"
}

# ---- Priority 1: Best flat baselines (needed for fair BPE comparison) ----

submit_lz78_run "freqgated-32k-flat-full" \
    "$BASE_DIR/tokenizers/freqgated_32k" "flat" \
    "$BASE_DIR/data/freqgated_32k" "freqgated-flat"

submit_lz78_run "lz78-32k-flat-full" \
    "$BASE_DIR/tokenizers/lz78_32k" "flat" \
    "$BASE_DIR/data/lz78_32k" "lz78-flat"

submit_lz78_run "trie2x-44k-flat-full" \
    "$BASE_DIR/tokenizers/trie2x_44k" "flat" \
    "$BASE_DIR/data/trie2x_44k" "trie2x-flat"

# ---- Priority 2: Structured embeddings ----

submit_lz78_run "freqgated-32k-struct-full" \
    "$BASE_DIR/tokenizers/freqgated_32k" "structured" \
    "$BASE_DIR/data/freqgated_32k" "freqgated-struct"

submit_lz78_run "lz78-32k-struct-full" \
    "$BASE_DIR/tokenizers/lz78_32k" "structured" \
    "$BASE_DIR/data/lz78_32k" "lz78-struct"

submit_lz78_run "trie2x-44k-struct-full" \
    "$BASE_DIR/tokenizers/trie2x_44k" "structured" \
    "$BASE_DIR/data/trie2x_44k" "trie2x-struct"

# ---- Priority 3: Tuple and hierarchical embeddings ----

submit_lz78_run "freqgated-32k-tuple-full" \
    "$BASE_DIR/tokenizers/freqgated_32k" "tuple" \
    "$BASE_DIR/data/freqgated_32k" "freqgated-tuple"

submit_lz78_run "trie2x-44k-hier-full" \
    "$BASE_DIR/tokenizers/trie2x_44k" "hierarchical" \
    "$BASE_DIR/data/trie2x_44k" "trie2x-hier"

submit_lz78_run "trie2x-44k-tuple-full" \
    "$BASE_DIR/tokenizers/trie2x_44k" "tuple" \
    "$BASE_DIR/data/trie2x_44k" "trie2x-tuple"

echo ""
echo "=== All 9 full-convergence runs submitted ==="
echo ""
echo "After preemption, resubmit with resume. Example:"
echo "  Edit the submit_lz78_run call to add resume step as 6th arg:"
echo "  submit_lz78_run ... \"freqgated-flat\" \"1000\""
echo ""
echo "Monitor: squeue -u \$USER"
echo "Check checkpoints: ls /large_storage/goodarzilab/parsaidp/nanochat/base_checkpoints/"

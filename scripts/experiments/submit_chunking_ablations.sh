#!/bin/bash
# Submit 4 chunking ablation experiments:
#   1. lz78-32k-chunked-c4-d12    (LZ78 with GPT-4 regex chunking, pretokenized)
#   2. freqgated-32k-chunked-c4-d12 (FreqGated with chunking, pretokenized)
#   3. trie2x-44k-chunked-c4-d12  (Trie2x with chunking, pretokenized)
#   4. bpe-32k-unchunked-c4-d12   (BPE without chunking, online tokenization)
#
# Usage: bash scripts/submit_chunking_ablations.sh

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

# Paths
BASE_DIR=/large_storage/goodarzilab/parsaidp/weezl/lz78_ablations
BPE_TOK_DIR=/large_storage/goodarzilab/parsaidp/nanochat/tokenizer-32k
mkdir -p logs

echo "=== Chunking Ablation Experiments ==="

# =============================================================================
# Step 1: Submit pre-tokenization jobs with CHUNKED=1 (CPU only)
# =============================================================================

echo "=== Submitting chunked pre-tokenization jobs ==="

PRETOK_LZ78=$(sbatch --parsable \
    --export=ALL,TOK_DIR=$BASE_DIR/tokenizers/lz78_32k,OUTPUT_DIR=$BASE_DIR/data/lz78_32k_chunked,TOK_TYPE=lz78,CHUNKED=1 \
    --job-name=pretok-lz78-chunked \
    --output=logs/pretok-lz78-chunked-%j.out \
    scripts/lz78_pretokenize.slurm)
echo "LZ78 chunked pretokenize job: $PRETOK_LZ78"

PRETOK_FG=$(sbatch --parsable \
    --export=ALL,TOK_DIR=$BASE_DIR/tokenizers/freqgated_32k,OUTPUT_DIR=$BASE_DIR/data/freqgated_32k_chunked,TOK_TYPE=freqgated,CHUNKED=1 \
    --job-name=pretok-freqgated-chunked \
    --output=logs/pretok-freqgated-chunked-%j.out \
    scripts/lz78_pretokenize.slurm)
echo "FreqGated chunked pretokenize job: $PRETOK_FG"

PRETOK_TRIE=$(sbatch --parsable \
    --export=ALL,TOK_DIR=$BASE_DIR/tokenizers/trie2x_44k,OUTPUT_DIR=$BASE_DIR/data/trie2x_44k_chunked,TOK_TYPE=trie2x,CHUNKED=1 \
    --job-name=pretok-trie2x-chunked \
    --output=logs/pretok-trie2x-chunked-%j.out \
    scripts/lz78_pretokenize.slurm)
echo "Trie2x chunked pretokenize job: $PRETOK_TRIE"

# =============================================================================
# Step 2: Submit training jobs (GPU, with dependencies)
# =============================================================================

echo "=== Submitting training jobs ==="

# Run 1: LZ78 chunked (flat embedding, pretokenized)
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_LZ78 \
    --export=ALL,RUN_NAME=lz78-32k-chunked-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/lz78_32k,EMB_MODE=flat,DATA_DIR=$BASE_DIR/data/lz78_32k_chunked \
    --job-name=lz78-32k-chunked-c4-d12 \
    --output=logs/lz78-32k-chunked-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: lz78-32k-chunked-c4-d12 (job $JOB, after pretok $PRETOK_LZ78)"

# Run 2: FreqGated chunked (flat embedding, pretokenized)
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_FG \
    --export=ALL,RUN_NAME=freqgated-32k-chunked-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/freqgated_32k,EMB_MODE=flat,DATA_DIR=$BASE_DIR/data/freqgated_32k_chunked \
    --job-name=freqgated-32k-chunked-c4-d12 \
    --output=logs/freqgated-32k-chunked-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: freqgated-32k-chunked-c4-d12 (job $JOB, after pretok $PRETOK_FG)"

# Run 3: Trie2x chunked (flat embedding, pretokenized)
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_TRIE \
    --export=ALL,RUN_NAME=trie2x-44k-chunked-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/trie2x_44k,EMB_MODE=flat,DATA_DIR=$BASE_DIR/data/trie2x_44k_chunked \
    --job-name=trie2x-44k-chunked-c4-d12 \
    --output=logs/trie2x-44k-chunked-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: trie2x-44k-chunked-c4-d12 (job $JOB, after pretok $PRETOK_TRIE)"

# Run 4: BPE unchunked (online tokenization, no pretok dependency)
BPE_WRAP="set -e
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
CONDA_INIT=''
for c in \"\$HOME/miniconda/etc/profile.d/conda.sh\" \"\$HOME/anaconda3/etc/profile.d/conda.sh\" \"\$HOME/miniconda3/etc/profile.d/conda.sh\" \"/opt/conda/etc/profile.d/conda.sh\"; do
  [ -f \"\$c\" ] && CONDA_INIT=\"\$c\" && break
done
[ -n \"\$CONDA_INIT\" ] && . \"\$CONDA_INIT\"
conda activate wave
cd $PROJECT_ROOT
pip install -e . 2>/dev/null || true

torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --tokenizer_dir=$BPE_TOK_DIR \
    --use_chunking=unchunked \
    --run=bpe-32k-unchunked-c4-d12 \
    --depth=12 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --core_metric_every=-1"

JOB=$(sbatch --parsable \
    --job-name=bpe-32k-unchunked-c4-d12 \
    --output=logs/bpe-32k-unchunked-c4-d12-%j.out \
    --partition=preemptible \
    --gres=gpu:2 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=24:00:00 \
    --wrap="$BPE_WRAP")
echo "Submitted: bpe-32k-unchunked-c4-d12 (job $JOB)"

echo ""
echo "=== All 4 chunking ablation experiments submitted ==="
echo "All training runs will log to wandb project 'nanochat' (entity: goodarzilab)"
echo "Monitor with: squeue -u \$USER"

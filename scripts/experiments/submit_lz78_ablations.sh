#!/bin/bash
# Submit all 8 LZ78 vs BPE ablation experiments on C4.
#
# Usage: bash scripts/submit_lz78_ablations.sh
#
# Prerequisites:
# - conda env 'wave' with nanochat installed
# - C4 text files at /large_storage/goodarzilab/parsaidp/weezl/c4_{train,val}.txt

set -e

# Resolve project root and cd into it
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

# Paths
TSV_DIR=/large_storage/goodarzilab/parsaidp/weezl/dictionaries
BASE_DIR=/large_storage/goodarzilab/parsaidp/weezl/lz78_ablations
mkdir -p "$BASE_DIR" logs

# =============================================================================
# Step 1: Setup tokenizers (runs locally, fast)
# =============================================================================

echo "=== Setting up tokenizers ==="

# Normal LZ78 (32K)
python -m scripts.lz78_setup_tokenizer \
    --tsv_path "$TSV_DIR/lz78_dict_32k.tsv" \
    --tsv_format lz78 \
    --output_dir "$BASE_DIR/tokenizers/lz78_32k"

# Freq Gated (32K)
python -m scripts.lz78_setup_tokenizer \
    --tsv_path "$TSV_DIR/freq_gated_full_32000.tsv" \
    --tsv_format lz78 \
    --output_dir "$BASE_DIR/tokenizers/freqgated_32k"

# Compressed Trie 2x (44K)
python -m scripts.lz78_setup_tokenizer \
    --tsv_path "$TSV_DIR/compressed_full_2x_32000.tsv" \
    --tsv_format compressed \
    --output_dir "$BASE_DIR/tokenizers/trie2x_44k"

echo "=== Tokenizers ready ==="

# =============================================================================
# Step 2: Submit pre-tokenization jobs (CPU only)
# =============================================================================

echo "=== Submitting pre-tokenization jobs ==="

PRETOK_LZ78=$(sbatch --parsable \
    --export=ALL,TOK_DIR=$BASE_DIR/tokenizers/lz78_32k,OUTPUT_DIR=$BASE_DIR/data/lz78_32k,TOK_TYPE=lz78 \
    --job-name=pretok-lz78 \
    scripts/lz78_pretokenize.slurm)
echo "LZ78 pretokenize job: $PRETOK_LZ78"

PRETOK_FG=$(sbatch --parsable \
    --export=ALL,TOK_DIR=$BASE_DIR/tokenizers/freqgated_32k,OUTPUT_DIR=$BASE_DIR/data/freqgated_32k,TOK_TYPE=freqgated \
    --job-name=pretok-freqgated \
    scripts/lz78_pretokenize.slurm)
echo "FreqGated pretokenize job: $PRETOK_FG"

PRETOK_TRIE=$(sbatch --parsable \
    --export=ALL,TOK_DIR=$BASE_DIR/tokenizers/trie2x_44k,OUTPUT_DIR=$BASE_DIR/data/trie2x_44k,TOK_TYPE=trie2x \
    --job-name=pretok-trie2x \
    scripts/lz78_pretokenize.slurm)
echo "Trie2x pretokenize job: $PRETOK_TRIE"

# =============================================================================
# Step 3: Submit training jobs (GPU, with dependencies)
# =============================================================================

echo "=== Submitting training jobs ==="

# Run 1: BPE baseline (uses online tokenization, no pretok dependency)
BPE_JOB=$(sbatch --parsable scripts/bpe_train.slurm)
echo "Submitted: bpe-50k-flat-c4-d12 (job $BPE_JOB)"

# Run 2: LZ78 flat
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_LZ78 \
    --export=ALL,RUN_NAME=lz78-32k-flat-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/lz78_32k,EMB_MODE=flat,DATA_DIR=$BASE_DIR/data/lz78_32k \
    --job-name=lz78-32k-flat-c4-d12 \
    --output=logs/lz78-32k-flat-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: lz78-32k-flat-c4-d12 (job $JOB, after $PRETOK_LZ78)"

# Run 3: LZ78 structured
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_LZ78 \
    --export=ALL,RUN_NAME=lz78-32k-struct-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/lz78_32k,EMB_MODE=structured,DATA_DIR=$BASE_DIR/data/lz78_32k \
    --job-name=lz78-32k-struct-c4-d12 \
    --output=logs/lz78-32k-struct-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: lz78-32k-struct-c4-d12 (job $JOB, after $PRETOK_LZ78)"

# Run 4: FreqGated flat
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_FG \
    --export=ALL,RUN_NAME=freqgated-32k-flat-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/freqgated_32k,EMB_MODE=flat,DATA_DIR=$BASE_DIR/data/freqgated_32k \
    --job-name=freqgated-32k-flat-c4-d12 \
    --output=logs/freqgated-32k-flat-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: freqgated-32k-flat-c4-d12 (job $JOB, after $PRETOK_FG)"

# Run 5: FreqGated structured
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_FG \
    --export=ALL,RUN_NAME=freqgated-32k-struct-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/freqgated_32k,EMB_MODE=structured,DATA_DIR=$BASE_DIR/data/freqgated_32k \
    --job-name=freqgated-32k-struct-c4-d12 \
    --output=logs/freqgated-32k-struct-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: freqgated-32k-struct-c4-d12 (job $JOB, after $PRETOK_FG)"

# Run 6: Trie2x flat
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_TRIE \
    --export=ALL,RUN_NAME=trie2x-44k-flat-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/trie2x_44k,EMB_MODE=flat,DATA_DIR=$BASE_DIR/data/trie2x_44k \
    --job-name=trie2x-44k-flat-c4-d12 \
    --output=logs/trie2x-44k-flat-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: trie2x-44k-flat-c4-d12 (job $JOB, after $PRETOK_TRIE)"

# Run 7: Trie2x structured
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_TRIE \
    --export=ALL,RUN_NAME=trie2x-44k-struct-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/trie2x_44k,EMB_MODE=structured,DATA_DIR=$BASE_DIR/data/trie2x_44k \
    --job-name=trie2x-44k-struct-c4-d12 \
    --output=logs/trie2x-44k-struct-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: trie2x-44k-struct-c4-d12 (job $JOB, after $PRETOK_TRIE)"

# Run 8: Trie2x hierarchical
JOB=$(sbatch --parsable --dependency=afterok:$PRETOK_TRIE \
    --export=ALL,RUN_NAME=trie2x-44k-hier-c4-d12,TOK_DIR=$BASE_DIR/tokenizers/trie2x_44k,EMB_MODE=hierarchical,DATA_DIR=$BASE_DIR/data/trie2x_44k \
    --job-name=trie2x-44k-hier-c4-d12 \
    --output=logs/trie2x-44k-hier-c4-d12-%j.out \
    scripts/lz78_train.slurm)
echo "Submitted: trie2x-44k-hier-c4-d12 (job $JOB, after $PRETOK_TRIE)"

echo ""
echo "=== All 8 experiments submitted ==="
echo "All training runs will log to wandb project 'nanochat' (entity: goodarzilab)"
echo "Monitor with: squeue -u \$USER"

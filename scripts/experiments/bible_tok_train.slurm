#!/bin/bash
#SBATCH --job-name=bible_tok
#SBATCH --output=bible_tok_%j.out
#SBATCH --error=bible_tok_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=preemptible

# Train a tokenizer on the Bible dataset.
# Specify tokenizer via TOKENIZER_DIR and VOCAB_SIZE environment variables.
# Examples:
#   TOKENIZER_DIR=tokenizer-2k VOCAB_SIZE=2048 sbatch scripts/bible_tok_train.slurm
#   TOKENIZER_DIR=tokenizer-4k VOCAB_SIZE=4096 sbatch scripts/bible_tok_train.slurm
#   TOKENIZER_DIR=tokenizer-8k VOCAB_SIZE=8192 sbatch scripts/bible_tok_train.slurm

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/bible_run}"

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

# Make sure Python can find the project modules
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Tokenizer training parameters
MAX_CHARS="${MAX_CHARS:-5000000}"          # 5M chars (small for bible dataset)
DOC_CAP="${DOC_CAP:-10000}"                # Maximum characters per document
VOCAB_SIZE="${VOCAB_SIZE:-8192}"           # Vocabulary size (default: 8K)
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer}" # Tokenizer directory name (relative to base_dir)

echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "max_chars: $MAX_CHARS"
echo "doc_cap: $DOC_CAP"
echo "vocab_size: $VOCAB_SIZE"
echo "tokenizer_dir: $TOKENIZER_DIR"
echo "Tokenizer will be saved to: $NANOCHAT_BASE_DIR/$TOKENIZER_DIR"
echo ""

# Check if parquet files exist
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
NUM_DATA_FILES=$(find "$DATA_DIR" -name "*.parquet" 2>/dev/null | wc -l)
if [ "$NUM_DATA_FILES" -eq 0 ]; then
    echo "ERROR: No parquet files found in $DATA_DIR"
    echo "Please create parquet files first using:"
    echo "  python bible_data/make_parquet.py --infile <pg10.txt> --outdir $DATA_DIR --basename kjv"
    exit 1
fi
echo "Found $NUM_DATA_FILES parquet files in $DATA_DIR"
echo ""

# Run tokenizer training
echo "Starting tokenizer training..."
python "$PROJECT_ROOT/scripts/tok_train.py" \
    --max_chars="$MAX_CHARS" \
    --doc_cap="$DOC_CAP" \
    --vocab_size="$VOCAB_SIZE" \
    --tokenizer_dir="$TOKENIZER_DIR"

echo ""
echo "Tokenizer training completed successfully!"
echo "Tokenizer saved to: $NANOCHAT_BASE_DIR/$TOKENIZER_DIR"

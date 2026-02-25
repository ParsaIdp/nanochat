#!/bin/bash
#SBATCH --job-name=pretok
#SBATCH --partition=preemptible
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pretok-%j.out

# Usage: sbatch --export=TOK_DIR=...,OUTPUT_DIR=...,TOK_TYPE=... scripts/lz78_pretokenize.slurm

set -e
export OMP_NUM_THREADS=1

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

C4_TRAIN=/large_storage/goodarzilab/parsaidp/weezl/c4_train.txt
C4_VAL=/large_storage/goodarzilab/parsaidp/weezl/c4_val.txt

echo "Tokenizer: $TOK_DIR"
echo "Output: $OUTPUT_DIR"
echo "Type: $TOK_TYPE"
echo "Chunked: ${CHUNKED:-0}"

CHUNKED_FLAG=""
if [ "${CHUNKED:-0}" = "1" ]; then
    CHUNKED_FLAG="--chunked"
fi

# Pretokenize train split
python -m scripts.lz78_pretokenize \
    --tokenizer_dir "$TOK_DIR" \
    --input_path "$C4_TRAIN" \
    --output_dir "$OUTPUT_DIR" \
    --split train \
    $CHUNKED_FLAG

# Pretokenize val split
python -m scripts.lz78_pretokenize \
    --tokenizer_dir "$TOK_DIR" \
    --input_path "$C4_VAL" \
    --output_dir "$OUTPUT_DIR" \
    --split val \
    $CHUNKED_FLAG

echo "Done pretokenizing $TOK_TYPE"

#!/bin/bash
#SBATCH --job-name=tok_train
#SBATCH --output=tok_train_%j.out
#SBATCH --error=tok_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=preemptible

# Train a BPE tokenizer using scripts/tok_train.py
# This script follows the same setup pattern as the other training scripts in this repo

set -e  # Exit on error

# Environment setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/large_storage/goodarzilab/parsaidp/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Get project root directory
# SLURM_SUBMIT_DIR is set by SLURM to the directory where sbatch was run from
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback: try to find project root from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Verify project root exists and has expected files
if [ ! -f "$PROJECT_ROOT/scripts/tok_train.py" ]; then
    echo "Error: Could not find scripts/tok_train.py in project root: $PROJECT_ROOT"
    echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"
echo "Current directory: $(pwd)"

# Initialize and activate conda environment
# Try common conda installation paths (check both miniconda and miniconda3)
CONDA_INIT=""
if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="$HOME/miniconda/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="/opt/conda/etc/profile.d/conda.sh"
fi

if [ -n "$CONDA_INIT" ] && [ -f "$CONDA_INIT" ]; then
    source "$CONDA_INIT"
    echo "Initialized conda from: $CONDA_INIT"
else
    # Try to find conda in PATH or use explicit path
    if command -v conda &> /dev/null; then
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            echo "Initialized conda from: $CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "Error: Could not find conda.sh. CONDA_BASE: $CONDA_BASE"
            exit 1
        fi
    else
        echo "Error: Could not find conda. Please ensure conda is installed and accessible."
        exit 1
    fi
fi

# Activate the wave conda environment
echo "Attempting to activate conda environment: wave"
conda activate wave || {
    echo "Error: Failed to activate 'wave' environment"
    echo "Available environments:"
    conda env list
    exit 1
}
echo "Activated conda environment: wave"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install project dependencies
echo "Installing project dependencies..."
set +e  # Temporarily disable exit on error for pip install
pip install -e . 2>&1
INSTALL_EXIT=$?
set -e  # Re-enable exit on error

if [ $INSTALL_EXIT -ne 0 ]; then
    echo "pip install -e . failed, installing dependencies directly..."
    # Install CPU version of torch for CPU partition
    pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.8.0
    # Install other dependencies
    pip install datasets>=4.0.0 \
        fastapi>=0.117.1 \
        psutil>=7.1.0 \
        pyarrow \
        regex>=2025.9.1 \
        requests \
        tiktoken>=0.11.0 \
        tokenizers>=0.22.0 \
        uvicorn>=0.36.0 \
        wandb>=0.21.3
fi

# Install rustbpe if needed
if ! python -c "import rustbpe" 2>/dev/null; then
    echo "rustbpe not found, attempting to install..."
    pip install rustbpe || echo "Warning: Could not install rustbpe, may need to build from source"
fi

# Setup Rust for the tokenizer (if needed)
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Build the Rust BPE tokenizer extension (if needed)
# Note: Assumes rustbpe is installed in conda or available via pip
if [ -d "rustbpe" ]; then
    echo "Building Rust BPE tokenizer extension"
    # Try using maturin directly if available, otherwise assume it's already built
    if command -v maturin &> /dev/null; then
        maturin develop --release --manifest-path rustbpe/Cargo.toml
    else
        echo "maturin not found, assuming rustbpe extension is already available"
    fi
fi

# Reset report (optional, comment out if you don't want to reset)
# python -m nanochat.report reset

# -----------------------------------------------------------------------------
# TOKENIZER TRAINING PARAMETERS
# Modify these values directly, or override via environment variables when submitting:
#   sbatch scripts/tok_train.sh                           # uses values below
#   MAX_CHARS=4000000000 TOKENIZER_DIR=my_tokenizer sbatch scripts/tok_train.sh  # overrides
# -----------------------------------------------------------------------------
MAX_CHARS="${MAX_CHARS:-40000000000}"      # Maximum characters to train on (default: 40B)
DOC_CAP="${DOC_CAP:-10000}"                # Maximum characters per document (default: 10K)
VOCAB_SIZE="${VOCAB_SIZE:-1048576}"        # Vocabulary size (default: 2^20 = 1M)
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer-big}" # Tokenizer directory name (relative to base_dir, default: "tokenizer")

# Make sure Python can find the project modules by adding project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "Added project root to PYTHONPATH: $PROJECT_ROOT"
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Download dataset if needed
# Check if data files exist, if not download some
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
NUM_DATA_FILES=$(find "$DATA_DIR" -name "*.parquet" 2>/dev/null | wc -l)
NUM_FILES_TO_DOWNLOAD="${NUM_FILES_TO_DOWNLOAD:-16}"  # Default: download 16 files

if [ "$NUM_DATA_FILES" -eq 0 ]; then
    echo "No dataset files found. Downloading $NUM_FILES_TO_DOWNLOAD shards..."
    python -m nanochat.dataset -n "$NUM_FILES_TO_DOWNLOAD" -w 4
    echo "Dataset download completed"
else
    echo "Found $NUM_DATA_FILES existing dataset files, skipping download"
fi

# Run tokenizer training (run script directly instead of as module)
echo "Starting tokenizer training..."
echo "max_chars: $MAX_CHARS"
echo "doc_cap: $DOC_CAP"
echo "vocab_size: $VOCAB_SIZE"
echo "tokenizer_dir: $TOKENIZER_DIR"
echo "Tokenizer will be saved to: $NANOCHAT_BASE_DIR/$TOKENIZER_DIR"
echo ""

python "$PROJECT_ROOT/scripts/tok_train.py" \
    --max_chars="$MAX_CHARS" \
    --doc_cap="$DOC_CAP" \
    --vocab_size="$VOCAB_SIZE" \
    --tokenizer_dir="$TOKENIZER_DIR"

echo ""
echo "Tokenizer training completed successfully!"
echo "Tokenizer saved to: $NANOCHAT_BASE_DIR/$TOKENIZER_DIR"

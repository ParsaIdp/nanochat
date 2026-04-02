"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
from collections.abc import Generator

import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from datasets import load_dataset
from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
def index_to_filename(index: int) -> str:
    """Convert a shard index to its parquet filename."""
    return f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


def load_numina_math_cot(split: str = "train", *, cache_subdir: str = "numina_math_cot"):
    """
    Load the NuminaMath-CoT dataset from Hugging Face.

    This is a thin wrapper around datasets.load_dataset so that training code
    can depend on nanochat.dataset instead of importing datasets directly.

    Args:
        split: Dataset split to load (e.g. "train", "test", "validation", or a HF slice).
        cache_subdir: Subdirectory under the nanochat base directory to use as the HF cache dir.

    Returns:
        A datasets.Dataset (or DatasetDict if split is None).
    """
    cache_dir = os.path.join(base_dir, cache_subdir)
    os.makedirs(cache_dir, exist_ok=True)
    return load_dataset("AI-MO/NuminaMath-CoT", split=split, cache_dir=cache_dir)


def _numina_first_generation_text(row: dict) -> str | None:
    """Return ``generations[0]`` as a string if present and non-empty, else None."""
    gens = row.get("generations")
    if gens is None:
        return None
    if isinstance(gens, (list, tuple)) and len(gens) == 0:
        return None
    g0 = gens[0]
    if g0 is None:
        return None
    if isinstance(g0, str):
        t = g0.strip()
    else:
        t = str(g0).strip()
    return t if t else None


def iter_numina_math_cot_docs(
    split: str = "train",
    *,
    max_problems: int | None = None,
    cache_subdir: str = "numina_math_cot",
) -> Generator[str, None, None]:
    """
    Iterate over NuminaMath-CoT documents (one per problem).

    When the row has a ``generations`` field, uses **only** ``generations[0]`` as the
    document (LLM trace to train on), not the gold ``solution``. If ``generations`` is
    missing or empty, falls back to ``problem + solution`` for older dataset revisions.

    Args:
        split: Dataset split (e.g. "train").
        max_problems: Maximum number of problems to yield (default: all).
        cache_subdir: Subdir under base dir for HF cache.

    Yields:
        One document string per problem.
    """
    ds = load_numina_math_cot(split=split, cache_subdir=cache_subdir)
    for i, row in enumerate(ds):
        if max_problems is not None and i >= max_problems:
            break
        trace = _numina_first_generation_text(row)
        if trace is not None:
            doc = trace
        else:
            problem = row.get("problem") or ""
            solution = row.get("solution") or ""
            doc = f"{problem}\n\n{solution}".strip()
        if doc:
            yield doc


def list_parquet_files(data_dir: str | None = None) -> list[str]:
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(
    split: str,
    start: int = 0,
    step: int = 1,
    data_dir: str | None = None,
) -> Generator[list[str], None, None]:
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


def iter_docs_by_split(
    data_dir: str | None,
    split: str,
    *,
    doc_cap: int | None = None,
    max_docs: int | None = None,
    max_chars: int | None = None,
) -> Generator[str, None, None]:
    """
    Same text iterator as tok_train: flatten parquets_iter_batched(split, data_dir),
    optionally crop each doc to doc_cap chars, stop after max_docs or max_chars.
    Used by both tok_train and bpe_eval so they see the same train/val documents.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    nchars = 0
    ndocs = 0
    for batch in parquets_iter_batched(split=split, data_dir=data_dir):
        for doc in batch:
            doc_text = doc
            if doc_cap is not None and len(doc_text) > doc_cap:
                doc_text = doc_text[:doc_cap]
            nchars += len(doc_text)
            ndocs += 1
            yield doc_text
            if max_docs is not None and ndocs >= max_docs:
                return
            if max_chars is not None and nchars >= max_chars:
                return


# -----------------------------------------------------------------------------
def download_single_file(index: int) -> bool:
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")

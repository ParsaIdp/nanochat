"""Download (or reuse cached) C4 via HuggingFace datasets and write it to local parquet shards.

This script is intentionally simple and produces parquet files compatible with nanochat's
local dataset pipeline (expects a 'text' column).

Example:
    python scripts/c4_to_parquet.py --out_dir base_data --max_docs 200000 --shard_size 50000

Then train tokenizer using the existing local-parquet iterator:
    set NANOCHAT_BASE_DIR=%cd%
    python scripts/tok_train.py --vocab_size 10000 --tokenizer_dir c4_dictionaries\\bpe_chunk
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from glob import glob
from itertools import islice

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import login

_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if _hf_token:
    login(token=_hf_token)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)

from nanochat.common import get_project_root


def _write_shard(texts: list[str], out_path: str) -> None:
    table = pa.table({"text": texts})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download C4 and write local parquet shards")
    parser.add_argument("--name", type=str, default="allenai/c4", help="HF dataset name (default: allenai/c4)")
    parser.add_argument("--config", type=str, default="en", help="HF dataset config (default: en)")
    parser.add_argument("--split", type=str, default="train", help="Split to export (default: train)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="base_data",
        help="Output directory for parquet shards. If relative, it is relative to repo root.",
    )
    parser.add_argument("--shard_size", type=int, default=50000, help="Docs per parquet shard")
    parser.add_argument("--max_docs", type=int, default=0, help="Optional max docs to export (0 = all)")
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field name to read text from (default: text)",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted(glob(os.path.join(out_dir, "shard_*.parquet")))
    shard_idx = len(existing)
    exported = shard_idx * args.shard_size

    shard_texts: list[str] = []

    while True:
        try:
            # Streaming download: reads dataset incrementally (still uses the HF cache for downloaded shards).
            ds = load_dataset(args.name, args.config, split=args.split, streaming=True)
            stream = islice(ds, exported, None)

            for row in stream:
                text = row.get(args.text_field, "")
                if not text:
                    continue

                shard_texts.append(text)
                exported += 1

                if (exported % 50_000) == 0:
                    print(f"exported={exported} shard_idx={shard_idx}")

                if len(shard_texts) >= args.shard_size:
                    out_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.parquet")
                    _write_shard(shard_texts, out_path)
                    shard_texts.clear()
                    shard_idx += 1

                if args.max_docs and exported >= args.max_docs:
                    raise StopIteration

            break
        except StopIteration:
            break
        except httpx.RemoteProtocolError as e:
            print(f"Download interrupted ({e}). Retrying... exported={exported} shard_idx={shard_idx}")
            time.sleep(5)
            continue

    if shard_texts:
        out_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.parquet")
        _write_shard(shard_texts, out_path)


if __name__ == "__main__":
    main()

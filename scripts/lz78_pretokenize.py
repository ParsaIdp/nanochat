"""
Pre-tokenize a text file (e.g. C4) using an LZ78 tokenizer and save as .npy shards.

Usage:
    python scripts/lz78_pretokenize.py \
        --tokenizer_dir PATH \
        --input_path PATH \
        --output_dir PATH \
        --split train \
        --shard_size 1000000

Output:
    output_dir/{split}/shard_000000.npy, shard_000001.npy, ...
"""

import os
import argparse

import numpy as np
from tqdm import tqdm

from nanochat.lz78_tokenizer import LZ78Tokenizer


def count_lines(path: str) -> int:
    """Fast line count."""
    count = 0
    with open(path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize text with LZ78 tokenizer")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to saved LZ78 tokenizer directory")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input text file (one document per line)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy shards")
    parser.add_argument("--split", type=str, default="train", help="Split name (train/val), used as subdirectory")
    parser.add_argument("--shard_size", type=int, default=1_000_000, help="Tokens per shard (default: 1M)")
    parser.add_argument("--chunked", action="store_true", help="Enable GPT-4-style regex chunking before encoding")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_dir}")
    tokenizer = LZ78Tokenizer.from_directory(args.tokenizer_dir)
    if args.chunked:
        tokenizer.set_chunking(True)
        print("Chunking: ENABLED (GPT-4 regex pre-splitting)")
    bos_token = tokenizer.get_bos_token_id()
    print(f"Vocab size: {tokenizer.get_vocab_size()}, BOS: {bos_token}")

    # Create output directory
    split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(split_dir, exist_ok=True)

    # Count lines for progress bar
    print(f"Counting lines in {args.input_path}...")
    total_lines = count_lines(args.input_path)
    print(f"Total lines: {total_lines:,}")

    # Process
    token_buffer = []
    shard_idx = 0
    total_tokens = 0

    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Tokenizing"):
            line = line.rstrip('\n')
            if not line:
                continue
            # Tokenize with BOS prepended
            ids = tokenizer.encode(line, prepend=bos_token)
            token_buffer.extend(ids)

            # Write shards when buffer is full
            while len(token_buffer) >= args.shard_size:
                shard_data = np.array(token_buffer[:args.shard_size], dtype=np.int32)
                shard_path = os.path.join(split_dir, f"shard_{shard_idx:06d}.npy")
                np.save(shard_path, shard_data)
                total_tokens += len(shard_data)
                shard_idx += 1
                token_buffer = token_buffer[args.shard_size:]

    # Write remaining tokens as final shard
    if token_buffer:
        shard_data = np.array(token_buffer, dtype=np.int32)
        shard_path = os.path.join(split_dir, f"shard_{shard_idx:06d}.npy")
        np.save(shard_path, shard_data)
        total_tokens += len(shard_data)
        shard_idx += 1

    print(f"\nDone! Wrote {shard_idx} shards, {total_tokens:,} tokens total")
    print(f"Output: {split_dir}")


if __name__ == "__main__":
    main()

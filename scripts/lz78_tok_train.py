"""Train an LZ78 tokenizer dictionary using the weezl Rust library.

Usage:
    python scripts/lz78_tok_train.py --strategy frequency_gated --vocab_size 32000
    python scripts/lz78_tok_train.py --strategy compressed --vocab_size 32000 --compress
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import get_base_dir, print0
from nanochat.dataset import parquets_iter_batched
from nanochat.lz78_trainer import train_lz78_dictionary


def text_iterator(data_dir: str, max_chars: int | None = None):
    """Yield text chunks from parquet shards."""
    total = 0
    for batch in parquets_iter_batched(data_dir):
        for text in batch:
            yield text
            total += len(text)
            if max_chars is not None and total >= max_chars:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LZ78 tokenizer dictionary")
    parser.add_argument("--strategy", type=str, default="frequency_gated",
                        choices=["standard", "frequency_gated", "multi_round",
                                 "cost_adjusted", "smart_prune", "flat_prune", "compressed"],
                        help="Training strategy")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Maximum vocabulary size")
    parser.add_argument("--max_chars", type=int, default=None,
                        help="Maximum characters to process (default: all)")
    parser.add_argument("--compress", action="store_true",
                        help="Build Patricia trie compression")
    parser.add_argument("--compact", action="store_true", default=True,
                        help="Run compact_output_vocab (default: True)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <base_dir>/tokenizer_lz78)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory with parquet shards")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base_dir = get_base_dir()
    if args.data_dir is None:
        args.data_dir = os.path.join(base_dir, "base_data")
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "tokenizer_lz78")

    os.makedirs(args.output_dir, exist_ok=True)
    tsv_name = "compressed_trie.tsv" if args.compress else "lz78_dict.tsv"
    output_path = os.path.join(args.output_dir, tsv_name)

    print0(f"Strategy: {args.strategy}")
    print0(f"Vocab size: {args.vocab_size}")
    print0(f"Data dir: {args.data_dir}")
    print0(f"Output: {output_path}")

    t0 = time.time()
    result_path = train_lz78_dictionary(
        strategy=args.strategy,
        text_iterator=text_iterator(args.data_dir, args.max_chars),
        max_vocab=args.vocab_size,
        max_chars=args.max_chars,
        compress=args.compress,
        output_path=output_path,
        compact=args.compact,
    )
    elapsed = time.time() - t0

    print0(f"Done in {elapsed:.1f}s -> {result_path}")


if __name__ == "__main__":
    main()

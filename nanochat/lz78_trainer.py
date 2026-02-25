"""LZ78 dictionary trainer using the weezl Rust library.

Thin Python wrapper around the lz_tokenizer_rs Rust bindings that provides
a unified interface for training LZ78 dictionaries with 7 strategies.
Outputs TSV files compatible with LZ78Tokenizer.from_tsv().

Strategies:
    standard: Basic LZ78 dictionary building.
    frequency_gated: LZ78 with periodic eviction of low-frequency leaves.
    multi_round: Train standard, prune by frequency, retrain on pruned vocab.
    cost_adjusted: Prune penalizing entries needing many ancestor slots.
    smart_prune: Output-only tokens; prefixes kept but don't count toward budget.
    flat_prune: Flat dictionary with no tree constraint.
    compressed: Build Patricia trie from standard or frequency_gated dict.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator

logger = logging.getLogger(__name__)


def _get_rust_tokenizer():
    """Import and return the Rust tokenizer module."""
    try:
        from lz_tokenizer import lz_tokenizer_rs
        return lz_tokenizer_rs
    except ImportError as e:
        raise ImportError(
            "lz_tokenizer_rs not found. Install the weezl Python package: "
            "cd /path/to/weezl/python_package && pip install -e ."
        ) from e


def _collect_text(text_iterator: Iterator[str], max_chars: int | None = None) -> str:
    """Collect text from an iterator up to max_chars."""
    parts: list[str] = []
    total = 0
    for chunk in text_iterator:
        parts.append(chunk)
        total += len(chunk)
        if max_chars is not None and total >= max_chars:
            break
    text = "".join(parts)
    if max_chars is not None:
        text = text[:max_chars]
    return text


def train_lz78_dictionary(
    strategy: str,
    text_iterator: Iterator[str],
    max_vocab: int = 32000,
    max_chars: int | None = None,
    compress: bool = False,
    output_path: str = "lz78_dict.tsv",
    compact: bool = True,
) -> str:
    """Train an LZ78 dictionary and save to TSV.

    Args:
        strategy: One of 'standard', 'frequency_gated', 'multi_round',
            'cost_adjusted', 'smart_prune', 'flat_prune', 'compressed'.
        text_iterator: Iterator yielding text chunks for training.
        max_vocab: Maximum vocabulary size.
        max_chars: Maximum characters to consume from iterator.
        compress: If True, build Patricia trie from the trained dictionary.
        output_path: Path to save the TSV file.
        compact: If True, run compact_output_vocab before saving.

    Returns:
        Path to the saved TSV file.
    """
    rs = _get_rust_tokenizer()

    valid_strategies = {
        "standard", "frequency_gated", "multi_round",
        "cost_adjusted", "smart_prune", "flat_prune", "compressed",
    }
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown strategy {strategy!r}, expected one of {valid_strategies}")

    logger.info(f"Collecting text (max_chars={max_chars})...")
    t0 = time.time()
    text = _collect_text(text_iterator, max_chars)
    logger.info(f"Collected {len(text):,} chars in {time.time() - t0:.1f}s")

    logger.info(f"Training LZ78 dictionary: strategy={strategy}, max_vocab={max_vocab}")
    t0 = time.time()

    if strategy == "standard":
        tok = rs.PyPretrainedTokenizer.train(text, max_vocab)

    elif strategy == "frequency_gated":
        tok = rs.PyPretrainedTokenizer.train_frequency_gated(text, max_vocab)

    elif strategy == "multi_round":
        # Train large, count frequencies, prune to target
        tok = rs.PyPretrainedTokenizer.train(text, max_vocab * 4)
        freqs = tok.count_frequencies(text)
        tok = tok.prune_to_size(max_vocab, freqs)

    elif strategy == "cost_adjusted":
        tok = rs.PyPretrainedTokenizer.train(text, max_vocab * 4)
        freqs = tok.count_frequencies(text)
        tok = tok.prune_cost_adjusted(max_vocab, freqs)

    elif strategy == "smart_prune":
        tok = rs.PyPretrainedTokenizer.train(text, max_vocab * 4)
        freqs = tok.count_frequencies(text)
        tok = tok.prune_smart(max_vocab, freqs)

    elif strategy == "flat_prune":
        tok = rs.PyPretrainedTokenizer.train(text, max_vocab * 4)
        freqs = tok.count_frequencies(text)
        tok = tok.prune_flat(max_vocab, freqs)

    elif strategy == "compressed":
        tok = rs.PyPretrainedTokenizer.train_frequency_gated(text, max_vocab)
        compress = True  # Force compression for this strategy

    train_time = time.time() - t0
    logger.info(f"Training done in {train_time:.1f}s, vocab_size={tok.vocab_size()}")

    if compact and not compress:
        logger.info("Compacting output vocab...")
        tok.compact_output_vocab(text)
        logger.info(f"Output vocab: {tok.output_vocab_size()} / {tok.vocab_size()}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if compress:
        logger.info("Building compressed Patricia trie...")
        compressed = rs.PyCompressedTokenizer.from_pretrained(tok)
        compressed.save(output_path)
        logger.info(
            f"Saved compressed trie: {compressed.vocab_size()} tokens, "
            f"{compressed.trie_node_count()} nodes -> {output_path}"
        )
        return output_path

    tok.save(output_path)
    logger.info(f"Saved dictionary: {tok.vocab_size()} entries -> {output_path}")
    return output_path

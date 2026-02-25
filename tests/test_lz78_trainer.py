"""Tests for LZ78 trainer (nanochat.lz78_trainer).

Tests all 7 training strategies on a small corpus, pruning validity,
compressed trie building, and full pipeline roundtrip.

These tests require the weezl Python package (lz_tokenizer) to be installed.
Mark as slow since training may take a few seconds.

Usage:
    python -m pytest tests/test_lz78_trainer.py -v
"""

import os
import tempfile

import pytest

# Skip entire module if lz_tokenizer is not installed
lz_tokenizer = pytest.importorskip("lz_tokenizer", reason="lz_tokenizer (weezl) not installed")

from nanochat.lz78_trainer import train_lz78_dictionary, _collect_text, _get_rust_tokenizer
from nanochat.lz78_tokenizer import LZ78Tokenizer


SMALL_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
) * 100  # Repeat to give enough data for training


def text_iter(text=SMALL_CORPUS):
    """Yield text in small chunks."""
    chunk_size = 1000
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_get_rust_tokenizer(self):
        rs = _get_rust_tokenizer()
        assert hasattr(rs, "PyPretrainedTokenizer")

    def test_collect_text(self):
        text = _collect_text(text_iter(), max_chars=500)
        assert len(text) == 500

    def test_collect_text_no_limit(self):
        text = _collect_text(text_iter())
        assert len(text) == len(SMALL_CORPUS)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

STRATEGIES = ["standard", "frequency_gated", "multi_round", "cost_adjusted",
              "smart_prune", "flat_prune", "compressed"]


@pytest.mark.slow
class TestStrategies:

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_train_produces_tsv(self, strategy, tmp_path):
        """Each strategy should produce a valid TSV file."""
        compress = strategy == "compressed"
        tsv_name = "compressed_trie.tsv" if compress else "lz78_dict.tsv"
        output_path = str(tmp_path / tsv_name)

        result_path = train_lz78_dictionary(
            strategy=strategy,
            text_iterator=text_iter(),
            max_vocab=512,
            max_chars=len(SMALL_CORPUS),
            compress=compress,
            output_path=output_path,
            compact=not compress,
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_standard_vocab_limit(self, tmp_path):
        """Standard strategy should respect max_vocab."""
        output_path = str(tmp_path / "lz78_dict.tsv")
        train_lz78_dictionary(
            strategy="standard",
            text_iterator=text_iter(),
            max_vocab=256,
            output_path=output_path,
            compact=False,
        )
        # Count lines (excluding comments/blanks)
        with open(output_path) as f:
            entries = [l for l in f if l.strip() and not l.startswith("#")]
        assert len(entries) <= 256

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            train_lz78_dictionary(
                strategy="nonexistent",
                text_iterator=text_iter(),
                max_vocab=256,
                output_path="/dev/null",
            )


# ---------------------------------------------------------------------------
# Pipeline roundtrip: train -> load -> encode -> decode
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPipelineRoundtrip:

    def test_standard_roundtrip(self, tmp_path):
        """Train a dict, load it into LZ78Tokenizer, encode/decode."""
        output_path = str(tmp_path / "lz78_dict.tsv")
        train_lz78_dictionary(
            strategy="standard",
            text_iterator=text_iter(),
            max_vocab=512,
            output_path=output_path,
            compact=False,
        )

        tok = LZ78Tokenizer.from_tsv(output_path, "lz78")
        text = "The quick brown fox"
        ids = tok.encode(text)
        assert len(ids) > 0
        assert tok.decode(ids) == text

    def test_frequency_gated_roundtrip(self, tmp_path):
        output_path = str(tmp_path / "lz78_dict.tsv")
        train_lz78_dictionary(
            strategy="frequency_gated",
            text_iterator=text_iter(),
            max_vocab=512,
            output_path=output_path,
        )

        tok = LZ78Tokenizer.from_tsv(output_path, "lz78")
        text = "Pack my box with five dozen liquor jugs"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_compressed_roundtrip(self, tmp_path):
        output_path = str(tmp_path / "compressed_trie.tsv")
        train_lz78_dictionary(
            strategy="compressed",
            text_iterator=text_iter(),
            max_vocab=512,
            compress=True,
            output_path=output_path,
        )

        tok = LZ78Tokenizer.from_tsv(output_path, "compressed")
        text = "How vexingly quick daft zebras jump"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

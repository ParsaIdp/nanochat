"""Tests for LZ78Tokenizer.

Covers roundtrip encoding/decoding for both TSV formats (lz78, compressed),
byte fallback, chunking, save/load, metadata generation, and edge cases.

Usage:
    python -m pytest tests/test_lz78_tokenizer.py -v
"""

import os
import tempfile

import pytest
import torch

from nanochat.lz78_tokenizer import LZ78Tokenizer, LZ78_SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Helpers: minimal TSV dictionaries for testing
# ---------------------------------------------------------------------------

LZ78_TSV_CONTENT = """\
# code\tparent_code\tchar\tpattern
1\t0\th\th
2\t0\te\te
3\t1\te\the
4\t0\tl\tl
5\t4\tl\tll
6\t0\to\to
7\t0\t \t\x20
8\t0\tw\tw
9\t6\tr\tor
10\t5\to\tllo
"""

COMPRESSED_TSV_CONTENT = """\
# node_idx\tedge_label\tparent_idx\toutput_code\tpattern
0\t\t0\t\t
1\th\t0\t1\th
2\te\t0\t2\te
3\te\t1\t3\the
4\tl\t0\t4\tl
5\tl\t4\t5\tll
6\to\t0\t6\to
7\t \t0\t7\t\x20
8\tw\t0\t8\tw
9\tr\t6\t9\tor
10\to\t5\t10\tllo
"""


@pytest.fixture
def lz78_tsv_path(tmp_path):
    p = tmp_path / "lz78_dict.tsv"
    p.write_text(LZ78_TSV_CONTENT)
    return str(p)


@pytest.fixture
def compressed_tsv_path(tmp_path):
    p = tmp_path / "compressed_trie.tsv"
    p.write_text(COMPRESSED_TSV_CONTENT)
    return str(p)


# ---------------------------------------------------------------------------
# LZ78 format tests
# ---------------------------------------------------------------------------

class TestLZ78Format:

    def test_from_tsv(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        assert tok.get_vocab_size() > 0

    def test_vocab_layout(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        # 10 dict codes + 256 byte fallback + 3 special tokens
        assert tok.get_vocab_size() == 10 + 1 + 256 + len(LZ78_SPECIAL_TOKENS)

    def test_special_tokens(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        specials = tok.get_special_tokens()
        assert "<|bos|>" in specials
        assert "<|eos|>" in specials
        assert "<|pad|>" in specials

    def test_roundtrip_known(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_roundtrip_byte_fallback(self, lz78_tsv_path):
        """Characters not in the dictionary should use byte fallback."""
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        text = "xyz"  # x, y, z not in dict
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_roundtrip_unicode(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        text = "café 🎉"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_empty_string(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        ids = tok.encode("")
        assert ids == []
        assert tok.decode([]) == ""

    def test_prepend_append(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        bos = tok.encode_special("<|bos|>")
        eos = tok.encode_special("<|eos|>")
        ids = tok.encode("he", prepend="<|bos|>", append="<|eos|>")
        assert ids[0] == bos
        assert ids[-1] == eos

    def test_batch_encode(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        texts = ["hello", "world"]
        result = tok.encode(texts)
        assert len(result) == 2
        for i, text in enumerate(texts):
            assert tok.decode(result[i]) == text

    def test_repr(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        r = repr(tok)
        assert "LZ78Tokenizer" in r
        assert "vocab_size=" in r

    def test_invalid_format(self, lz78_tsv_path):
        with pytest.raises(ValueError, match="Unknown tsv_format"):
            LZ78Tokenizer.from_tsv(lz78_tsv_path, "invalid")


# ---------------------------------------------------------------------------
# Compressed trie format tests
# ---------------------------------------------------------------------------

class TestCompressedFormat:

    def test_from_tsv(self, compressed_tsv_path):
        tok = LZ78Tokenizer.from_tsv(compressed_tsv_path, "compressed")
        assert tok.get_vocab_size() > 0

    def test_roundtrip(self, compressed_tsv_path):
        tok = LZ78Tokenizer.from_tsv(compressed_tsv_path, "compressed")
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_hier_parent_codes(self, compressed_tsv_path):
        """Compressed format should populate hierarchical parent codes."""
        tok = LZ78Tokenizer.from_tsv(compressed_tsv_path, "compressed")
        assert tok._hier_parent_codes is not None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:

    def test_chunking_roundtrip(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        tok.set_chunking(True)
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_chunking_toggle(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        text = "hello world"
        ids_no_chunk = tok.encode(text)
        tok.set_chunking(True)
        ids_chunked = tok.encode(text)
        tok.set_chunking(False)
        ids_off = tok.encode(text)
        # With/without chunking may produce different token sequences
        # but both must roundtrip
        assert tok.decode(ids_no_chunk) == text
        assert tok.decode(ids_chunked) == text
        assert tok.decode(ids_off) == text
        # After disabling, should match no-chunk
        assert ids_off == ids_no_chunk


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_load_roundtrip(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        with tempfile.TemporaryDirectory() as tmpdir:
            tok.save(tmpdir)
            loaded = LZ78Tokenizer.from_directory(tmpdir)
            text = "hello world"
            assert loaded.encode(text) == tok.encode(text)
            assert loaded.get_vocab_size() == tok.get_vocab_size()
            assert loaded.decode(tok.encode(text)) == text

    def test_save_creates_artifacts(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        with tempfile.TemporaryDirectory() as tmpdir:
            tok.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "lz78_config.json"))
            assert os.path.exists(os.path.join(tmpdir, "lz78_codes.tsv"))
            assert os.path.exists(os.path.join(tmpdir, "token_bytes.pt"))
            assert os.path.exists(os.path.join(tmpdir, "token_metadata.pt"))
            assert os.path.exists(os.path.join(tmpdir, "token_ancestors.pt"))
            assert os.path.exists(os.path.join(tmpdir, "token_ancestor_depths.pt"))


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_token_metadata_structured(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        meta = tok.get_token_metadata(mode="structured")
        assert meta.shape == (tok.get_vocab_size(), 2)
        assert meta.dtype == torch.long

    def test_token_metadata_hierarchical_fallback(self, lz78_tsv_path):
        """LZ78 format has no hier parents; should fall back to structured."""
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        meta_s = tok.get_token_metadata(mode="structured")
        meta_h = tok.get_token_metadata(mode="hierarchical")
        assert torch.equal(meta_s, meta_h)

    def test_token_metadata_hierarchical_compressed(self, compressed_tsv_path):
        """Compressed format should have distinct hierarchical metadata."""
        tok = LZ78Tokenizer.from_tsv(compressed_tsv_path, "compressed")
        meta = tok.get_token_metadata(mode="hierarchical")
        assert meta.shape == (tok.get_vocab_size(), 2)

    def test_id_to_token(self, lz78_tsv_path):
        tok = LZ78Tokenizer.from_tsv(lz78_tsv_path, "lz78")
        # Code 1 = "h"
        assert tok.id_to_token(1) == "h"
        # Code 3 = "he"
        assert tok.id_to_token(3) == "he"
        # Byte fallback
        fb = tok._byte_fallback_offset + ord("x")
        assert "byte" in tok.id_to_token(fb)
        # Special token
        bos_id = tok.encode_special("<|bos|>")
        assert tok.id_to_token(bos_id) == "<|bos|>"

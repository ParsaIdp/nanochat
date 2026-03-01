"""Tests for BPE tokenizers (CharLevel, HuggingFace, RustBPE).

Covers roundtrip encoding/decoding, special tokens, edge cases (empty, Unicode),
and batch encoding.

Usage:
    python -m pytest tests/test_tokenizer.py -v
"""

import os
import tempfile

import pytest

from nanochat.tokenizer import CharLevelTokenizer, SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# CharLevelTokenizer
# ---------------------------------------------------------------------------

class TestCharLevelTokenizer:

    def setup_method(self):
        self.tok = CharLevelTokenizer()

    def test_vocab_size(self):
        assert self.tok.get_vocab_size() == 256 + len(SPECIAL_TOKENS)

    def test_roundtrip_ascii(self):
        text = "Hello, world!"
        ids = self.tok.encode(text)
        assert self.tok.decode(ids) == text

    def test_roundtrip_unicode(self):
        text = "café ñ 日本語 🎉"
        ids = self.tok.encode(text)
        assert self.tok.decode(ids) == text

    def test_empty_string(self):
        ids = self.tok.encode("")
        assert ids == []
        assert self.tok.decode([]) == ""

    def test_special_tokens(self):
        specials = self.tok.get_special_tokens()
        assert "<|bos|>" in specials
        assert "<|assistant_start|>" in specials
        bos_id = self.tok.encode_special("<|bos|>")
        assert bos_id == self.tok.get_bos_token_id()

    def test_prepend_append(self):
        text = "hi"
        bos = self.tok.encode_special("<|bos|>")
        ids = self.tok.encode(text, prepend="<|bos|>")
        assert ids[0] == bos
        assert self.tok.decode(ids[1:]) == text

    def test_batch_encode(self):
        texts = ["hello", "world"]
        result = self.tok.encode(texts)
        assert isinstance(result, list)
        assert len(result) == 2
        for i, text in enumerate(texts):
            assert self.tok.decode(result[i]) == text

    def test_batch_prepend_append(self):
        texts = ["a", "b"]
        bos = self.tok.encode_special("<|bos|>")
        result = self.tok.encode(texts, prepend="<|bos|>")
        assert all(row[0] == bos for row in result)

    def test_id_to_token_byte(self):
        # ASCII 'A' = 65
        assert self.tok.id_to_token(65) == "A"

    def test_id_to_token_special(self):
        bos_id = self.tok.encode_special("<|bos|>")
        assert self.tok.id_to_token(bos_id) == "<|bos|>"

    def test_encode_special_unknown(self):
        with pytest.raises(ValueError):
            self.tok.encode_special("<|nonexistent|>")

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tok.save(tmpdir)
            loaded = CharLevelTokenizer.from_directory(tmpdir)
            text = "roundtrip test"
            assert loaded.encode(text) == self.tok.encode(text)
            assert loaded.get_vocab_size() == self.tok.get_vocab_size()

    def test_callable(self):
        """Tokenizer can be called directly like tok(text)."""
        ids = self.tok("test")
        assert ids == self.tok.encode("test")


# ---------------------------------------------------------------------------
# RustBPETokenizer (requires rustbpe + tiktoken)
# ---------------------------------------------------------------------------

class TestRustBPETokenizer:
    """Tests for RustBPETokenizer using a small trained tokenizer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Train a tiny tokenizer on a small corpus."""
        from nanochat.tokenizer import RustBPETokenizer
        corpus = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "Hello world! This is a test of the tokenizer. " * 20,
            "Python is a great programming language. " * 20,
        ]
        self.tok = RustBPETokenizer.train_from_iterator(iter(corpus), vocab_size=300, allow_superchunk=False)

    def test_vocab_size(self):
        assert self.tok.get_vocab_size() == 300

    def test_roundtrip_simple(self):
        text = "The quick brown fox"
        ids = self.tok.encode(text)
        assert self.tok.decode(ids) == text

    def test_roundtrip_unicode(self):
        text = "café"
        ids = self.tok.encode(text)
        assert self.tok.decode(ids) == text

    def test_empty_string(self):
        ids = self.tok.encode("")
        assert ids == []

    def test_special_tokens(self):
        specials = self.tok.get_special_tokens()
        assert "<|bos|>" in specials
        bos_id = self.tok.get_bos_token_id()
        assert bos_id == self.tok.encode_special("<|bos|>")

    def test_prepend(self):
        bos = self.tok.get_bos_token_id()
        ids = self.tok.encode("hello", prepend="<|bos|>")
        assert ids[0] == bos

    def test_batch_encode(self):
        texts = ["hello", "world"]
        result = self.tok.encode(texts)
        assert len(result) == 2
        for i, text in enumerate(texts):
            assert self.tok.decode(result[i]) == text

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tok.save(tmpdir)
            from nanochat.tokenizer import RustBPETokenizer
            loaded = RustBPETokenizer.from_directory(tmpdir)
            text = "The quick brown fox"
            assert loaded.encode(text) == self.tok.encode(text)
            assert loaded.get_vocab_size() == self.tok.get_vocab_size()

    def test_render_conversation(self):
        conversation = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        ids, mask = self.tok.render_conversation(conversation)
        assert len(ids) == len(mask)
        assert len(ids) > 0
        # BOS should be first
        assert ids[0] == self.tok.get_bos_token_id()
        # Mask should have some 1s (assistant tokens) and some 0s
        assert 1 in mask
        assert 0 in mask

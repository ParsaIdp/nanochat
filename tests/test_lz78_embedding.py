"""Tests for LZ78Embedding module.

Covers all 4 modes (flat, structured, hierarchical, tuple),
output shape correctness, and init_weights.

Usage:
    python -m pytest tests/test_lz78_embedding.py -v
"""

import pytest
import torch

from nanochat.lz78_embedding import LZ78Embedding


VOCAB_SIZE = 64
N_EMBD = 32
B, T = 2, 8


def _make_metadata():
    """Create dummy (parent_code, char_byte) metadata for each token."""
    metadata = torch.zeros(VOCAB_SIZE, 2, dtype=torch.long)
    for i in range(VOCAB_SIZE):
        metadata[i, 0] = max(0, i - 1)  # parent = previous token
        metadata[i, 1] = i % 256  # char byte
    return metadata


class TestFlatMode:

    def test_output_shape(self):
        emb = LZ78Embedding("flat", VOCAB_SIZE, N_EMBD)
        ids = torch.randint(0, VOCAB_SIZE, (B, T))
        out = emb(ids)
        assert out.shape == (B, T, N_EMBD)

    def test_init_weights(self):
        emb = LZ78Embedding("flat", VOCAB_SIZE, N_EMBD)
        emb.init_weights()
        # After init, weights should not be all zeros
        assert emb.emb.weight.abs().sum() > 0

    def test_weight_property(self):
        emb = LZ78Embedding("flat", VOCAB_SIZE, N_EMBD)
        assert emb.weight is emb.emb.weight


class TestStructuredMode:

    def test_output_shape(self):
        meta = _make_metadata()
        emb = LZ78Embedding("structured", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        ids = torch.randint(0, VOCAB_SIZE, (B, T))
        out = emb(ids)
        assert out.shape == (B, T, N_EMBD)

    def test_init_weights(self):
        meta = _make_metadata()
        emb = LZ78Embedding("structured", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        emb.init_weights()
        assert emb.code_emb.weight.abs().sum() > 0
        assert emb.char_emb.weight.abs().sum() > 0

    def test_requires_metadata(self):
        with pytest.raises(AssertionError):
            LZ78Embedding("structured", VOCAB_SIZE, N_EMBD, token_metadata=None)

    def test_weight_property(self):
        meta = _make_metadata()
        emb = LZ78Embedding("structured", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        assert emb.weight is emb.code_emb.weight


class TestHierarchicalMode:

    def test_output_shape(self):
        meta = _make_metadata()
        emb = LZ78Embedding("hierarchical", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        ids = torch.randint(0, VOCAB_SIZE, (B, T))
        out = emb(ids)
        assert out.shape == (B, T, N_EMBD)

    def test_init_weights(self):
        meta = _make_metadata()
        emb = LZ78Embedding("hierarchical", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        emb.init_weights()
        assert emb.code_emb.weight.abs().sum() > 0


class TestTupleMode:

    def test_output_shape(self):
        meta = _make_metadata()
        emb = LZ78Embedding("tuple", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        ids = torch.randint(0, VOCAB_SIZE, (B, T))
        out = emb(ids)
        assert out.shape == (B, T, N_EMBD)

    def test_init_weights(self):
        meta = _make_metadata()
        emb = LZ78Embedding("tuple", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        emb.init_weights()
        assert emb.code_emb.weight.abs().sum() > 0
        assert emb.char_emb.weight.abs().sum() > 0
        # proj should be xavier initialized
        assert emb.proj.weight.abs().sum() > 0

    def test_requires_metadata(self):
        with pytest.raises(AssertionError):
            LZ78Embedding("tuple", VOCAB_SIZE, N_EMBD, token_metadata=None)

    def test_has_projection(self):
        meta = _make_metadata()
        emb = LZ78Embedding("tuple", VOCAB_SIZE, N_EMBD, token_metadata=meta)
        assert hasattr(emb, "proj")
        half = N_EMBD // 2
        assert emb.proj.weight.shape == (N_EMBD, 2 * half)


class TestUnknownMode:

    def test_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding mode"):
            LZ78Embedding("invalid_mode", VOCAB_SIZE, N_EMBD)


class TestGradientFlow:

    @pytest.mark.parametrize("mode", ["flat", "structured", "hierarchical", "tuple"])
    def test_gradients(self, mode):
        """Ensure gradients flow through all modes."""
        meta = _make_metadata()
        kwargs = {"token_metadata": meta} if mode != "flat" else {}
        emb = LZ78Embedding(mode, VOCAB_SIZE, N_EMBD, **kwargs)
        ids = torch.randint(0, VOCAB_SIZE, (B, T))
        out = emb(ids)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in emb.parameters())
        assert has_grad, f"No gradient flow in mode={mode}"

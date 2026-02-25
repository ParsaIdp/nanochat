"""Tests for prefix-smooth cross-entropy loss.

Covers gradient flow, equivalence with standard CE when pw=0,
and correct handling of masked (ignored) positions.

Usage:
    python -m pytest tests/test_prefix_loss.py -v
"""

import torch
import torch.nn.functional as F

from nanochat.prefix_loss import prefix_smooth_ce


def _make_simple_data(B=2, T=4, V=8, max_depth=3):
    """Create simple test data for prefix_smooth_ce.

    Returns logits, targets, ancestor_indices, ancestor_depths.
    Each token's ancestor chain is: [self, self-1, self-2, ...] clamped to 0.
    """
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))

    # Build simple ancestor chains: each token i has ancestors [i, i-1, ..., 0]
    ancestor_indices = torch.zeros(V, max_depth, dtype=torch.long)
    ancestor_depths = torch.ones(V, dtype=torch.long)  # depth >= 1
    for i in range(V):
        chain = list(range(i, max(i - max_depth, -1), -1))
        depth = len(chain)
        ancestor_indices[i, :depth] = torch.tensor(chain)
        ancestor_depths[i] = depth

    return logits, targets, ancestor_indices, ancestor_depths


class TestPrefixSmoothCE:

    def test_output_is_scalar(self):
        logits, targets, anc_idx, anc_dep = _make_simple_data()
        loss = prefix_smooth_ce(logits, targets, anc_idx, anc_dep, prefix_weight=1.0)
        assert loss.dim() == 0  # scalar

    def test_gradient_flow(self):
        """Loss should have gradients flowing back to logits."""
        logits, targets, anc_idx, anc_dep = _make_simple_data()
        loss = prefix_smooth_ce(logits, targets, anc_idx, anc_dep, prefix_weight=0.5)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_pw_zero_equals_standard_ce(self):
        """With prefix_weight=0, only the exact token gets mass -> standard CE."""
        B, T, V = 2, 4, 8
        torch.manual_seed(42)
        logits = torch.randn(B, T, V, requires_grad=True)
        targets = torch.randint(0, V, (B, T))

        # Ancestor chains: depth=1 for all tokens (only self)
        max_depth = 3
        ancestor_indices = torch.zeros(V, max_depth, dtype=torch.long)
        ancestor_depths = torch.ones(V, dtype=torch.long)
        for i in range(V):
            ancestor_indices[i, 0] = i

        # prefix_smooth_ce with pw=0 should behave like standard CE
        loss_prefix = prefix_smooth_ce(
            logits, targets, ancestor_indices, ancestor_depths, prefix_weight=0.0
        )

        # Standard cross-entropy
        logits_detach = logits.detach().clone().requires_grad_(True)
        loss_standard = F.cross_entropy(
            logits_detach.view(-1, V), targets.view(-1)
        )

        assert torch.allclose(loss_prefix, loss_standard, atol=1e-5), \
            f"pw=0 loss ({loss_prefix.item()}) != standard CE ({loss_standard.item()})"

    def test_masked_positions(self):
        """Targets with -1 should be ignored."""
        B, T, V = 1, 6, 8
        logits = torch.randn(B, T, V, requires_grad=True)
        targets = torch.randint(0, V, (B, T))
        # Mask half the positions
        targets[0, ::2] = -1

        max_depth = 2
        ancestor_indices = torch.zeros(V, max_depth, dtype=torch.long)
        ancestor_depths = torch.ones(V, dtype=torch.long)
        for i in range(V):
            ancestor_indices[i, 0] = i

        loss = prefix_smooth_ce(logits, targets, ancestor_indices, ancestor_depths)
        loss.backward()

        # Loss should still be valid (not NaN or Inf)
        assert torch.isfinite(loss)
        assert logits.grad is not None

    def test_all_masked(self):
        """If all positions are masked, loss should be 0."""
        B, T, V = 1, 4, 8
        logits = torch.randn(B, T, V)
        targets = torch.full((B, T), -1, dtype=torch.long)

        max_depth = 2
        ancestor_indices = torch.zeros(V, max_depth, dtype=torch.long)
        ancestor_depths = torch.ones(V, dtype=torch.long)
        for i in range(V):
            ancestor_indices[i, 0] = i

        loss = prefix_smooth_ce(logits, targets, ancestor_indices, ancestor_depths)
        assert loss.item() == 0.0

    def test_higher_pw_shifts_distribution(self):
        """Higher prefix_weight should produce different (generally lower) loss
        when ancestor tokens have high logits."""
        B, T, V = 1, 4, 8
        max_depth = 3
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        targets = torch.randint(1, V, (B, T))

        ancestor_indices = torch.zeros(V, max_depth, dtype=torch.long)
        ancestor_depths = torch.ones(V, dtype=torch.long)
        for i in range(V):
            chain = list(range(i, max(i - max_depth, -1), -1))
            depth = len(chain)
            ancestor_indices[i, :depth] = torch.tensor(chain)
            ancestor_depths[i] = depth

        loss_low = prefix_smooth_ce(logits, targets, ancestor_indices, ancestor_depths, prefix_weight=0.0)
        loss_high = prefix_smooth_ce(logits, targets, ancestor_indices, ancestor_depths, prefix_weight=1.0)
        # They should at least be different
        assert not torch.allclose(loss_low, loss_high)

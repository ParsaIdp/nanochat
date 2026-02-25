"""
Prefix-smoothed cross-entropy loss for nanochat.

Instead of a one-hot label vector, the target for each position is a soft
distribution that places mass on both the exact next token AND all tokens
that are prefixes of it.

Example: if the next token is "hello" (id=1000) and BPE/LZ78 also has
tokens "hell"(500), "hel"(200), "he"(50), "h"(5), the label vector has
non-zero entries at all five, normalized to sum to 1.

The `prefix_weight` parameter controls how much mass goes to prefixes:
  - prefix_weight=1.0: uniform over exact + all prefixes
  - prefix_weight=0.5: prefixes get half the weight of the exact token
  - prefix_weight=0.1: mild smoothing toward prefixes
"""

import torch
import torch.nn.functional as F


def prefix_smooth_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ancestor_indices: torch.Tensor,
    ancestor_depths: torch.Tensor,
    prefix_weight: float = 1.0,
) -> torch.Tensor:
    """
    Cross-entropy with soft labels that include prefix ancestors.

    For each target token, the label vector has:
    - Weight 1.0 at the exact target token
    - Weight `prefix_weight` at each prefix ancestor
    Then normalized to sum to 1.

    Args:
        logits: (B, T, V) float tensor
        targets: (B, T) long tensor, -1 = ignore
        ancestor_indices: (V, max_depth) long tensor — ancestor_indices[tok][0] = tok itself,
                          ancestor_indices[tok][1] = longest prefix, etc.
        ancestor_depths: (V,) long tensor — number of valid ancestors per token
        prefix_weight: weight given to each prefix ancestor (exact token always gets 1.0)

    Returns:
        scalar loss (mean over valid tokens)
    """
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V)       # (N, V)
    targets_flat = targets.view(-1)        # (N,)

    # Mask for valid (non-ignored) targets
    valid = targets_flat >= 0
    safe_targets = targets_flat.clamp(min=0)

    # Look up ancestor chains and depths for each target token
    chains = ancestor_indices[safe_targets]    # (N, max_depth)
    depths = ancestor_depths[safe_targets]     # (N,)
    max_depth = chains.shape[1]

    # Build weight vector: 1.0 for exact (position 0), prefix_weight for ancestors (positions 1+)
    w = torch.full((max_depth,), prefix_weight, dtype=torch.float32, device=logits.device)
    w[0] = 1.0  # exact token

    # Mask out padding positions (beyond each token's actual depth)
    depth_range = torch.arange(max_depth, device=logits.device).unsqueeze(0)  # (1, D)
    valid_mask = depth_range < depths.unsqueeze(1)  # (N, D)
    weights = w.unsqueeze(0) * valid_mask.float()  # (N, D)

    # Normalize weights to sum to 1 per token (making it a valid distribution)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, D)

    # Compute soft cross-entropy: -sum(weight_j * log_softmax(logits)[ancestor_j])
    log_probs = F.log_softmax(logits_flat, dim=-1)  # (N, V)
    ancestor_log_probs = log_probs.gather(1, chains)  # (N, max_depth)
    per_token_loss = -(weights * ancestor_log_probs).sum(dim=1)  # (N,)

    # Mask out ignored positions and average
    per_token_loss = per_token_loss * valid.float()
    return per_token_loss.sum() / valid.sum().clamp(min=1)

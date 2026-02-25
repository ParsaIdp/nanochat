# LZ78 Prefix Label Loss

## Idea

In standard next-token prediction, the target is a **one-hot** vector: the model gets full credit for predicting the exact correct token, zero credit otherwise.

But LZ78 tokens have **tree structure** — every token is a prefix extension of its parent. If the correct token is "hello" (code 500), its LZ78 ancestors might be:

```
code 500: "hello"   ← target
code 200: "hell"    ← parent
code  50: "hel"     ← grandparent
code  10: "he"      ← great-grandparent
code   3: "h"       ← root-level
```

The idea: **give partial credit for predicting any prefix of the correct token**. Instead of a one-hot target at code 500, use a soft target that distributes weight across the entire ancestor chain.

## Why This Makes Sense

1. **Partial matches are meaningful.** If the model predicts "hell" instead of "hello", that's much better than predicting "xyz". Standard cross-entropy treats both as equally wrong.

2. **Richer gradient signal.** Early in training, the model is unlikely to predict the exact correct token from a 32K vocabulary. Prefix labels provide useful gradients even when the exact prediction is wrong — the model can learn to climb the LZ78 tree incrementally.

3. **Natural curriculum.** Short prefixes (near root) are easier to predict than long specific tokens (deep in tree). The loss function rewards getting the broad strokes right before demanding fine details.

4. **Works with any tokenizer.** LZ78 has a native tree structure. For BPE, prefix relationships can be derived by finding all BPE tokens whose byte sequence is a prefix of the target token. The hypothesis is that LZ78's deeper tree structure (mean depth 4.7) provides richer prefix signal than BPE's shallower prefix relationships (mean depth 1.72).

## Label Construction

For each target token with code `c`, walk up the LZ78 tree to root, collecting the ancestor chain `[c, parent(c), parent(parent(c)), ..., root]`.

### Option A: Uniform over ancestors

Distribute weight equally across all ancestors (excluding root):

```
target[c]          = 1/depth
target[parent(c)]  = 1/depth
target[grandp(c)]  = 1/depth
...
```

### Option B: Exponential decay (recommended)

Weight decreases exponentially as you move up the tree. The correct token gets the most weight, with geometrically decaying credit for prefixes:

```
target[c]          = w^0 * Z   (most weight)
target[parent(c)]  = w^1 * Z
target[grandp(c)]  = w^2 * Z
...
```

Where `w ∈ (0, 1)` is a decay factor (e.g., 0.5) and `Z` normalizes to sum to 1.

With `w = 0.5` and depth 5: weights are `[0.516, 0.258, 0.129, 0.065, 0.032]`.

### Option C: Linear decay

```
target[c]          = depth / sum
target[parent(c)]  = (depth-1) / sum
...
```

### Option D: Interpolation with one-hot

Mix a fraction `α` of the prefix distribution with `(1-α)` of the standard one-hot:

```
target = (1 - α) * one_hot(c) + α * prefix_distribution(c)
```

This lets you tune how much to rely on the tree structure vs. standard training.

## Implementation Sketch

### 1. Precompute ancestor chains

During tokenizer setup, for each code `c`, compute and store the full ancestor chain:

```python
# ancestors[c] = [(code, depth), (parent_code, depth-1), ..., (root_ancestor, 1)]
ancestors = {}
for code in range(1, max_code + 1):
    chain = []
    cur = code
    while cur != 0:  # walk up to root
        chain.append(cur)
        cur = parent_codes[cur]
    ancestors[code] = chain  # [code, parent, grandparent, ..., root_child]
```

Save as `token_ancestors.pt`: list of variable-length ancestor chains.

### 2. Modified loss function

Replace `F.cross_entropy(logits, targets)` with a soft cross-entropy that uses the prefix distribution:

```python
def prefix_label_loss(logits, targets, ancestor_chains, decay=0.5):
    """
    logits: (B*T, vocab_size) - model output
    targets: (B*T,) - target token IDs
    ancestor_chains: precomputed per-token ancestor lists
    decay: exponential decay factor
    """
    B_T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)  # (B*T, V)

    loss = torch.zeros(B_T, device=logits.device)
    for i in range(B_T):
        t = targets[i].item()
        if t < 0:
            continue  # skip padding
        chain = ancestor_chains[t]  # [t, parent, grandparent, ...]
        depth = len(chain)
        # Compute weights: w^0, w^1, w^2, ... normalized
        weights = torch.tensor([decay ** j for j in range(depth)], device=logits.device)
        weights = weights / weights.sum()
        # Soft cross-entropy: -sum(weight_j * log_prob[chain_j])
        for j, code in enumerate(chain):
            loss[i] -= weights[j] * log_probs[i, code]

    return loss.mean()
```

### 3. Efficient batched version

The naive loop above is too slow. For efficiency:

```python
def prefix_label_loss_batched(logits, targets, ancestor_matrix, weight_matrix):
    """
    ancestor_matrix: (vocab_size, max_depth) - padded ancestor chains, 0-padded
    weight_matrix: (vocab_size, max_depth) - precomputed normalized weights, 0-padded
    """
    # Gather ancestor chains for all targets at once
    chains = ancestor_matrix[targets]      # (B*T, max_depth)
    weights = weight_matrix[targets]       # (B*T, max_depth)

    log_probs = F.log_softmax(logits, dim=-1)  # (B*T, V)

    # Gather log probs for all ancestors
    ancestor_log_probs = log_probs.gather(1, chains)  # (B*T, max_depth)

    # Weighted sum
    loss = -(weights * ancestor_log_probs).sum(dim=1)  # (B*T,)
    return loss.mean()
```

This is fully batched and GPU-friendly. The `ancestor_matrix` and `weight_matrix` are precomputed once and stored as buffers.

### 4. Maximum tree depth

For a 32K LZ78 dictionary, the maximum tree depth is bounded by the construction process. Typical max depths are 10-20 characters. The `ancestor_matrix` would be `(32K, max_depth)` — very small.

## Experiment Plan

| # | Run Name | Loss | Decay | Notes |
|---|----------|------|-------|-------|
| 1 | `lz78-32k-onehot-c4-d12` | Standard CE | — | Baseline (same as `lz78-32k-flat-c4-d12`) |
| 2 | `lz78-32k-prefix-d0.5-c4-d12` | Prefix label | 0.5 | Moderate decay |
| 3 | `lz78-32k-prefix-d0.3-c4-d12` | Prefix label | 0.3 | Faster decay (more weight on exact match) |
| 4 | `lz78-32k-prefix-d0.7-c4-d12` | Prefix label | 0.7 | Slower decay (more weight on prefixes) |
| 5 | `lz78-32k-prefix-interp0.2-c4-d12` | Interpolated | α=0.2 | 80% one-hot + 20% prefix |

Compare BPB across all runs. The hypothesis is that prefix labels provide a better training signal, especially early in training, leading to faster convergence and/or lower final BPB.

## Key Questions

1. **Does partial credit help or hurt?** The richer signal might speed up learning, but it could also encourage the model to be "lazy" and predict generic prefixes instead of specific tokens.

2. **What decay factor is best?** Too high = too much credit for vague prefixes. Too low = almost identical to one-hot.

3. **Does it interact with embedding mode?** Combining prefix labels with structured embeddings (which also exploit tree structure) might be complementary or redundant.

4. **Training dynamics.** Does the loss curve look different? Do prefix-labeled models converge faster to the same BPB, or reach a different final BPB?

## Implementation

**Implemented and submitted.** All code and jobs are live.

### Files

| File | Description |
|------|-------------|
| `nanochat/prefix_loss.py` | `prefix_label_loss()`, `prefix_interp_loss()`, and `prefix_bce_loss()` — batched GPU-efficient loss functions using ancestor chains. `build_prefix_weights()` computes normalized decay weights. |
| `nanochat/lz78_tokenizer.py` | Added `_save_ancestor_data()` — walks each token's parent chain to root, saves `token_ancestors.pt` (indices) and `token_ancestor_depths.pt` (depths). |
| `scripts/base_train.py` | Added `--loss_mode {standard,prefix,prefix_interp,prefix_bce}`, `--prefix_decay`, `--prefix_alpha` flags. Training loop uses prefix loss when configured; **validation BPB always uses standard CE** for fair comparison. Works with both LZ78 and BPE tokenizers. |
| `scripts/submit_prefix_ablations.sh` | Submits LZ78 prefix loss experiments (6 runs). |
| `scripts/bpe_generate_ancestors.py` | Generates ancestor data for BPE tokenizer by byte-level prefix matching. For each BPE token, finds all other BPE tokens whose byte sequence is a prefix. |
| `scripts/submit_bpe_prefix_ablations.sh` | Submits BPE prefix loss experiments (3 runs). |

### Multi-hot BCE Loss (`prefix_bce`)

An alternative to the soft cross-entropy approach. Instead of distributing probability mass, uses binary cross-entropy with a multi-hot target vector:
- **1** at the exact target token AND all tokens that are byte-level prefixes of it
- **0** everywhere else
- Uses `pos_weight = V / mean_num_positives` to balance the severe class imbalance (~5 positives vs ~V negatives)
- Uses sigmoid (independent per-token classification) instead of softmax

### Ancestor Data Stats

| Tokenizer | Max Depth | Mean Depth | Tokens with Ancestors |
|-----------|-----------|------------|----------------------|
| LZ78 32K | 19 | 4.7 | 32,271 |
| FreqGated 32K | 32 | — | 32,651 |
| Trie2x 44K | 10 | — | 44,428 |
| BPE 512 | 5 | 1.72 | 245 |

### BPE Ancestor Generation

For BPE, prefix relationships are derived from byte sequences rather than tree structure:
1. Get the UTF-8 byte sequence of each BPE token
2. Build a trie from all byte sequences
3. For each token, walk through the trie collecting all tokens that are byte-level prefixes
4. Save as `token_ancestors.pt` and `token_ancestor_depths.pt` (same format as LZ78)

Example: BPE token " the" → ancestors [" the", " th", " t", " "]

Script: `python -m scripts.bpe_generate_ancestors --tokenizer_dir /path/to/tokenizer`

### Submitted Runs — LZ78 (SLURM)

| # | Job ID | Run Name | Loss Mode | Decay | Alpha |
|---|--------|----------|-----------|-------|-------|
| 0 | 1700628 | `lz78-32k-standard-c4-d12` | standard | — | — |
| 1 | 1700629 | `lz78-32k-prefix-d0.5-c4-d12` | prefix | 0.5 | — |
| 2 | 1700630 | `lz78-32k-prefix-d0.3-c4-d12` | prefix | 0.3 | — |
| 3 | 1700631 | `lz78-32k-prefix-d0.7-c4-d12` | prefix | 0.7 | — |
| 4 | 1700632 | `lz78-32k-prefix-interp0.2-c4-d12` | prefix_interp | 0.5 | 0.2 |
| 5 | 1700647 | `lz78-32k-prefix-bce-c4-d12` | prefix_bce | — | — |

### Submitted Runs — BPE (SLURM)

| # | Job ID | Run Name | Loss Mode | Decay | Alpha |
|---|--------|----------|-----------|-------|-------|
| baseline | 1700603 | `bpe-50k-flat-c4-d12` | standard | — | — |
| 1 | 1700643 | `bpe-prefix-d0.5-c4-d12` | prefix | 0.5 | — |
| 2 | 1700644 | `bpe-prefix-interp0.2-c4-d12` | prefix_interp | 0.5 | 0.2 |
| 3 | 1700645 | `bpe-prefix-bce-c4-d12` | prefix_bce | — | — |

All log to wandb project `nanochat` (entity: goodarzilab).

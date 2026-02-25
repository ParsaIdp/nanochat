# LZ78 Tokenizer Experiments — Project Status

## Overview

Comparing LZ78 tokenizer variants (from weezl) against BPE baseline on C4 language modeling, with ablations on embedding strategy.

---

## Tokenizer Types

| Type | TSV Source | Vocab (raw) | Vocab (with fallback+special) | Description |
|------|-----------|-------------|-------------------------------|-------------|
| **BPE** | nanochat RustBPE | ~50K | 50,304 (padded) | Baseline — standard byte-level BPE |
| **Normal LZ78** | `lz78_dict_32k.tsv` | 32,012 codes | 32,272 | Standard LZ78 pre-trained dictionary. Codes 1..32012 are sequential. |
| **Freq Gated** | `freq_gated_full_32000.tsv` | 32,392 codes | 32,652 | LZ78 with frequency gating — only frequent patterns kept. Original codes are sparse (up to 90M), remapped to dense 1..32392. |
| **Compressed Trie** | `compressed_full_2x_32000.tsv` | 44,169 codes | 44,429 | Patricia trie (collapsed single-child chains). Output codes 1..44169 are sequential. |

All LZ78 TSVs live at `/large_storage/goodarzilab/parsaidp/weezl/dictionaries/`.

### Vocab Layout (all LZ78 tokenizers)

```
[0]              reserved (root / padding, never emitted)
[1..N]           dictionary codes from TSV
[N+1..N+256]     byte fallback tokens (for bytes 0x00..0xFF)
[N+257]          <|bos|>
[N+258]          <|eos|>
[N+259]          <|pad|>
```

Total vocab = N + 260.

### Encoding

All LZ78 variants use the same encoding algorithm:
1. Convert input text to UTF-8 bytes
2. Build a byte-level trie from all dictionary patterns
3. Greedy longest-match: walk the trie consuming bytes, emit the code of the longest matched pattern
4. Byte fallback: any unmatched byte emits token `N + 1 + byte_value`

This is consistent across all three TSV formats — the only difference is which patterns are in the dictionary.

### Code Remapping

The Freq Gated dictionary has sparse original codes (range 1..90M for only 32K entries). During tokenizer construction, these are remapped to a dense range `1..32392` by sorting original codes and assigning sequential new codes. Parent codes are also remapped. This is transparent to downstream code.

---

## Embedding Strategies

| Mode | Embedding | Parameters | Description |
|------|-----------|------------|-------------|
| **flat** | `nn.Embedding(V, d)` | V × d | Standard — each token ID gets its own learned vector |
| **structured** | `code_emb(parent) + char_emb(byte)` | V × d + 256 × d | Decomposes each token into its LZ78 tree parent code + extension character byte, embeds each separately and sums |
| **hierarchical** | Same as structured | V × d + 256 × d | Uses the trie parent's code instead of the LZ78 tree parent. Only differs from structured for compressed trie (in practice, identical for this particular trie since collapsed edges mean LZ78 parent == trie parent) |

### Metadata

For structured/hierarchical embeddings, each token stores `(parent_code, char_byte)`:
- **parent_code**: the dictionary code of this token's parent in the LZ78 tree (0 = root)
- **char_byte**: the last byte of this token's UTF-8 pattern (the "extension character")
- For byte fallback tokens: parent=0, char=byte_value
- For special tokens: parent=0, char=0

### Embedding Size Analysis (depth=12 model)

Model body: n_layer=12, n_embd=768, n_head=6 → ~85M transformer params (untied wte/lm_head).

| Tokenizer | Vocab (padded) | wte params | lm_head params | Body | Total | Emb % |
|-----------|---------------|------------|----------------|------|-------|-------|
| BPE 50K | 50,304 | 38.6M | 38.6M | 85M | 162M | 47% |
| LZ78 32K | 32,320 | 24.8M | 24.8M | 85M | 135M | 37% |
| FreqGated 32K | 32,704 | 25.1M | 25.1M | 85M | 135M | 37% |
| Trie2x 44K | 44,480 | 34.2M | 34.2M | 85M | 153M | 44% |

---

## Run Matrix — 8 Experiments on C4

All runs: depth=12, device_batch_size=32, total_batch_size=524288, 2× GPU, `--core_metric_every=-1`.

All log to **wandb project `nanochat`**, entity `goodarzilab`.

| # | Run Name | Tokenizer | Vocab | Embedding | SLURM Job |
|---|----------|-----------|-------|-----------|-----------|
| 1 | `bpe-50k-flat-c4-d12` | BPE | 50K | flat | 1700603 |
| 2 | `lz78-32k-flat-c4-d12` | Normal LZ78 | 32K | flat | 1700604 |
| 3 | `lz78-32k-struct-c4-d12` | Normal LZ78 | 32K | structured | 1700605 |
| 4 | `freqgated-32k-flat-c4-d12` | Freq Gated | 32K | flat | 1700606 |
| 5 | `freqgated-32k-struct-c4-d12` | Freq Gated | 32K | structured | 1700607 |
| 6 | `trie2x-44k-flat-c4-d12` | Compressed Trie | 44K | flat | 1700608 |
| 7 | `trie2x-44k-struct-c4-d12` | Compressed Trie | 44K | structured | 1700609 |
| 8 | `trie2x-44k-hier-c4-d12` | Compressed Trie | 44K | hierarchical | 1700610 |

### Pipeline

1. **Tokenizer setup** (local, fast): `scripts/lz78_setup_tokenizer.py` builds tokenizer from TSV, saves to disk
2. **Pre-tokenization** (SLURM CPU jobs 1700600-1700602): `scripts/lz78_pretokenize.py` tokenizes C4 train/val `.txt` → `.npy` shards
3. **Training** (SLURM GPU jobs, depend on pretok): `scripts/base_train.py` with `--tokenizer_type=lz78 --data_mode=pretokenized`

BPE baseline (run 1) uses online tokenization from parquet files (no pretok dependency).

### Key Paths

```
/large_storage/goodarzilab/parsaidp/weezl/
├── dictionaries/                          # Source TSV files
│   ├── lz78_dict_32k.tsv
│   ├── freq_gated_full_32000.tsv
│   └── compressed_full_2x_32000.tsv
├── c4_train.txt                           # C4 training data (~1GB)
├── c4_val.txt                             # C4 validation data (~21MB)
└── lz78_ablations/
    ├── tokenizers/                        # Built tokenizers
    │   ├── lz78_32k/
    │   ├── freqgated_32k/
    │   └── trie2x_44k/
    └── data/                              # Pre-tokenized .npy shards
        ├── lz78_32k/{train,val}/
        ├── freqgated_32k/{train,val}/
        └── trie2x_44k/{train,val}/
```

---

## Implementation

### New Files

| File | Description |
|------|-------------|
| `nanochat/lz78_tokenizer.py` | Unified `LZ78Tokenizer` class — handles both TSV formats (lz78/compressed), byte-level trie encoding, sparse code remapping, structured/hierarchical metadata generation |
| `nanochat/lz78_embedding.py` | `LZ78Embedding` module — flat/structured/hierarchical modes. Forward: `token_ids → (B, T, n_embd)` |
| `nanochat/lz78_dataloader.py` | Pre-tokenized `.npy` shard reader with distributed training support and resume via `state_dict` |
| `scripts/lz78_setup_tokenizer.py` | CLI to build + save tokenizer from TSV, with encode/decode roundtrip verification |
| `scripts/lz78_pretokenize.py` | Tokenize C4 `.txt` files to `.npy` shards (1M tokens/shard default) |
| `scripts/lz78_pretokenize.slurm` | SLURM job for CPU pre-tokenization (train + val splits) |
| `scripts/lz78_train.slurm` | SLURM job for 2-GPU LZ78 training |
| `scripts/bpe_train.slurm` | SLURM job for 2-GPU BPE baseline training |
| `scripts/submit_lz78_ablations.sh` | Master script: setup tokenizers → submit pretok → submit 8 training jobs with dependencies |
| `nanochat/prefix_loss.py` | Prefix label loss functions: `prefix_label_loss` (soft CE), `prefix_interp_loss` (interpolated), `prefix_bce_loss` (multi-hot BCE) |
| `scripts/submit_prefix_ablations.sh` | Submit LZ78 prefix loss experiments (6 runs) |
| `scripts/bpe_generate_ancestors.py` | Generate ancestor data for BPE tokenizer by byte-level prefix matching |
| `scripts/submit_bpe_prefix_ablations.sh` | Submit BPE prefix loss experiments (3 runs) |

### Modified Files

| File | Changes |
|------|---------|
| `nanochat/gpt.py` | Added `embedding_mode` and `token_metadata_path` to `GPTConfig`. `GPT.__init__` creates `LZ78Embedding` when mode is structured/hierarchical. `init_weights` loads metadata from disk after `to_empty`. Updated `estimate_flops` to count all embedding params. |
| `scripts/base_train.py` | Added CLI flags: `--tokenizer_type {rustbpe,lz78}`, `--tokenizer_dir`, `--embedding_mode {flat,structured,hierarchical}`, `--data_mode {online,pretokenized}`, `--pretokenized_dir`. Conditional tokenizer loading and dataloader selection. |

### Evaluation

All runs use **bits per byte (BPB)** as the primary metric, which normalizes for different vocab sizes — making cross-tokenizer comparison fair. Each token's loss contribution is weighted by the number of bytes that token represents (`token_bytes.pt`).

---

---

## Prefix Label Loss Experiments

Exploiting prefix/tree structure in the loss function. Instead of one-hot targets, distributes label weight across tokens that are prefixes of the target. See [full writeup](lz78_prefix_labels.md).

### Loss Modes

| Mode | Description |
|------|-------------|
| **standard** | Standard one-hot cross-entropy (baseline) |
| **prefix** | Soft cross-entropy: distributes weight across ancestor chain with exponential decay. Weight for position j = decay^j, normalized to sum to 1. |
| **prefix_interp** | Interpolation: (1-α) × standard CE + α × prefix loss |
| **prefix_bce** | Multi-hot binary cross-entropy: target vector has 1s at the exact token AND all tokens that are byte-level prefixes. Uses sigmoid per-token with pos_weight to balance ~5 positives vs ~V negatives. |

Key: validation BPB always uses standard CE (one-hot) for fair comparison. Prefix loss is training-only.

### LZ78 Prefix Runs (6 runs)

All use LZ78 32K tokenizer, flat embedding, depth=12, C4 data. Ancestor chains from LZ78 tree structure (parent → grandparent → ... → root).

| # | Job ID | Run Name | Loss | Decay | Alpha |
|---|--------|----------|------|-------|-------|
| 0 | 1700628 | `lz78-32k-standard-c4-d12` | Standard CE | — | — |
| 1 | 1700629 | `lz78-32k-prefix-d0.5-c4-d12` | Prefix | 0.5 | — |
| 2 | 1700630 | `lz78-32k-prefix-d0.3-c4-d12` | Prefix | 0.3 | — |
| 3 | 1700631 | `lz78-32k-prefix-d0.7-c4-d12` | Prefix | 0.7 | — |
| 4 | 1700632 | `lz78-32k-prefix-interp0.2-c4-d12` | Interp | 0.5 | 0.2 |
| 5 | 1700647 | `lz78-32k-prefix-bce-c4-d12` | BCE | — | — |

### BPE Prefix Runs (3 runs)

Same prefix loss idea applied to BPE tokenizer. Ancestor chains derived from byte-level prefix matching: for each BPE token, find all other BPE tokens whose byte sequence is a prefix of this token's byte sequence.

Example: token " the" (id=261) has ancestors [" the", " th", " t", " "] — all valid BPE tokens whose bytes are a prefix.

BPE tokenizer: default (`/home/parsaidp/.cache/nanochat/tokenizer`, 512 vocab). Ancestor stats: max depth 5, mean depth 1.72, 245/512 tokens with prefix ancestors.

| # | Job ID | Run Name | Loss | Decay | Alpha |
|---|--------|----------|------|-------|-------|
| baseline | 1700603 | `bpe-50k-flat-c4-d12` | Standard CE | — | — |
| 1 | 1700643 | `bpe-prefix-d0.5-c4-d12` | Prefix | 0.5 | — |
| 2 | 1700644 | `bpe-prefix-interp0.2-c4-d12` | Interp | 0.5 | 0.2 |
| 3 | 1700645 | `bpe-prefix-bce-c4-d12` | BCE | — | — |

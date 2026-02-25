# nanochat Architecture

## Pipeline Overview

nanochat follows a standard LLM training pipeline, from raw text to a chat-capable model:

```
tokenizer --> pretrain --> midtrain --> SFT --> RL --> inference
   |             |            |          |       |        |
tok_train.py  base_train.py mid_train.py chat_sft.py chat_rl.py chat_cli.py
lz78_tok_train.py                                       chat_web.py
```

1. **Tokenizer training** -- Train a BPE or LZ78 tokenizer on a text corpus.
2. **Pretraining** -- Train a GPT model from scratch on FineWeb-Edu with next-token prediction.
3. **Midtraining** -- Continued pretraining on domain-specific or curated data.
4. **Supervised finetuning (SFT)** -- Train on conversational data with a supervision mask that targets only assistant turns.
5. **Reinforcement learning (RL)** -- GRPO-style policy gradient on GSM8K math problems with correctness reward.
6. **Inference** -- KV-cached autoregressive generation with optional calculator tool use.

---

## Module Map

### Core library (`nanochat/`)

| Module | Description |
|--------|-------------|
| `common.py` | Shared utilities: colored logging, DDP setup (`compute_init`/`compute_cleanup`), device autodetection, `print0` for rank-0 printing. |
| `configurator.py` | CLI configuration override mechanism via `exec()` -- parses `--key=value` arguments into global variables. |
| `tokenizer.py` | BPE tokenizer implementations: `CharLevelTokenizer` (byte-level baseline), `HuggingFaceTokenizer`, and `RustBPETokenizer` (default, trains via rustbpe, infers via tiktoken). |
| `lz78_tokenizer.py` | LZ78 trie-based tokenizer with greedy longest-match encoding, byte fallback, and structured metadata for embeddings. |
| `lz78_embedding.py` | Structured embedding strategies for LZ78 tokens that exploit parent-child trie relationships. |
| `lz78_dataloader.py` | Pre-tokenized data loading for LZ78 -- reads `.npy` shard files instead of tokenizing on the fly. |
| `lz78_trainer.py` | LZ78 dictionary training via the `weezl` Rust library. |
| `prefix_loss.py` | Prefix-smoothed cross-entropy loss that distributes label mass across a token and all of its prefix ancestors. |
| `gpt.py` | GPT model architecture with RoPE, GQA, QK-norm, ReluSquared MLP, RMSNorm, logit soft-capping, and untied embeddings. |
| `engine.py` | Inference engine: KV cache management, batched autoregressive generation, and calculator tool use via special tokens. |
| `checkpoint_manager.py` | Model/optimizer/metadata checkpoint save/load, plus convenience functions for nanochat's directory layout. |
| `dataloader.py` | Streaming tokenizing data loader that reads parquet shards, tokenizes in batches, and yields `(input, target)` tensor pairs. |
| `dataset.py` | FineWeb-Edu parquet dataset utilities: shard listing, on-demand downloading, and batched document iteration. |
| `adamw.py` | Distributed AdamW optimizer with fused kernels and per-rank sharding. |
| `muon.py` | Muon optimizer (MomentUm Orthogonalized by Newton-schulz) for attention and MLP weight matrices. |
| `core_eval.py` | CORE benchmark evaluation (from the DCLM paper): multiple-choice, schema, and language-modeling task types. |
| `loss_eval.py` | Bits-per-byte (BPB) evaluation that normalizes loss by token byte length for vocab-size-independent comparison. |
| `execution.py` | Sandboxed Python execution for tool use: process isolation, timeout, memory limits, and dangerous-function blocking. |
| `report.py` | Training report generation for logging experiment results. |

### Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `tok_train.py` | Train a BPE tokenizer from FineWeb-Edu text. |
| `tok_eval.py` | Evaluate tokenizer compression ratio. |
| `lz78_tok_train.py` | Train an LZ78 tokenizer dictionary via the weezl Rust library. |
| `lz78_setup_tokenizer.py` | Build an `LZ78Tokenizer` from a TSV dictionary file. |
| `lz78_pretokenize.py` | Pre-tokenize the dataset with an LZ78 tokenizer into `.npy` shards. |
| `bpe_generate_ancestors.py` | Generate BPE ancestor chains (prefix relationships) for prefix-smooth loss. |
| `base_train.py` | Pretrain a base GPT model on FineWeb-Edu. |
| `base_loss.py` | Evaluate a base model's bits-per-byte on the validation set. |
| `base_eval.py` | Evaluate a base model on the CORE benchmark. |
| `mid_train.py` | Midtraining (continued pretraining on curated data). |
| `chat_sft.py` | Supervised finetuning on conversation data. |
| `chat_rl.py` | Reinforcement learning via GRPO on GSM8K. |
| `chat_eval.py` | Evaluate a chat model. |
| `chat_cli.py` | Interactive CLI chat interface. |
| `chat_web.py` | Web UI chat interface. |

---

## Tokenizer System

nanochat supports two tokenizer families: **BPE** and **LZ78**. Both share a common interface that the rest of the codebase programs against.

### Common interface

All tokenizers implement:

```python
train_from_iterator(text_iter, vocab_size) -> tokenizer   # classmethod
from_directory(path) -> tokenizer                         # classmethod
encode(text, prepend=, append=) -> list[int]
decode(ids) -> str
get_vocab_size() -> int
get_special_tokens() -> set[str]
get_bos_token_id() -> int
encode_special(name) -> int
id_to_token(token_id) -> str
save(directory)
```

### BPE tokenizers

Three implementations are provided in `tokenizer.py`:

- **CharLevelTokenizer** -- Byte-level baseline. Vocab = 256 bytes + special tokens. No training required. Useful as a compression baseline.
- **HuggingFaceTokenizer** -- BPE training and inference via the HuggingFace `tokenizers` library.
- **RustBPETokenizer** (default) -- Training via `rustbpe` (fast Rust BPE), inference via `tiktoken`. This is the recommended tokenizer. It also provides chat-specific helpers: `render_conversation()` produces token IDs with a supervision mask, and `render_for_completion()` primes the model for autoregressive completion during RL.

All BPE tokenizers use a GPT-4-style split pattern that pre-chunks text by regex before applying byte-pair merges. The pattern limits digit runs to 1-2 characters (instead of GPT-4's 1-3) to avoid wasting tokens at smaller vocab sizes.

**Special tokens:** `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`.

### LZ78 tokenizer

Implemented in `lz78_tokenizer.py`. Instead of BPE merges, it builds a byte-level trie from a dictionary (trained via the `weezl` Rust library) and encodes text via greedy longest-match traversal. Unmatched bytes fall through to a byte-fallback range.

**Vocab layout:**
- `[0]` -- reserved root/padding (never emitted)
- `[1..N]` -- dictionary codes from the trained trie
- `[N+1..N+256]` -- byte fallback tokens
- `[N+257..]` -- special tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`)

The LZ78 tokenizer maintains parent-child metadata per token (parent code and last-char byte), enabling **structured embeddings** (`lz78_embedding.py`) that exploit trie structure, and **prefix-smooth loss** (`prefix_loss.py`) that places label mass on prefix ancestors.

---

## Data Pipeline

Data flows from raw text on HuggingFace to GPU training batches through these stages:

### 1. Dataset (`dataset.py`)

The pretraining corpus is **FineWeb-Edu 100B** (shuffled), hosted as 1823 parquet shards on HuggingFace. Shards are downloaded on demand. The last shard is reserved for validation; all others are training data.

### 2. Data loader (`dataloader.py`)

The `tokenizing_distributed_data_loader_with_state` generator:

1. Iterates over parquet files, reading row groups (each ~1024 documents).
2. Each DDP rank reads a strided subset of row groups (`rg_idx = rank, rank + world_size, ...`).
3. Documents are batch-tokenized (default 128 at a time, 4 threads) with the BOS token prepended.
4. Tokens stream into a `deque` buffer.
5. When enough tokens accumulate (`B * T + 1`), a batch is popped: `inputs = tokens[:-1]`, `targets = tokens[1:]`, reshaped to `(B, T)`.
6. Tensors are pinned to CPU memory and transferred to GPU asynchronously.

The loader returns a `state_dict` with each batch (parquet file index + row group index) enabling approximate resume after checkpoint recovery.

### 3. LZ78 pre-tokenized path (`lz78_dataloader.py`)

For LZ78 tokenizers (where on-the-fly tokenization is slower), data can be pre-tokenized into `.npy` shards via `lz78_pretokenize.py`, then loaded directly by `pretokenized_data_loader_with_state`.

---

## Model Architecture

The GPT model is defined in `gpt.py` via the `GPTConfig` dataclass and `GPT` module.

### Configuration defaults

```python
GPTConfig(
    sequence_len = 1024,   # max context length
    vocab_size   = 50304,  # vocabulary size
    n_layer      = 12,     # number of transformer blocks
    n_head       = 6,      # number of query attention heads
    n_kv_head    = 6,      # number of KV heads (for GQA)
    n_embd       = 768,    # embedding dimension
)
```

In practice, model dimensions are derived from a single `depth` parameter: `n_layer = depth`, `n_embd = depth * 64`, `n_head = ceil(n_embd / 128)`.

### Transformer block

Each `Block` applies pre-norm attention followed by pre-norm MLP, with residual connections:

```
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

**RMSNorm** is parameter-free (`F.rms_norm` with no learnable scale/bias).

### Attention (`CausalSelfAttention`)

- Separate Q, K, V projections (no bias).
- **Rotary positional embeddings (RoPE)** applied to Q and K, with base theta 10000.
- **QK normalization**: RMSNorm applied to Q and K after RoPE, before dot product.
- **Group-Query Attention (GQA)**: `n_kv_head` can be less than `n_head`, with KV heads shared across query head groups.
- Causal masking via `F.scaled_dot_product_attention` with `is_causal=True` during training.
- During inference with KV cache, handles three cases: full causal (Tq == Tk), single-token decode (Tq == 1), and chunked prefix attention.

### MLP

- Two linear layers (no bias), expansion ratio 4x.
- **ReluSquared** activation: `relu(x)^2` -- a sparsifying activation.

### Logit soft-capping

Output logits are squashed via `softcap * tanh(logits / softcap)` with `softcap = 15`, preventing extreme logit values while preserving gradients.

### Weight initialization

- Token embedding: Normal(0, 1), cast to bf16 on CUDA.
- LM head: Normal(0, 0.001).
- Attention Q/K/V and MLP up-projection: Uniform with std = 1/sqrt(n_embd).
- Attention output projection and MLP down-projection: zeros (residual stream starts clean).
- Embedding and LM head weights are **untied** (separate parameters).

### Vocab size padding

The vocab size is padded to a multiple of 64 for DDP compatibility and GPU efficiency.

---

## Training Pipeline

### Optimizers

nanochat uses a **split optimizer** strategy (`GPT.setup_optimizers`):

- **AdamW** (via `DistAdamW` in DDP) for the embedding and LM head:
  - Embedding LR: 0.2 (default), unembedding LR: 0.004.
  - LR scaled by `(n_embd / 768)^{-0.5}` for model size transfer.
  - Betas: (0.8, 0.95), eps: 1e-10.
- **Muon** (via `DistMuon` in DDP) for all transformer block linear layers:
  - LR: 0.02 (default), momentum: 0.95 (warmed from 0.85 over 300 steps).
  - Applies SGD with momentum, then orthogonalizes the update via Newton-Schulz iteration (5 steps). The update is replaced with the nearest orthogonal matrix, computed stably in bf16.
  - Aspect-ratio scaling: step size multiplied by `sqrt(max(1, rows/cols))`.

### Distributed training

- **DDP** via `torchrun`. Each rank processes a strided subset of parquet row groups.
- `DistMuon` performs its own gradient synchronization: `reduce_scatter` for gradient averaging, then `all_gather` to replicate updated weights.
- `DistAdamW` handles distributed AdamW synchronization similarly.
- Gradient accumulation: `total_batch_size / (device_batch_size * seq_len * world_size)` micro-steps per optimizer step.
- Gradient clipping at 1.0 (configurable).
- Mixed precision: bf16 autocast on CUDA, with loss computation in fp32.

### Learning rate schedule

- **Warmup** phase (default 0% of iterations): linear ramp from 0 to peak LR.
- **Constant** phase: full learning rate.
- **Warmdown** phase (default 20% of iterations): linear decay from peak to `final_lr_frac * peak` (default 0).

### Training horizon (Chinchilla scaling)

Three ways to set the training length, in order of precedence:

1. `num_iterations` -- explicit step count.
2. `target_flops` -- compute the steps needed to reach a target FLOP budget.
3. `target_param_data_ratio` -- compute steps to maintain a fixed tokens-to-parameters ratio (Chinchilla-optimal is 20).

The data:param ratio and total FLOPs are logged for scaling law analysis.

### Checkpointing (`checkpoint_manager.py`)

Checkpoints consist of three files per save:
- `model_{step:06d}.pt` -- model state dict (saved by rank 0 only).
- `optim_{step:06d}_rank{rank}.pt` -- optimizer state (each rank saves its own, since Muon shards momentum buffers).
- `meta_{step:06d}.json` -- metadata: model config, user config, dataloader state for resume, loop state (min val BPB, EMA loss, wall time).

The directory structure groups checkpoints by model tag (e.g., `base_checkpoints/d20/`). Loading auto-detects the largest model and latest step if not specified.

### Compilation

The model is compiled with `torch.compile(model, dynamic=False)` for training. The uncompiled model is kept for evaluation (where input shapes vary) and checkpoint saving.

---

## Evaluation System

### CORE benchmark (`core_eval.py`)

Implements the CORE metric from the DCLM paper (arXiv:2406.11794). Supports three task types:

- **Multiple choice** -- All options share a common prefix; the model scores each continuation by average cross-entropy loss. Lowest loss wins.
- **Schema** -- Contexts vary but the continuation is shared; scored by the common suffix's loss under each context.
- **Language modeling** -- The model must predict a continuation token-by-token; evaluated by exact-match of argmax predictions.

Few-shot examples are sampled per-item with a deterministic seed. Evaluation is distributed across DDP ranks (each rank evaluates a strided subset, results are `all_reduce`-summed).

### Bits-per-byte evaluation (`loss_eval.py`)

The `evaluate_bpb` function computes a tokenizer-agnostic loss metric:

```
BPB = sum(nats) / (ln(2) * sum(bytes))
```

Each target token's loss is weighted by how many bytes that token represents (via the `token_bytes.pt` tensor). Special tokens (0 bytes) and ignored tokens (`target == -1`) are excluded. This allows fair comparison across different vocabulary sizes.

---

## Inference Engine

### KV cache (`engine.py: KVCache`)

The KV cache stores key and value tensors for all transformer layers in a single contiguous tensor of shape `(num_layers, 2, batch_size, num_kv_heads, seq_len, head_dim)`.

- **Lazy initialization**: allocated on first `insert_kv` call to match dtype/device.
- **Dynamic growth**: if the sequence exceeds the allocated length, the cache is extended by at least 1024 positions (rounded up to a multiple of 1024).
- **Position tracking**: `pos` auto-advances after the last transformer layer inserts.
- **Prefill + expand**: a batch-1 prefill cache can be cloned and expanded to `num_samples` rows for parallel generation via `prefill()`.

### Generation loop (`engine.py: Engine`)

The `Engine` class drives autoregressive generation:

1. **Prefill**: run the full prompt through the model with a batch-1 KV cache.
2. **Clone**: replicate the KV cache across `num_samples` rows.
3. **Decode loop**: on each step, sample the next token (with temperature and top-k), update per-row state, and yield `(token_column, token_masks)`.
4. **Termination**: rows complete on `<|assistant_end|>` or `<|bos|>`.

### Tool use: calculator

During generation, the engine tracks a simple state machine per row:

1. When `<|python_start|>` is generated, the engine begins collecting expression tokens.
2. On `<|python_end|>`, the collected tokens are decoded and evaluated via `use_calculator()`.
3. If evaluation succeeds, the result is injected as forced tokens: `<|output_start|> result_tokens <|output_end|>`.

The calculator supports:
- **Pure math**: expressions containing only `0-9 * + - / . ( ) space` (power operator `**` is blocked).
- **String operations**: currently only `.count()` on string literals.
- **Safety**: expressions are validated against a character allowlist and a blocklist of dangerous patterns (`__`, `import`, `exec`, `eval`, etc.). Evaluation runs with `{"__builtins__": {}}` and a 3-second timeout via `SIGALRM`.

### Sandboxed execution (`execution.py`)

For more complex code execution (e.g., during evaluation), `execution.py` provides process-isolated sandboxing: each execution runs in a forked process with timeout enforcement, 256MB memory limits, stdout/stderr capture, and dangerous standard library functions disabled. This is used for evaluating model-generated code, not for the calculator tool.

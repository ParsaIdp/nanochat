"""
Engine for efficient inference of nanochat models.

Provides token-level generation with KV-cache management and optional tool use
(calculator). The engine operates purely on token-id sequences and knows nothing
about tokenization itself.

Key components:
    - KVCache: manages the key/value cache across transformer layers.
    - Engine: drives autoregressive generation with batched sampling and
      streaming output.
    - Calculator helpers: sandboxed eval of simple math / string expressions
      invoked via special tokens during generation.
"""

import signal
import warnings
from collections import deque
from contextlib import contextmanager, nullcontext
from typing import Generator, Optional

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init

# -----------------------------------------------------------------------------
# Calculator tool helpers

# Characters permitted in pure math expressions.
MATH_CHARS: str = "0123456789*+-/.() "

# Characters permitted in string-operation expressions (e.g. "abc".count("a")).
ALLOWED_CHARS: str = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789'\"()._ "
)

# Substrings that must never appear in calculator expressions.
DANGEROUS_PATTERNS: list[str] = [
    "__", "import", "exec", "eval", "compile", "open", "file",
    "input", "raw_input", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
]


@contextmanager
def timeout(duration: int, formula: str) -> Generator[None, None, None]:
    def timeout_handler(signum: int, frame: object) -> None:
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    # NOTE: signal.SIGALRM is Unix-only; this will not work on Windows.
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula: str, max_time: int = 3) -> Optional[object]:
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr: str) -> Optional[object]:
    """Evaluate a Python expression safely.

    Supports pure math expressions and simple string operations like
    ``"hello".count("l")``.  Returns ``None`` when the expression is
    rejected or evaluation fails.
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression
    if all(x in MATH_CHARS for x in expr):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    if not all(x in ALLOWED_CHARS for x in expr):
        return None

    # Disallow dangerous patterns
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in DANGEROUS_PATTERNS):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int, num_layers: int) -> None:
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape: tuple[int, ...] = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache: Optional[torch.Tensor] = None
        self.pos: int = 0  # current position in time in the cache

    def reset(self) -> None:
        self.pos = 0

    def get_pos(self) -> int:
        return self.pos

    def prefill(self, other: "KVCache") -> None:
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"

        # Extract dimensions explicitly
        self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = self.kv_shape
        other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = other.kv_shape

        # Validate dimensions
        assert self_layers == other_layers, f"Layer count mismatch: {self_layers} != {other_layers}"
        assert self_kv == other_kv, f"K/V dimension mismatch: {self_kv} != {other_kv}"
        assert self_heads == other_heads, f"Head count mismatch: {self_heads} != {other_heads}"
        assert self_head_dim == other_head_dim, f"Head dim mismatch: {self_head_dim} != {other_head_dim}"

        # Batch size can be expanded (other can be 1, self can be larger)
        assert self_batch == other_batch or other_batch == 1, f"Batch size mismatch: {self_batch} vs {other_batch} (other must be 1 or equal)"

        # Sequence length: self must be longer than other
        assert self_seq >= other_seq, f"Sequence length mismatch: {self_seq} < {other_seq}"

        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits: torch.Tensor, rng: torch.Generator, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    """Per-row state tracking during generation."""

    def __init__(self, current_tokens: Optional[list[int]] = None) -> None:
        self.current_tokens: list[int] = current_tokens or []
        self.forced_tokens: deque[int] = deque()
        self.in_python_block: bool = False
        self.python_expr_tokens: list[int] = []
        self.completed: bool = False

class Engine:

    def __init__(self, model: object, tokenizer: object) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int] | None = None,
        num_samples: int = 1,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Generator[tuple[list[int], list[int]], None, None]:
        """Run autoregressive generation with batched sampling.

        Performs a single batch-1 prefill of the prompt, clones the KV cache
        across ``num_samples`` rows, then streams ``(token_column, token_masks)``
        tuples on each step. Pass either ``tokens`` or ``input_embeds``.
        """
        if (tokens is None) == (input_embeds is None):
            raise ValueError("Pass exactly one of `tokens` or `input_embeds`.")
        if tokens is not None:
            assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
            
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        def get_special(s):
            try:
                return self.tokenizer.encode_special(s)
            except (ValueError, KeyError):
                return None  # token not in this tokenizer's vocabulary
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") or get_special("<|eos|>")  # fallback for LZ78
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        seq_len = len(tokens) if tokens is not None else input_embeds.size(1)
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=seq_len,
            **kv_model_kwargs,
        )
        if tokens is not None:
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        else:
            logits = self.model.forward(None, kv_cache=kv_cache_prefill, input_embeds=input_embeds.unsqueeze(0))
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (seq_len + (max_tokens or 0)) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState((tokens or []).copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(
        self, tokens: list[int], num_samples: int = 1, **kwargs: object
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Run non-streaming batch generation and return final token sequences.

        Returns ``(results, masks)`` where each entry is a list of token-id
        lists.  Terminal tokens (assistant_end, bos) are excluded from results.
        """
        # Use <|assistant_end|> if available, fall back to <|eos|>
        try:
            assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        except (ValueError, KeyError):
            try:
                assistant_end = self.tokenizer.encode_special("<|eos|>")
            except (ValueError, KeyError):
                assistant_end = None
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0] # only print out the first row
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")

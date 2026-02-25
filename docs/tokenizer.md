# Tokenizer

nanochat uses GPT-4-style byte-level BPE tokenization. The tokenizer module (`nanochat/tokenizer.py`) provides three interchangeable implementations and a set of chat-specific helpers for rendering conversations into supervised training sequences.

## Implementations

### RustBPETokenizer (default)

Trains with [rustbpe](https://github.com/karpathy/rustbpe) (a fast Rust BPE implementation) and wraps the resulting merge table in a [tiktoken](https://github.com/openai/tiktoken) `Encoding` for efficient inference. This is the tokenizer used throughout nanochat.

- **Training**: `rustbpe.Tokenizer` learns merges from a text iterator.
- **Inference**: the merge table is converted to `tiktoken.Encoding`, which handles fast batch encoding with `num_threads`.
- **Serialization**: the tiktoken `Encoding` object is pickled to `tokenizer.pkl`.

### HuggingFaceTokenizer

Wraps the HuggingFace `tokenizers` library. Supports both training and loading from disk (`tokenizer.json`). Useful as an alternative backend, but not the default.

### CharLevelTokenizer

A trivial byte-level tokenizer with no merges. Each of the 256 byte values maps directly to a token ID. Useful as a baseline for comparing compression ratios and model performance against BPE tokenizers at various vocab sizes.

- Vocab size is always `256 + len(SPECIAL_TOKENS)` (currently 265).
- "Training" is a no-op.
- Saves a `char_tokenizer.marker` file and a `token_bytes.pt` tensor for pipeline compatibility.

## Special Tokens

All tokenizers share the same set of special tokens, defined in `SPECIAL_TOKENS`:

| Token | Purpose |
|---|---|
| `<\|bos\|>` | Beginning of sequence. Prepended to every document. |
| `<\|user_start\|>` / `<\|user_end\|>` | Delimit user messages during finetuning. |
| `<\|assistant_start\|>` / `<\|assistant_end\|>` | Delimit assistant messages during finetuning. |
| `<\|python_start\|>` / `<\|python_end\|>` | Delimit python tool-call code within assistant turns. |
| `<\|output_start\|>` / `<\|output_end\|>` | Delimit python REPL output returned to the assistant. |

Special tokens are appended **after** the BPE vocabulary during training, so they occupy the highest token IDs.

## Split Pattern

The pre-tokenization regex split pattern is shared across all BPE tokenizers:

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

This is the GPT-4 split pattern with one modification: `\p{N}{1,2}` instead of `\p{N}{1,3}`. This limits number tokens to at most 2 digits (0-99) instead of 3 (0-999), saving vocabulary budget for smaller models.

## Common Interface

All three tokenizers expose the same API:

```python
# Construction
tok = RustBPETokenizer.train_from_iterator(text_iter, vocab_size)  # train new
tok = RustBPETokenizer.from_directory("out/tokenizer")              # load from disk

# Encoding / decoding
ids = tok.encode("hello world")                          # str -> list[int]
ids = tok.encode(["hello", "world"])                     # list[str] -> list[list[int]]
ids = tok.encode("hello", prepend="<|bos|>")             # prepend a special token
text = tok.decode(ids)                                   # list[int] -> str

# Vocabulary info
tok.get_vocab_size()         # int
tok.get_special_tokens()     # set[str]
tok.get_bos_token_id()       # int
tok.encode_special("<|bos|>") # str -> int (single special token)
tok.id_to_token(42)          # int -> str

# Persistence
tok.save("out/tokenizer")
```

The `encode` method accepts optional `prepend` and `append` arguments, which can be either a special token name (str) or a raw token ID (int). These are inserted at the boundaries of each encoded sequence.

## Chat Rendering

`RustBPETokenizer` provides three additional methods for chat finetuning and RL:

### `render_conversation(conversation, max_tokens=2048)`

Converts a conversation dict into a flat token sequence with a supervision mask.

**Input format** (conversation dict):
```python
{
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        ...
    ]
}
```

**Returns** `(ids, mask)` where:
- `ids`: `list[int]` - the full token sequence.
- `mask`: `list[int]` - same length, where `1` = supervised (model trains on this), `0` = unsupervised.

**Token layout**:
```
<|bos|> <|user_start|> ...user tokens... <|user_end|> <|assistant_start|> ...assistant tokens... <|assistant_end|> ...
  0            0              0               0                0                   1                    1
```

Key details:
- Messages must strictly alternate user/assistant, starting with user.
- If the first message has `role: "system"`, it is merged into the following user message.
- User tokens and all delimiting special tokens have `mask=0` (unsupervised).
- Assistant text tokens and their surrounding `<|assistant_start|>`/`<|assistant_end|>` have `mask=1` (supervised), except `<|assistant_start|>` which is `mask=0`.
- For tool use, `<|python_start|>` / `<|python_end|>` and the code inside are supervised (`mask=1`), while `<|output_start|>` / `<|output_end|>` and the output inside are unsupervised (`mask=0`) since the output comes from the Python runtime at inference time.
- The sequence is truncated to `max_tokens` to prevent OOMs.

### `render_for_completion(conversation)`

Used during **Reinforcement Learning**. Takes a conversation where the last message is from the assistant, removes that last assistant message, renders the rest with `render_conversation`, then appends `<|assistant_start|>` to prime the model for generating a completion.

Returns `ids` only (no mask needed for RL).

### `visualize_tokenization(ids, mask, with_token_id=False)`

Debug helper that returns a colorized string (ANSI escape codes) showing each token pipe-separated, colored green for supervised (`mask=1`) and red for unsupervised (`mask=0`). Pass `with_token_id=True` to also show the numeric token ID after each token.

## Convenience Functions

```python
from nanochat.tokenizer import get_tokenizer, get_token_bytes

# Load the default RustBPETokenizer from <base_dir>/tokenizer/
tok = get_tokenizer()
tok = get_tokenizer(tokenizer_path="path/to/custom/tokenizer")

# Load the token_bytes.pt tensor (shape: [vocab_size, max_token_len], dtype: uint8)
# Written by tok_train.py, used for embedding visualization.
token_bytes = get_token_bytes(device="cuda")
```

## Disk Layout

After training (`tok_train.py`), the tokenizer directory contains:

| File | Tokenizer | Description |
|---|---|---|
| `tokenizer.pkl` | RustBPETokenizer | Pickled tiktoken `Encoding` object |
| `tokenizer.json` | HuggingFaceTokenizer | HuggingFace tokenizer JSON |
| `char_tokenizer.marker` | CharLevelTokenizer | Marker file indicating char-level tokenizer |
| `token_bytes.pt` | All | `torch.Tensor` of shape `[vocab_size, max_token_len]` mapping each token ID to its byte representation |

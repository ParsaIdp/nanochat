# LZ78 Tokenizer

## 1. Overview

The LZ78 tokenizer is an alternative to Byte Pair Encoding (BPE) for subword tokenization in language models. Instead of learning merge rules from corpus statistics, LZ78 builds a dictionary on-the-fly using the classic Lempel-Ziv 78 compression algorithm. The tokenizer scans text left-to-right, extending the current match in a prefix tree (trie) until it encounters a character that creates a new entry. Each new dictionary entry consists of a parent code (the longest previously seen prefix) and a single extending character.

This dictionary-based approach produces a tree-structured vocabulary where every token is a single-character extension of an existing token, enabling structured embedding strategies that exploit parent-child relationships.

## 2. Training Strategies

Seven training strategies are available, each offering different trade-offs between dictionary quality, compression ratio, and compatibility with downstream embedding modes.

### standard

Basic LZ78 dictionary building. Scans the training text left-to-right, building a prefix tree one character at a time. Every novel prefix encountered becomes a new dictionary entry. Training stops when the dictionary reaches the target vocabulary size.

### frequency_gated

LZ78 with periodic eviction of low-frequency leaves. During dictionary construction, leaves that have not been referenced frequently enough are pruned at regular intervals, freeing slots for more useful entries. This is the **best-performing strategy**, achieving **1.0999 BPB with chunking** -- the closest any LZ78 variant gets to BPE performance.

### multi_round

Trains a large dictionary at 4x the target vocabulary size, counts how frequently each entry is used across the corpus, then prunes down to the target size by keeping the most frequently used entries. The pruning preserves tree structure by retaining all ancestors of kept nodes.

### cost_adjusted

Similar to `multi_round`, but penalizes dictionary entries that require many ancestor slots. An entry's effective value is its frequency divided by the number of ancestor nodes it forces into the dictionary. This encourages a flatter, more slot-efficient vocabulary.

### smart_prune

Produces an output-only vocabulary. Prefixes that serve only as internal trie nodes are kept in the trie for matching but do not count toward the vocabulary budget. Only entries that are actually emitted during tokenization consume a vocab slot. Must be used with the `tokenize_smart()` method at inference time.

### flat_prune

Produces a flat dictionary with no tree constraint. Entries are independent strings rather than parent-child pairs in a trie. This breaks the tree structure entirely, so structured/hierarchical embedding modes cannot be used. Must be used with the `tokenize_flat()` method at inference time.

### compressed

Builds a Patricia trie from a previously trained dictionary. Single-child chains in the trie are collapsed into single nodes with multi-character edge labels, eliminating prefix chain overhead. Achieves approximately **6.0x node compression** compared to the standard trie representation. Uses a different TSV format (see Section 3).

## 3. TSV Formats

Trained dictionaries are serialized to TSV files in one of two formats depending on the training strategy.

### LZ78 Format

Used by: `standard`, `frequency_gated`, `multi_round`, `cost_adjusted`, `smart_prune`, `flat_prune`.

```
code\tparent_code\tchar\tpattern
```

| Column        | Description                                          |
|---------------|------------------------------------------------------|
| `code`        | Integer dictionary code for this entry               |
| `parent_code` | Code of the parent entry in the trie (0 = root)      |
| `char`        | The single character extending the parent             |
| `pattern`     | The full string this entry represents (for debugging) |

Example:

```
1	0	h	h
2	0	e	e
3	1	e	he
4	3	l	hel
5	4	l	hell
6	5	o	hello
```

### Compressed Trie Format

Used by: `compressed`.

```
node_idx\tedge_label\tparent_idx\toutput_code\tpattern
```

| Column        | Description                                                   |
|---------------|---------------------------------------------------------------|
| `node_idx`    | Index of this node in the Patricia trie                       |
| `edge_label`  | Multi-character edge label from parent to this node           |
| `parent_idx`  | Index of the parent node in the Patricia trie                 |
| `output_code` | The vocabulary code emitted when this node matches (or -1)    |
| `pattern`     | The full string this node represents (for debugging)          |

The compressed format eliminates prefix chain overhead by collapsing single-child chains. For example, the chain `h -> he -> hel -> hell -> hello` becomes a single node with edge label `hello`, achieving a typical **6.0x compression ratio** in node count.

## 4. Vocab Layout

The full vocabulary is laid out as follows:

```
[0]            reserved (root/padding, never emitted)
[1..N]         dictionary codes from TSV
[N+1..N+256]   byte fallback tokens (for bytes 0..255)
[N+257]        <|bos|>
[N+258]        <|eos|>
[N+259]        <|pad|>
```

- **Code 0** is reserved for the trie root and used as a padding token. It is never emitted during tokenization.
- **Codes 1 through N** correspond to the dictionary entries loaded from the TSV file.
- **Byte fallback tokens** (N+1 through N+256) handle any byte that does not appear in the dictionary, ensuring complete coverage of all possible inputs. Byte fallback for byte value `b` is at index `N + 1 + b`.
- **Special tokens** are placed at the end: beginning-of-sequence, end-of-sequence, and padding.

## 5. Embedding Modes

Four embedding strategies are available, each defining how token IDs are mapped to dense vectors before being fed into the transformer.

### flat

Standard lookup embedding. Each token ID maps to a single learned vector.

```
embed = Embedding[token_id]
```

This is the simplest mode and performs **best at convergence**. It ignores the tree structure of the LZ78 dictionary entirely.

### structured

Additive decomposition using the LZ78 parent-child structure. The embedding is the sum of a parent embedding and a character embedding.

```
embed = ParentEmbed[parent_code] + CharEmbed[char_code]
```

This mode exploits the fact that every LZ78 entry is a one-character extension of its parent, encouraging the model to learn compositional representations.

### hierarchical

Similar to `structured`, but uses the trie parent rather than the LZ78 parent. In a standard trie, the parent of a node is found by removing the last character of its string. In the LZ78 tree, the parent is the longest prefix seen before this entry was created, which may differ from the trie parent.

### tuple

Concatenation followed by linear projection. The parent and character embeddings are concatenated and projected down to the model dimension.

```
embed = Linear(concat(ParentEmbed[parent_code], CharEmbed[char_code]))
```

This mode gives the model more capacity to learn non-additive interactions between parent and character components.

## 6. Usage Examples

### Train a Dictionary

```bash
python scripts/lz78_tok_train.py \
    --strategy frequency_gated \
    --vocab_size 32000
```

This reads the training corpus and produces a TSV file containing the learned LZ78 dictionary.

### Build a Tokenizer from a TSV

```bash
python scripts/lz78_setup_tokenizer.py \
    --tsv_path path/to/dictionary.tsv \
    --tsv_format lz78
```

For compressed dictionaries, use `--tsv_format compressed`.

### Pre-tokenize a Dataset

```bash
python scripts/lz78_pretokenize.py
```

Pre-tokenizes the training data using the built tokenizer and saves the result for faster training iteration.

### Use in Python Code

```python
from nanochat.tokenizer import LZ78Tokenizer

# Load from a TSV file
tokenizer = LZ78Tokenizer.from_tsv(
    tsv_path="path/to/dictionary.tsv",
    tsv_format="lz78"
)

# Encode text to token IDs
token_ids = tokenizer.encode("Hello, world!")

# Decode token IDs back to text
text = tokenizer.decode(token_ids)
```

## 7. Compression Benchmarks

Results from weezl experiments comparing LZ78 tokenizer variants against BPE.

| Tokenizer Variant              | BPB    | Gap to BPE |
|---------------------------------|--------|------------|
| BPE (baseline)                  | 0.9433 | --         |
| FreqGated + chunking            | 1.0999 | +16.6%     |
| FreqGated (no chunking)         | 1.1757 | +24.6%     |

Key findings:

- **BPE achieves 0.9433 BPB**, the best overall result.
- **FreqGated with chunking achieves 1.0999 BPB**, the best LZ78 result, sitting 16.6% behind BPE.
- **Chunking reduces the BPB gap** from 24.6% to 16.6%, a substantial improvement that comes from breaking the input into fixed-size chunks before LZ78 encoding, which prevents the dictionary from over-specializing on long, rare prefixes.
- **Patricia trie compression** achieves approximately **6.0x node compression** by collapsing single-child chains, significantly reducing memory usage and lookup overhead without affecting tokenization quality.

## 8. Compact Output Vocab

The `compact_output_vocab()` function identifies and marks dictionary entries that serve only as internal prefixes -- entries that are never emitted as final tokens during tokenization. These prefix-only entries arise because the LZ78 trie requires all ancestors of a kept node to exist in the tree, even if those ancestors are never the longest match at any position in the corpus.

By marking these entries, the effective output vocabulary size is reduced. This has two benefits:

1. **Smaller softmax layer**: The language model's output projection only needs columns for entries that can actually be emitted, reducing parameter count and computation.
2. **Better training signal**: The model does not waste capacity learning to predict tokens that never appear in the training data.

Entries marked as prefix-only remain in the trie for matching purposes but are excluded from the output vocabulary. This is the mechanism underlying the `smart_prune` training strategy, which takes this idea further by not counting prefix-only entries toward the vocabulary budget at all.

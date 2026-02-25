"""
Generate ancestor (prefix) data for a BPE tokenizer.

For each BPE token, finds all other BPE tokens whose byte sequence is a
prefix of this token's byte sequence. This gives us the same kind of
ancestor chain as LZ78, just derived from byte-level prefix relationships.

Example: if BPE has tokens "h"(5), "he"(200), "hel"(500), "hello"(1000):
  - Token "hello" (1000): ancestors = [1000, 500, 200, 5]
  - Token "hel"   (500):  ancestors = [500, 200, 5]
  - Token "he"    (200):  ancestors = [200, 5]
  - Token "h"     (5):    ancestors = [5]

Saves token_ancestors.pt and token_ancestor_depths.pt to the tokenizer dir.

Usage:
    python -m scripts.bpe_generate_ancestors --tokenizer_dir /path/to/tokenizer
"""

import os
import argparse
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ancestor data for BPE tokenizer")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to BPE tokenizer directory")
    args = parser.parse_args()

    from nanochat.tokenizer import RustBPETokenizer
    tokenizer = RustBPETokenizer.from_directory(args.tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    special_tokens = tokenizer.get_special_tokens()

    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    # Step 1: Get byte sequence for each token
    token_to_bytes = {}
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_tokens:
            token_to_bytes[token_id] = None  # skip special tokens
        else:
            token_to_bytes[token_id] = token_str.encode("utf-8")

    # Step 2: Build a trie from all BPE token byte sequences
    # Each node stores: children dict, and the token_id if this node is a complete token
    class TrieNode:
        __slots__ = ['children', 'token_id']
        def __init__(self):
            self.children = {}
            self.token_id = -1  # -1 means no token ends here

    root = TrieNode()
    for token_id, byte_seq in token_to_bytes.items():
        if byte_seq is None:
            continue
        node = root
        for b in byte_seq:
            if b not in node.children:
                node.children[b] = TrieNode()
            node = node.children[b]
        node.token_id = token_id

    # Step 3: For each token, walk through the trie collecting all prefix token IDs
    # The ancestor chain is: [self, longest_prefix, next_longest_prefix, ..., shortest_prefix]
    ancestor_chains = {}
    max_depth = 0

    for token_id, byte_seq in token_to_bytes.items():
        if byte_seq is None:
            ancestor_chains[token_id] = [token_id]  # special tokens: just themselves
            continue

        # Walk the trie, collecting all tokens that are prefixes
        prefixes = []  # list of (length, token_id) for all prefix matches
        node = root
        for i, b in enumerate(byte_seq):
            if b not in node.children:
                break
            node = node.children[b]
            if node.token_id >= 0:
                prefixes.append((i + 1, node.token_id))

        # Sort by length descending: [self, longest_prefix, ..., shortest_prefix]
        prefixes.sort(key=lambda x: -x[0])
        chain = [tid for _, tid in prefixes]

        if not chain or chain[0] != token_id:
            # Token itself should always be in the chain (it's in the trie)
            chain = [token_id]

        ancestor_chains[token_id] = chain
        max_depth = max(max_depth, len(chain))

    print(f"Max ancestor depth: {max_depth}")

    # Compute stats
    depths = [len(ancestor_chains[tid]) for tid in range(vocab_size)]
    mean_depth = sum(depths) / len(depths)
    tokens_with_prefixes = sum(1 for d in depths if d > 1)
    print(f"Mean ancestor depth: {mean_depth:.2f}")
    print(f"Tokens with prefix ancestors: {tokens_with_prefixes} / {vocab_size}")

    # Step 4: Build padded tensors
    ancestor_indices = torch.zeros(vocab_size, max_depth, dtype=torch.long)
    ancestor_depths = torch.zeros(vocab_size, dtype=torch.long)

    for token_id in range(vocab_size):
        chain = ancestor_chains[token_id]
        depth = len(chain)
        ancestor_depths[token_id] = depth
        ancestor_indices[token_id, :depth] = torch.tensor(chain, dtype=torch.long)

    # Save
    anc_path = os.path.join(args.tokenizer_dir, "token_ancestors.pt")
    dep_path = os.path.join(args.tokenizer_dir, "token_ancestor_depths.pt")
    torch.save(ancestor_indices, anc_path)
    torch.save(ancestor_depths, dep_path)
    print(f"Saved token_ancestors.pt: shape {ancestor_indices.shape}")
    print(f"Saved token_ancestor_depths.pt: shape {ancestor_depths.shape}")

    # Show some examples
    print("\nExample ancestor chains:")
    examples = []
    for token_id in range(vocab_size):
        if ancestor_depths[token_id] >= 3:
            examples.append(token_id)
        if len(examples) >= 5:
            break

    for token_id in examples:
        chain = ancestor_chains[token_id]
        chain_strs = []
        for tid in chain:
            s = tokenizer.decode([tid])
            chain_strs.append(f"{tid}:{repr(s)}")
        print(f"  Token {token_id} ({repr(tokenizer.decode([token_id]))}): [{', '.join(chain_strs)}]")


if __name__ == "__main__":
    main()

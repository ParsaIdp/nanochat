"""
LZ78 tokenizer for nanochat.

Handles two TSV dictionary formats:
- lz78: Standard LZ78 / Freq-Gated dictionaries (code, parent_code, char, pattern)
- compressed: Compressed trie / Patricia trie dictionaries (node_idx, edge_label, parent_idx, output_code, pattern)

Both are encoded via a byte-level trie with greedy longest-match, plus byte fallback
for unmatched bytes.

Vocab layout:
  [0]            reserved (root / padding, never emitted)
  [1..N]         dictionary codes from TSV
  [N+1..N+256]   byte fallback tokens (for bytes 0..255)
  [N+257]        <|bos|>
  [N+258]        <|eos|>
  [N+259]        <|pad|>
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import regex
import torch

LZ78_SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]

# GPT-4 style regex chunking pattern (non-possessive for Python regex compatibility)
LZ78_CHUNKING_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class _TrieNode:
    """Byte-level trie node for greedy longest-match encoding."""
    __slots__ = ['children', 'code']

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}  # byte (int 0..255) -> _TrieNode
        self.code: int = -1  # token code at this node, -1 if none


class LZ78Tokenizer:
    """Byte-level LZ78 tokenizer with greedy longest-match trie encoding.

    Supports both standard LZ78 and compressed trie dictionary formats,
    with byte fallback for unmatched bytes and special token handling.
    """

    def __init__(
        self,
        root: _TrieNode,
        code_to_bytes: dict[int, bytes],
        max_code: int,
        special_tokens: dict[str, int],
        token_parent_codes: dict[int, int],
        token_char_bytes: dict[int, int],
        hier_parent_codes: Optional[dict[int, int]] = None,
    ) -> None:
        """
        Args:
            root: Root of the byte-level trie.
            code_to_bytes: Maps code -> raw bytes for decoding.
            max_code: Largest dictionary code.
            special_tokens: Special token name -> id mapping.
            token_parent_codes: Code -> parent code (for structured embedding).
            token_char_bytes: Code -> last char byte (for structured embedding).
            hier_parent_codes: Code -> hierarchical parent code, or None.
        """
        self._root = root
        self._code_to_bytes = code_to_bytes
        self._max_code = max_code
        self._byte_fallback_offset = max_code + 1
        self._special_tokens = special_tokens
        self._special_tokens_reverse = {v: k for k, v in special_tokens.items()}
        self._vocab_size = max_code + 1 + 256 + len(LZ78_SPECIAL_TOKENS)
        self._token_parent_codes = token_parent_codes
        self._token_char_bytes = token_char_bytes
        self._hier_parent_codes = hier_parent_codes
        self._chunking_pattern = None
        self.bos_token_id = special_tokens["<|bos|>"]

    def __repr__(self) -> str:
        return (
            f"LZ78Tokenizer(vocab_size={self._vocab_size}, "
            f"max_code={self._max_code}, "
            f"byte_fallback_offset={self._byte_fallback_offset})"
        )

    def set_chunking(self, enabled: bool) -> None:
        """Enable/disable GPT-4-style regex chunking before encoding."""
        if enabled:
            self._chunking_pattern = regex.compile(LZ78_CHUNKING_PATTERN)
        else:
            self._chunking_pattern = None

    # -------------------------------------------------------------------------
    # Construction from TSV
    # -------------------------------------------------------------------------

    @classmethod
    def from_tsv(cls, tsv_path: str, tsv_format: str) -> LZ78Tokenizer:
        """
        Build tokenizer from a TSV dictionary file.

        Args:
            tsv_path: Path to the TSV file.
            tsv_format: "lz78" or "compressed".
        """
        if tsv_format not in ("lz78", "compressed"):
            raise ValueError(
                f"Unknown tsv_format {tsv_format!r}, expected 'lz78' or 'compressed'"
            )

        if tsv_format == "lz78":
            return cls._from_lz78_tsv(tsv_path)
        else:
            return cls._from_compressed_tsv(tsv_path)

    @classmethod
    def _from_lz78_tsv(cls, tsv_path: str) -> LZ78Tokenizer:
        """Parse LZ78 / FreqGated TSV: code, parent_code, char, pattern."""
        raw_entries = []  # list of (orig_code, orig_parent_code, pattern_bytes)
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#') or not line:
                    continue
                parts = line.split('\t')
                code = int(parts[0])
                parent_code = int(parts[1])
                # pattern is always the last field (safe even if char contains tab)
                pattern = parts[-1]
                pattern_bytes = pattern.encode('utf-8')
                raw_entries.append((code, parent_code, pattern_bytes))

        # Check if codes are sparse; if so, remap to dense range [1..N]
        orig_codes = [c for c, _, _ in raw_entries]
        is_sequential = orig_codes == list(range(1, len(orig_codes) + 1))

        if is_sequential:
            code_remap = None
            max_code = len(raw_entries)
        else:
            # Build remapping: orig_code -> new_code (1..N), 0 maps to 0 (root)
            sorted_codes = sorted(set(orig_codes))
            code_remap = {0: 0}
            for i, orig in enumerate(sorted_codes):
                code_remap[orig] = i + 1
            max_code = len(sorted_codes)

        # Build byte trie
        root = _TrieNode()
        code_to_bytes = {}
        token_parent_codes = {}
        token_char_bytes = {}

        for orig_code, orig_parent, pattern_bytes in raw_entries:
            code = code_remap[orig_code] if code_remap else orig_code
            parent_code = code_remap.get(orig_parent, 0) if code_remap else orig_parent

            # Insert pattern into trie
            node = root
            for b in pattern_bytes:
                if b not in node.children:
                    node.children[b] = _TrieNode()
                node = node.children[b]
            node.code = code
            code_to_bytes[code] = pattern_bytes

            # Metadata for structured embedding
            token_parent_codes[code] = parent_code
            token_char_bytes[code] = pattern_bytes[-1] if pattern_bytes else 0

        # Build special tokens
        special_tokens = {}
        for i, name in enumerate(LZ78_SPECIAL_TOKENS):
            special_tokens[name] = max_code + 1 + 256 + i

        return cls(root, code_to_bytes, max_code, special_tokens,
                   token_parent_codes, token_char_bytes)

    @classmethod
    def _from_compressed_tsv(cls, tsv_path: str) -> LZ78Tokenizer:
        """Parse Compressed Trie TSV: node_idx, edge_label, parent_idx, output_code, pattern."""
        entries = []  # list of (node_idx, edge_label, parent_idx, output_code, pattern_bytes)
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#') or not line:
                    continue
                parts = line.split('\t')
                node_idx = int(parts[0])
                edge_label = parts[1]
                parent_idx = int(parts[2])
                output_code_str = parts[3]
                pattern = parts[-1]

                output_code = int(output_code_str) if output_code_str else -1
                pattern_bytes = pattern.encode('utf-8') if pattern else b''
                entries.append((node_idx, edge_label, parent_idx, output_code, pattern_bytes))

        # Find max output_code
        max_code = max(oc for _, _, _, oc, _ in entries if oc > 0)

        # Build mappings
        node_to_output_code = {}
        code_to_bytes = {}
        code_to_pattern_str = {}
        pattern_str_to_code = {}

        for node_idx, edge_label, parent_idx, output_code, pattern_bytes in entries:
            node_to_output_code[node_idx] = output_code
            if output_code > 0:
                code_to_bytes[output_code] = pattern_bytes
                pattern_str = pattern_bytes.decode('utf-8', errors='replace')
                code_to_pattern_str[output_code] = pattern_str
                pattern_str_to_code[pattern_str] = output_code

        # Build byte trie from patterns
        root = _TrieNode()
        for code, pattern_bytes in code_to_bytes.items():
            node = root
            for b in pattern_bytes:
                if b not in node.children:
                    node.children[b] = _TrieNode()
                node = node.children[b]
            node.code = code

        # Metadata: structured (LZ78-equivalent parent)
        token_parent_codes = {}
        token_char_bytes = {}
        for code, pattern_str in code_to_pattern_str.items():
            char_byte = code_to_bytes[code][-1] if code_to_bytes[code] else 0
            token_char_bytes[code] = char_byte
            # Find longest prefix that has a code (LZ78-equivalent parent)
            parent_code = 0
            for i in range(len(pattern_str) - 1, 0, -1):
                prefix = pattern_str[:i]
                if prefix in pattern_str_to_code:
                    parent_code = pattern_str_to_code[prefix]
                    break
            token_parent_codes[code] = parent_code

        # Metadata: hierarchical (trie parent's output_code)
        hier_parent_codes = {}
        for node_idx, edge_label, parent_idx, output_code, pattern_bytes in entries:
            if output_code > 0:
                parent_oc = node_to_output_code.get(parent_idx, -1)
                hier_parent_codes[output_code] = parent_oc if parent_oc > 0 else 0

        # Build special tokens
        special_tokens = {}
        for i, name in enumerate(LZ78_SPECIAL_TOKENS):
            special_tokens[name] = max_code + 1 + 256 + i

        return cls(root, code_to_bytes, max_code, special_tokens,
                   token_parent_codes, token_char_bytes, hier_parent_codes)

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, tokenizer_dir: str) -> None:
        """Serialize the tokenizer to a directory for later loading."""
        os.makedirs(tokenizer_dir, exist_ok=True)

        # Save config
        config = {
            "type": "lz78",
            "max_code": self._max_code,
            "vocab_size": self._vocab_size,
            "byte_fallback_offset": self._byte_fallback_offset,
            "special_tokens": self._special_tokens,
            "has_hier": self._hier_parent_codes is not None,
        }
        with open(os.path.join(tokenizer_dir, "lz78_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        # Save code_to_bytes as a simple text file (code \t hex_bytes)
        with open(os.path.join(tokenizer_dir, "lz78_codes.tsv"), 'w') as f:
            for code in sorted(self._code_to_bytes.keys()):
                hex_str = self._code_to_bytes[code].hex()
                f.write(f"{code}\t{hex_str}\n")

        # Save token_bytes.pt for BPB evaluation
        self._save_token_bytes(tokenizer_dir)

        # Save token_metadata.pt for structured embedding
        self._save_token_metadata(tokenizer_dir)

        # Save ancestor chains for prefix label loss
        self._save_ancestor_data(tokenizer_dir)

        print(f"Saved LZ78Tokenizer to {tokenizer_dir}")
        print(f"  vocab_size: {self._vocab_size}")
        print(f"  max_code: {self._max_code}")

    def _save_token_bytes(self, tokenizer_dir: str) -> None:
        """Save byte-length per token for BPB evaluation."""
        token_byte_lengths = torch.zeros(self._vocab_size, dtype=torch.long)
        # Dictionary tokens
        for code, raw_bytes in self._code_to_bytes.items():
            token_byte_lengths[code] = len(raw_bytes)
        # Byte fallback tokens: each is 1 byte
        for b in range(256):
            token_byte_lengths[self._byte_fallback_offset + b] = 1
        # Special tokens: 0 bytes (excluded from BPB)
        # (already zero)
        torch.save(token_byte_lengths, os.path.join(tokenizer_dir, "token_bytes.pt"))

    def _save_token_metadata(self, tokenizer_dir: str) -> None:
        """Save (parent_code, char_byte) per token for structured embedding."""
        # Structured metadata: (vocab_size, 2) -> [parent_code, char_byte]
        metadata = torch.zeros(self._vocab_size, 2, dtype=torch.long)
        for code in self._code_to_bytes:
            metadata[code, 0] = self._token_parent_codes.get(code, 0)
            metadata[code, 1] = self._token_char_bytes.get(code, 0)
        # Byte fallback: parent=0, char=byte_value
        for b in range(256):
            tid = self._byte_fallback_offset + b
            metadata[tid, 0] = 0
            metadata[tid, 1] = b
        torch.save(metadata, os.path.join(tokenizer_dir, "token_metadata.pt"))

        # Hierarchical metadata (if available)
        if self._hier_parent_codes is not None:
            hier_metadata = torch.zeros(self._vocab_size, 2, dtype=torch.long)
            for code in self._code_to_bytes:
                hier_metadata[code, 0] = self._hier_parent_codes.get(code, 0)
                hier_metadata[code, 1] = self._token_char_bytes.get(code, 0)
            for b in range(256):
                tid = self._byte_fallback_offset + b
                hier_metadata[tid, 0] = 0
                hier_metadata[tid, 1] = b
            torch.save(hier_metadata, os.path.join(tokenizer_dir, "token_metadata_hier.pt"))

    def _save_ancestor_data(self, tokenizer_dir: str) -> None:
        """Save ancestor chains for prefix label loss.

        Outputs:
            token_ancestors.pt: (vocab_size, max_depth) padded ancestor chain indices
            token_ancestor_depths.pt: (vocab_size,) actual depth of each chain
        """
        # Compute ancestor chains for all dictionary tokens
        chains = {}
        max_depth = 0

        for code in self._code_to_bytes:
            chain = []
            cur = code
            while cur > 0:
                chain.append(cur)
                cur = self._token_parent_codes.get(cur, 0)
            if chain:
                chains[code] = chain
                max_depth = max(max_depth, len(chain))

        # Byte fallback tokens: depth 1
        for b in range(256):
            tid = self._byte_fallback_offset + b
            chains[tid] = [tid]

        # Special tokens: depth 1
        for tid in self._special_tokens.values():
            chains[tid] = [tid]

        max_depth = max(max_depth, 1)

        # Build padded tensors
        ancestor_indices = torch.zeros(self._vocab_size, max_depth, dtype=torch.long)
        ancestor_depths = torch.zeros(self._vocab_size, dtype=torch.long)

        for tid, chain in chains.items():
            depth = len(chain)
            ancestor_indices[tid, :depth] = torch.tensor(chain, dtype=torch.long)
            ancestor_depths[tid] = depth

        torch.save(ancestor_indices, os.path.join(tokenizer_dir, "token_ancestors.pt"))
        torch.save(ancestor_depths, os.path.join(tokenizer_dir, "token_ancestor_depths.pt"))
        print(f"  ancestor data: max_depth={max_depth}, tokens_with_chains={len(chains)}")

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> LZ78Tokenizer:
        """Load a previously saved LZ78Tokenizer."""
        config_path = os.path.join(tokenizer_dir, "lz78_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        max_code = config["max_code"]
        special_tokens = config["special_tokens"]

        # Load code_to_bytes
        code_to_bytes = {}
        codes_path = os.path.join(tokenizer_dir, "lz78_codes.tsv")
        with open(codes_path, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                code_str, hex_str = line.split('\t')
                code_to_bytes[int(code_str)] = bytes.fromhex(hex_str)

        # Rebuild byte trie
        root = _TrieNode()
        for code, pattern_bytes in code_to_bytes.items():
            node = root
            for b in pattern_bytes:
                if b not in node.children:
                    node.children[b] = _TrieNode()
                node = node.children[b]
            node.code = code

        # Load metadata
        metadata_path = os.path.join(tokenizer_dir, "token_metadata.pt")
        metadata = torch.load(metadata_path, map_location="cpu")
        token_parent_codes = {}
        token_char_bytes = {}
        for code in code_to_bytes:
            token_parent_codes[code] = metadata[code, 0].item()
            token_char_bytes[code] = metadata[code, 1].item()

        # Load hierarchical metadata if available
        hier_parent_codes = None
        if config.get("has_hier", False):
            hier_path = os.path.join(tokenizer_dir, "token_metadata_hier.pt")
            if os.path.exists(hier_path):
                hier_metadata = torch.load(hier_path, map_location="cpu")
                hier_parent_codes = {}
                for code in code_to_bytes:
                    hier_parent_codes[code] = hier_metadata[code, 0].item()

        return cls(root, code_to_bytes, max_code, special_tokens,
                   token_parent_codes, token_char_bytes, hier_parent_codes)

    # -------------------------------------------------------------------------
    # Encode / Decode
    # -------------------------------------------------------------------------

    def _encode_bytes(self, data: bytes) -> list[int]:
        """Greedy longest-match encoding on raw bytes."""
        tokens = []
        root = self._root
        fallback = self._byte_fallback_offset
        i = 0
        n = len(data)
        while i < n:
            node = root
            best_code = -1
            best_end = i
            j = i
            while j < n:
                b = data[j]
                child = node.children.get(b)
                if child is None:
                    break
                node = child
                j += 1
                if node.code >= 0:
                    best_code = node.code
                    best_end = j
            if best_code >= 0:
                tokens.append(best_code)
                i = best_end
            else:
                # byte fallback
                tokens.append(fallback + data[i])
                i += 1
        return tokens

    def _encode_text(self, text: str) -> list[int]:
        """Encode a single string, respecting chunking setting."""
        if self._chunking_pattern is not None:
            chunks = self._chunking_pattern.findall(text)
            ids = []
            for chunk in chunks:
                ids.extend(self._encode_bytes(chunk.encode('utf-8')))
            return ids
        return self._encode_bytes(text.encode('utf-8'))

    def encode(
        self,
        text: str | list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int | None = None,
    ) -> list[int] | list[list[int]]:
        """Encode text to token IDs.

        Args:
            text: A string or list of strings to encode.
            prepend: Special token name or token id to prepend.
            append: Special token name or token id to append.
            num_threads: Unused. Accepted for API compatibility with
                other tokenizer backends that support parallel encoding.
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self._encode_text(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = [self._encode_text(t) for t in text]
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
            if append is not None:
                for row in ids:
                    row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs) -> list[int] | list[list[int]]:
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        byte_chunks = []
        for tid in ids:
            if tid in self._code_to_bytes:
                byte_chunks.append(self._code_to_bytes[tid])
            elif self._byte_fallback_offset <= tid < self._byte_fallback_offset + 256:
                byte_chunks.append(bytes([tid - self._byte_fallback_offset]))
            elif tid in self._special_tokens_reverse:
                byte_chunks.append(self._special_tokens_reverse[tid].encode('utf-8'))
            # else: skip (padding / unknown)
        return b''.join(byte_chunks).decode('utf-8', errors='replace')

    # -------------------------------------------------------------------------
    # API (matches nanochat tokenizer interface)
    # -------------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        """Return the total vocabulary size."""
        return self._vocab_size

    def get_special_tokens(self) -> set[str]:
        """Return the set of special token names."""
        return set(self._special_tokens.keys())

    def get_bos_token_id(self) -> int:
        """Return the beginning-of-sequence token id."""
        return self.bos_token_id

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        """Return the token id for a special token name."""
        if text not in self._special_tokens:
            raise ValueError(f"Unknown special token: {text}")
        return self._special_tokens[text]

    def id_to_token(self, token_id: int) -> str:
        """Return the string representation for a given token id."""
        if token_id in self._code_to_bytes:
            return self._code_to_bytes[token_id].decode('utf-8', errors='replace')
        elif self._byte_fallback_offset <= token_id < self._byte_fallback_offset + 256:
            b = token_id - self._byte_fallback_offset
            return f"<byte_{b:02x}>"
        elif token_id in self._special_tokens_reverse:
            return self._special_tokens_reverse[token_id]
        else:
            return f"<unk_{token_id}>"

    def get_token_metadata(self, mode: str = "structured") -> torch.Tensor:
        """Return (vocab_size, 2) tensor of [parent_code, char_byte] per token.

        Args:
            mode: "structured" or "hierarchical".
        """
        if mode == "hierarchical" and self._hier_parent_codes is not None:
            parent_codes = self._hier_parent_codes
        else:
            parent_codes = self._token_parent_codes

        metadata = torch.zeros(self._vocab_size, 2, dtype=torch.long)
        for code in self._code_to_bytes:
            metadata[code, 0] = parent_codes.get(code, 0)
            metadata[code, 1] = self._token_char_bytes.get(code, 0)
        # Byte fallback: parent=0, char=byte_value
        for b in range(256):
            tid = self._byte_fallback_offset + b
            metadata[tid, 0] = 0
            metadata[tid, 1] = b
        return metadata

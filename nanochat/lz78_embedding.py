"""
LZ78 embedding module for nanochat.

Four modes:
- flat: Standard nn.Embedding(vocab_size, n_embd)
- structured: code_embed(parent_code) + char_embed(char_byte)
  Decomposes each token into its LZ78 tree parent code + extension character byte.
- hierarchical: Same decomposition but uses the trie parent's code
  (only meaningful for compressed trie; for LZ78/freq_gated, same as structured).
- tuple: concat(code_embed(parent_code), char_embed(char_byte)) -> Linear -> n_embd
  More expressive than structured since the linear layer can learn interactions.
"""

from typing import Optional

import torch
import torch.nn as nn


class LZ78Embedding(nn.Module):
    def __init__(
        self,
        mode: str,
        vocab_size: int,
        n_embd: int,
        token_metadata: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            mode: "flat", "structured", "hierarchical", or "tuple"
            vocab_size: total vocabulary size (padded)
            n_embd: embedding dimension
            token_metadata: (vocab_size, 2) tensor of [parent_code, char_byte] per token
                            Required for structured/hierarchical/tuple modes.
        """
        super().__init__()
        self.mode = mode

        if mode == "flat":
            self.emb = nn.Embedding(vocab_size, n_embd)
        elif mode in ("structured", "hierarchical"):
            assert token_metadata is not None, f"{mode} mode requires token_metadata"
            self.code_emb = nn.Embedding(vocab_size, n_embd)
            self.char_emb = nn.Embedding(256, n_embd)
            self.register_buffer('parent_codes', token_metadata[:vocab_size, 0].long())
            self.register_buffer('char_bytes', token_metadata[:vocab_size, 1].long())
        elif mode == "tuple":
            assert token_metadata is not None, "tuple mode requires token_metadata"
            # Each component gets half the final dimension, then project to n_embd
            half = n_embd // 2
            self.code_emb = nn.Embedding(vocab_size, half)
            self.char_emb = nn.Embedding(256, half)
            self.proj = nn.Linear(2 * half, n_embd, bias=False)
            self.register_buffer('parent_codes', token_metadata[:vocab_size, 0].long())
            self.register_buffer('char_bytes', token_metadata[:vocab_size, 1].long())
        else:
            raise ValueError(f"Unknown embedding mode: {mode}")

    @property
    def weight(self) -> torch.nn.Parameter:
        """For compatibility with code that accesses .weight (e.g. device detection)."""
        if self.mode == "flat":
            return self.emb.weight
        else:
            return self.code_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) tensor of token IDs
        Returns:
            (B, T, n_embd) tensor of embeddings
        """
        if self.mode == "flat":
            return self.emb(token_ids)
        elif self.mode == "tuple":
            parent = self.parent_codes[token_ids]  # (B, T)
            char = self.char_bytes[token_ids]       # (B, T)
            p_emb = self.code_emb(parent)           # (B, T, half)
            c_emb = self.char_emb(char)             # (B, T, half)
            return self.proj(torch.cat([p_emb, c_emb], dim=-1))  # (B, T, n_embd)
        else:
            parent = self.parent_codes[token_ids]  # (B, T)
            char = self.char_bytes[token_ids]       # (B, T)
            return self.code_emb(parent) + self.char_emb(char)

    def init_weights(self) -> None:
        """Initialize embedding weights."""
        # std=1.0 is used here (rather than the typical 0.02) because the GPT
        # model class applies its own weight scaling after calling this method.
        if self.mode == "flat":
            nn.init.normal_(self.emb.weight, mean=0.0, std=1.0)
        elif self.mode == "tuple":
            nn.init.normal_(self.code_emb.weight, mean=0.0, std=1.0)
            nn.init.normal_(self.char_emb.weight, mean=0.0, std=1.0)
            # Xavier init for the projection layer
            nn.init.xavier_normal_(self.proj.weight)
        else:
            nn.init.normal_(self.code_emb.weight, mean=0.0, std=1.0)
            nn.init.normal_(self.char_emb.weight, mean=0.0, std=1.0)

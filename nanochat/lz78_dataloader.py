"""
Pre-tokenized dataloader for LZ78 experiments.

Reads .npy shard files produced by lz78_pretokenize.py, fills a token buffer,
and yields (inputs, targets, state_dict) batches for training.

Supports distributed training: each rank processes shards
rank, rank+world_size, rank+2*world_size, ...

Resumable via state_dict = {"shard_idx": int, "position": int}
"""

import os
from collections import deque
from typing import Any, Generator, Optional

import numpy as np
import torch

from nanochat.common import get_dist_info


def pretokenized_data_loader_with_state(
    B: int,
    T: int,
    split: str,
    pretokenized_dir: str,
    device: str = "cuda",
    resume_state_dict: Optional[dict[str, int]] = None,
) -> Generator[tuple[torch.Tensor, torch.Tensor, dict[str, int]], None, None]:
    """
    Stream pre-tokenized data from .npy shards.

    Args:
        B: batch size (per device)
        T: sequence length
        split: "train" or "val"
        pretokenized_dir: directory containing {split}/ subdirectory with .npy shards
        device: target device
        resume_state_dict: optional {"shard_idx": int, "position": int} for resuming
    """
    assert split in ("train", "val")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # Find shard files
    split_dir = os.path.join(pretokenized_dir, split)
    shard_files = sorted([
        os.path.join(split_dir, f) for f in os.listdir(split_dir)
        if f.endswith('.npy')
    ])
    assert len(shard_files) > 0, f"No .npy shards found in {split_dir}"

    needed_tokens = B * T + 1  # +1 for the target at the last position

    # Infinite iterator over shards (multi-epoch)
    def shard_iterator() -> Generator[tuple[np.ndarray, int, int], None, None]:
        """Yield (shard_data, shard_idx, start_position) cycling over shards indefinitely."""
        resume_shard_idx = resume_state_dict["shard_idx"] if resume_state_dict else 0
        resume_position = resume_state_dict["position"] if resume_state_dict else 0
        first_pass = True
        while True:
            start_idx = resume_shard_idx if first_pass else 0
            # Each rank takes every world_size-th shard
            for shard_idx in range(start_idx + ddp_rank, len(shard_files), ddp_world_size):
                data = np.load(shard_files[shard_idx])
                # On resume, skip to the saved position within the shard
                pos = 0
                if first_pass and shard_idx == resume_shard_idx + ddp_rank:
                    pos = resume_position
                yield data, shard_idx, pos
            first_pass = False

    shards = shard_iterator()
    token_buffer = deque()
    current_shard_idx = 0

    while True:
        # Fill buffer until we have enough tokens
        while len(token_buffer) < needed_tokens:
            data, current_shard_idx, pos = next(shards)
            for i in range(pos, len(data)):
                token_buffer.append(int(data[i]))

        # Extract tokens for one batch
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        use_cuda = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        state_dict = {"shard_idx": current_shard_idx, "position": 0}
        yield inputs, targets, state_dict


def pretokenized_data_loader(*args: Any, **kwargs: Any) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Helper that only yields (inputs, targets) without state_dict."""
    for inputs, targets, _ in pretokenized_data_loader_with_state(*args, **kwargs):
        yield inputs, targets

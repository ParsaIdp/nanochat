"""nanochat — a full-stack LLM training framework.

Pipeline: tokenizer → pretrain → midtrain → SFT → RL → inference.

Key classes:
    GPT, GPTConfig: Transformer model and configuration.
    RustBPETokenizer: Default BPE tokenizer (rustbpe + tiktoken).
    LZ78Tokenizer: Dictionary-based LZ78 tokenizer.
    Engine: Inference engine with KV cache and tool use.
"""

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.lz78_tokenizer import LZ78Tokenizer
from nanochat.engine import Engine

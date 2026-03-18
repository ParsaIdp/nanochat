import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyarrow.parquet as pq
import regex
import tiktoken

from nanochat.dataset import list_parquet_files, iter_docs_by_split, iter_numina_math_cot_docs

# GPT-4 pre-tokenization pattern: number of chunks per token = len(regex.findall(..., token_str))
GPT4_PATTERN = regex.compile(
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

# Dataset name -> data dir for parquet shards (override with data_dir when calling)
DATASET_PATHS = {
    "c4": r"C:\large_storage\c4",
}


def _byte_lens_from_file(path):
    """Load a tiktoken Encoding .pkl and return an array of per-token byte lengths."""
    with open(path, "rb") as f:
        enc = pickle.load(f)
    # enc._mergeable_ranks maps token_bytes -> rank (int). Ranks may not be
    # perfectly contiguous in [0, len(mapping)), so we build an array of length
    # max_rank + 1 and fill indices by rank.
    rank_to_bytes = {rank: token for token, rank in enc._mergeable_ranks.items()}
    if not rank_to_bytes:
        return np.array([], dtype=np.int32)
    max_rank = max(rank_to_bytes.keys())
    byte_lens = np.zeros(max_rank + 1, dtype=np.int32)
    for rank, token_bytes in rank_to_bytes.items():
        byte_lens[rank] = len(token_bytes)
    return byte_lens


def _load_encoding(path):
    """Load tiktoken Encoding from .pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _chunk_counts_from_file(path):
    """Load a tiktoken Encoding .pkl and return per-token chunk counts (GPT-4 regex matches)."""
    with open(path, "rb") as f:
        enc = pickle.load(f)
    # enc._mergeable_ranks maps token_bytes -> rank (int). Ranks may be sparse,
    # so build an array indexed by rank up to max_rank.
    rank_to_bytes = {rank: token for token, rank in enc._mergeable_ranks.items()}
    if not rank_to_bytes:
        return np.zeros(0, dtype=np.int64)
    max_rank = max(rank_to_bytes.keys())
    chunk_counts = np.zeros(max_rank + 1, dtype=np.int64)
    for rank, token_bytes in rank_to_bytes.items():
        token_str = token_bytes.decode("utf-8", errors="replace")
        chunk_counts[rank] = len(GPT4_PATTERN.findall(token_str))
    return chunk_counts


def plot_bytes_per_token(dictionaries_path, vocab_sizes: list[int]):
    """
    For every tokenizer .pkl (tiktoken Encoding) found in subdirectories of
    dictionaries_path (e.g. bpe/, bpe_chunk/, bpe_superchunk/), plot vocab size
    vs. average bytes-per-token at powers of 2 (256, 512, 1k, 2k, …) up to the
    minimum vocab size across all dictionaries.
    Lines are colored by subfolder and styled by file.
    """
    linestyles = ["-", "--", "-.", ":"]

    # Collect (subdir, filename) pairs, grouped by subdir
    groups = {}
    for entry in sorted(os.scandir(dictionaries_path), key=lambda e: e.name):
        if entry.is_dir() and not entry.name.startswith("."):
            files = sorted(
                f.name for f in os.scandir(entry.path)
                if f.is_file() and f.name.endswith(".pkl") and not f.name.startswith(".")
            )
            if files:
                groups[entry.name] = files

    subdir_colors = {name: color for name, color in zip(
        groups, cm.tab10(np.linspace(0, 1, len(groups)))
    )}

    fig, ax = plt.subplots(figsize=(11, 6))

    for subdir, files in groups.items():
        color = subdir_colors[subdir]
        for i, fname in enumerate(files):
            fpath = os.path.join(dictionaries_path, subdir, fname)
            byte_lens = _byte_lens_from_file(fpath)
            n_tokens = len(byte_lens)
            xs = [vs for vs in vocab_sizes if vs <= n_tokens]
            ys = [byte_lens[:vs].mean() for vs in xs]

            label = f"{subdir} / {os.path.splitext(fname)[0].lstrip('_')}"
            ls = linestyles[i % len(linestyles)]
            ax.plot(xs, ys, marker="o", linewidth=2, linestyle=ls, label=label, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels([f"{v // 1000}k" if v >= 1000 else str(v) for v in vocab_sizes])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Avg bytes / token")
    ax.set_title("Tokenizer efficiency: avg bytes per token vs. vocab size")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_chunk_distribution(
    dict_filepath: str,
    vocab_sizes: list[int],
    figsize=(10, 4),
    min_freq = 20,
):
    """
    Plot the distribution of token counts by number of chunks (GPT-4 regex
    segments per token) for each vocab size. One subplot per vocab size,
    stacked vertically.

    Args:
        dict_filepath: Path to a tokenizer .pkl (tiktoken Encoding).
        vocab_sizes: List of vocab sizes to plot (e.g. [1024, 4096, 16384]).
        figsize: (width, height_per_subplot); total height = figsize[1] * n_plots.
        min_freq: If set, only include chunk counts that have at least this many
            tokens in the distribution (filters out rare chunk values).

    Returns:
        matplotlib Figure.
    """
    chunk_counts = _chunk_counts_from_file(f'{dict_filepath}/tokenizer.pkl')
    n_tokens = len(chunk_counts)
    # We only care about chunk counts up to 10
    max_chunks = min(10, int(chunk_counts.max()))
    vocab_sizes = [vs for vs in vocab_sizes if vs <= n_tokens]
    if not vocab_sizes:
        raise ValueError(f"All requested vocab_sizes exceed tokenizer size {n_tokens}")

    n_plots = len(vocab_sizes)
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(figsize[0], figsize[1] * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, vs in zip(axes, vocab_sizes):
        counts_at_vs = chunk_counts[:vs]
        chunks, num_tokens = np.unique(counts_at_vs, return_counts=True)
        # Keep only chunks up to max_chunks and above the min_freq threshold
        mask = (chunks <= max_chunks) & (num_tokens >= min_freq)
        chunks = chunks[mask]
        num_tokens = num_tokens[mask]
        ax.bar(chunks, num_tokens, width=0.8, align="center", color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_ylabel("Number of tokens")
        ax.set_title(f"Vocab size = {vs:,}")
        ax.set_xlim(0, max_chunks + 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Chunks per token (GPT-4 regex segments)")
    fig.suptitle(f"Token distribution by chunks: {os.path.basename(os.path.dirname(dict_filepath))} / {os.path.basename(dict_filepath)}", y=1.02)
    plt.tight_layout()
    return fig


# No-split pattern for expanding OOV tokens so superchunk tokens (e.g. " We've")
# are not re-split by the GPT-4 regex when we decode and re-encode.
_NO_SPLIT_PATTERN = r"[\s\S]+"


def _effective_token_count(
    doc_strings: list[str],
    enc: tiktoken.Encoding,
    vocab_size: int,
    _truncated_cache: dict | None = None,
) -> float:
    """
    Token count when using only the first vocab_size tokens.

    Encodes each document directly with a truncated encoder that has only the
    first vocab_size merges and uses a no-split pre-tokenization pattern.
    """
    if _truncated_cache is None:
        _truncated_cache = {}
    if vocab_size not in _truncated_cache:
        mr = enc._mergeable_ranks
        truncated_mr = {k: v for k, v in mr.items() if v < vocab_size}
        _truncated_cache[vocab_size] = tiktoken.Encoding(
            name=f"trunc_{vocab_size}",
            pat_str=_NO_SPLIT_PATTERN,
            mergeable_ranks=truncated_mr,
            special_tokens={},
        )
    truncated_enc = _truncated_cache[vocab_size]

    sample = ("".join(doc_strings))[:200] if doc_strings else " " * 200
    if len(sample) < 200 and doc_strings:
        sample = (sample + " " * 200)[:200]
    token_ids = truncated_enc.encode(sample)
    token_strings = [truncated_enc.decode([tid]) for tid in token_ids]
    print(f"effective_token_count vocab_size={vocab_size} 200-char sample token strings: {token_strings}")

    total = sum(len(truncated_enc.encode(doc)) for doc in doc_strings)
    return float(total)


def plot_bytes_per_token_vs_vocab(
    dataset_name: str,
    tokenizer_dirs: list[str],
    n_docs: int,
    vocab_sizes: list[int],
    data_dir: str | None = None,
    figsize=(10, 4),
):
    """
    Use the same iterator as tok_train: train = first n_docs from split "train",
    val = first n_docs from split "val" (parquet) or from "test" (nmc). For each
    tokenizer encode and compute bytes/token at each vocab size V (first V tokens).
    One subplot per tokenizer, stacked vertically. Higher is better.

    Args:
        dataset_name: Key for DATASET_PATHS (e.g. "c4"), or "nmc" for NuminaMath-CoT.
        tokenizer_dirs: List of tokenizer directories (each contains tokenizer.pkl).
        n_docs: Number of *documents* (or problems for nmc) to use for train and val.
        vocab_sizes: Vocab sizes to evaluate (first V tokens) per tokenizer.
        data_dir: Override dataset path. Use "nmc" or "numina_math_cot" for NuminaMath-CoT.
        figsize: (width, height_per_subplot); total height = figsize[1] * len(tokenizer_dirs).

    Returns:
        matplotlib Figure.
    """
    use_nmc = dataset_name == "nmc" or data_dir in ("nmc", "numina_math_cot")
    if use_nmc:
        docs = list(iter_numina_math_cot_docs("train", max_problems=n_docs * 2))
        train_docs = docs[:n_docs]
        val_docs = docs[n_docs:]
        display_name = "nmc"
    else:
        data_dir = data_dir or DATASET_PATHS.get(dataset_name)
        if not data_dir or not os.path.isdir(data_dir):
            raise ValueError(f"data_dir must be an existing directory (got dataset_name={dataset_name!r}, data_dir={data_dir!r})")
        train_docs = list(iter_docs_by_split(data_dir, "train", max_docs=n_docs))
        val_docs = list(iter_docs_by_split(data_dir, "val", max_docs=n_docs))
        if len(train_docs) < n_docs:
            raise ValueError(f"Only {len(train_docs)} train docs available, need {n_docs}")
        if len(val_docs) < n_docs:
            raise ValueError(f"Only {len(val_docs)} val docs available, need {n_docs}")
        display_name = dataset_name

    train_bytes = sum(len(d.encode("utf-8")) for d in train_docs)
    val_bytes = sum(len(d.encode("utf-8")) for d in val_docs)

    n_plots = len(tokenizer_dirs)
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(figsize[0], figsize[1] * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, tok_dir in zip(axes, tokenizer_dirs):
        pkl_path = os.path.join(tok_dir, "tokenizer.pkl")
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"Tokenizer not found: {pkl_path}")
        enc = _load_encoding(pkl_path)
        n_tokens = len(enc._mergeable_ranks)
        vs_list = [v for v in vocab_sizes if v <= n_tokens]
        if not vs_list:
            raise ValueError(f"No vocab_sizes <= tokenizer size {n_tokens} at {tok_dir}")

        trunc_cache = {}
        train_bpt_list = []
        val_bpt_list = []
        for v in vs_list:
            print(tok_dir, v)
            eff_train = _effective_token_count(train_docs, enc, v, _truncated_cache=trunc_cache)
            eff_val = _effective_token_count(val_docs, enc, v, _truncated_cache=trunc_cache)
            train_bpt_list.append(train_bytes / eff_train if eff_train else 0.0)
            val_bpt_list.append(val_bytes / eff_val if eff_val else 0.0)

        label = os.path.basename(os.path.normpath(tok_dir))
        ax.plot(vs_list, train_bpt_list, marker="o", label="Train bytes/token", linewidth=2)
        ax.plot(vs_list, val_bpt_list, marker="s", label="Val bytes/token", linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_ylabel("Bytes / token")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Vocabulary size (first V tokens)")
    fig.suptitle(f"Bytes per token vs vocab size ({display_name}, n_docs={n_docs:,})", y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BPE tokenizers with various plots.")
    parser.add_argument(
        "--eval",
        type=str,
        default="bytes_per_token_vs_vocab",
        choices=["bytes_per_token", "chunk_distribution", "bytes_per_token_vs_vocab"],
        help="Which evaluation to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        help="Short dataset name (e.g. 'c4', 'nmc'). Used to infer dictionary folder {dataset}_dictionaries and plot output path docs/plots/{dataset}.",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=50_000,
        help="For bytes_per_token_vs_vocab: number of documents/problems per split.",
    )
    args = parser.parse_args()

    dictionaries_path = f"{args.dataset}_dictionaries"
    os.makedirs(dictionaries_path, exist_ok=True)

    plots_dir = os.path.join("docs", "plots", args.dataset)
    os.makedirs(plots_dir, exist_ok=True)

    if args.eval == "bytes_per_token":
        vocab_sizes = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
        fig = plot_bytes_per_token(dictionaries_path, vocab_sizes)
        fig.show()
        plt.show()
        fig.savefig(os.path.join(plots_dir, "bytes_per_token.png"))
    elif args.eval == "chunk_distribution":
        fig = plot_chunk_distribution(
            f"{dictionaries_path}/bpe_superchunk_para",
            [8_000, 32_000, 128_000],
        )
        fig.show()
        plt.show()
        fig.savefig(os.path.join(plots_dir, "chunk_distribution.png"))
    else:
        # bytes_per_token_vs_vocab
        tokenizer_dirs = [
            f"{dictionaries_path}/bpe",
            f"{dictionaries_path}/bpe_chunk",
            f"{dictionaries_path}/bpe_superchunk_para",
        ]
        vocab_sizes = [1_000, 4_000, 16_000, 64_000, 128_000]
        fig = plot_bytes_per_token_vs_vocab(
            args.dataset,
            tokenizer_dirs,
            args.n_docs,
            vocab_sizes,
            data_dir="nmc" if args.dataset == "nmc" else None,
        )
        fig.show()
        plt.show()
        fig.savefig(os.path.join(plots_dir, "bytes_per_token_vs_vocab.png"))

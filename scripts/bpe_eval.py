import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyarrow.parquet as pq
import regex
import tiktoken

from nanochat.dataset import list_parquet_files, iter_docs_by_split

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
    id_to_bytes = {rank: token for token, rank in enc._mergeable_ranks.items()}
    byte_lens = [len(id_to_bytes[i]) for i in range(len(id_to_bytes))]
    return np.array(byte_lens)


def _load_encoding(path):
    """Load tiktoken Encoding from .pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _chunk_counts_from_file(path):
    """Load a tiktoken Encoding .pkl and return per-token chunk counts (GPT-4 regex matches)."""
    with open(path, "rb") as f:
        enc = pickle.load(f)
    id_to_bytes = {rank: token for token, rank in enc._mergeable_ranks.items()}
    n = len(id_to_bytes)
    chunk_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        token_bytes = id_to_bytes[i]
        token_str = token_bytes.decode("utf-8", errors="replace")
        chunk_counts[i] = len(GPT4_PATTERN.findall(token_str))
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
    max_chunks = int(chunk_counts.max())
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
        mask = num_tokens >= min_freq
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


def _effective_token_count(
    doc_strings: list[str],
    enc: tiktoken.Encoding,
    vocab_size: int,
    _truncated_cache: dict | None = None,
) -> float:
    """
    Token count when using only the first vocab_size tokens (standard BPE).
    Encode each doc with a truncated vocab (first V tokens) and sum lengths;
    avoids one giant string so encoding is much faster.
    """
    if vocab_size not in _truncated_cache:
        mr = enc._mergeable_ranks
        truncated_mr = {k: v for k, v in mr.items() if v < vocab_size}
        _truncated_cache[vocab_size] = tiktoken.Encoding(
            name=f"trunc_{vocab_size}",
            pat_str=enc._pat_str,
            mergeable_ranks=truncated_mr,
            special_tokens={},
        )
    truncated_enc = _truncated_cache[vocab_size]
    # Debug: print token strings for a 200-char sample (every call)
    sample = ("".join(doc_strings))[:200] if doc_strings else " " * 200
    if len(sample) < 200 and doc_strings:
        sample = (sample + " " * 200)[:200]
    token_ids = truncated_enc.encode(sample)
    token_strings = [truncated_enc.decode([tid]) for tid in token_ids]
    print(f"effective_token_count vocab_size={vocab_size} 200-char sample token strings: {token_strings}")
    return sum(len(truncated_enc.encode(d)) for d in doc_strings)


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
    val = first n_docs from split "val" (last parquet file). For each tokenizer
    encode and compute bytes/token at each vocab size V (first V tokens). One
    subplot per tokenizer, stacked vertically. Higher is better.

    Args:
        dataset_name: Key for DATASET_PATHS (e.g. "c4"); ignored if data_dir set.
        tokenizer_dirs: List of tokenizer directories (each contains tokenizer.pkl).
        n_docs: Number of *documents* to use for train and for val (each split gets this many docs).
        vocab_sizes: Vocab sizes to evaluate (first V tokens) per tokenizer.
        data_dir: Override dataset path. If None, use DATASET_PATHS[dataset_name].
        figsize: (width, height_per_subplot); total height = figsize[1] * len(tokenizer_dirs).

    Returns:
        matplotlib Figure.
    """
    data_dir = data_dir or DATASET_PATHS.get(dataset_name)
    if not data_dir or not os.path.isdir(data_dir):
        raise ValueError(f"data_dir must be an existing directory (got dataset_name={dataset_name!r}, data_dir={data_dir!r})")

    train_docs = list(iter_docs_by_split(data_dir, "train", max_docs=n_docs))
    val_docs = list(iter_docs_by_split(data_dir, "val", max_docs=n_docs))
    if len(train_docs) < n_docs:
        raise ValueError(f"Only {len(train_docs)} train docs available, need {n_docs}")
    if len(val_docs) < n_docs:
        raise ValueError(f"Only {len(val_docs)} val docs available, need {n_docs}")

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
    fig.suptitle(f"Bytes per token vs vocab size ({dataset_name}, n_docs={n_docs:,})", y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    dictionaries_path = "c4_dictionaries"

    # bytes_per_token = plot_bytes_per_token(dictionaries_path, [2**k for k in range(8, 18)])
    # bytes_per_token.show()
    # plt.show()
    # bytes_per_token.savefig('docs/plots/bytes_per_token.png')

    # chunk_distribution = plot_chunk_distribution(f'{dictionaries_path}/bpe_superchunk_para', [8_000, 32_000, 128_000])
    # chunk_distribution.show()
    # plt.show()
    # chunk_distribution.savefig('docs/plots/chunk_distribution.png')

    bytes_per_token_vs_vocab = plot_bytes_per_token_vs_vocab(
        "c4",
        [
            f"{dictionaries_path}/bpe",
            f"{dictionaries_path}/bpe_chunk",
            f"{dictionaries_path}/bpe_superchunk_para",
        ],
        10_000,
        [1_000, 4_000, 16_000, 64_000, 128_000],
    )
    bytes_per_token_vs_vocab.show()
    plt.show()
    bytes_per_token_vs_vocab.savefig('docs/plots/bytes_per_token_vs_vocab.png')

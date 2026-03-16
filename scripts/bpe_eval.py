import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyarrow.parquet as pq
import regex

from nanochat.dataset import list_parquet_files, parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer

# GPT-4 pre-tokenization pattern: number of chunks per token = len(regex.findall(..., token_str))
GPT4_PATTERN = regex.compile(
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

# Dataset name -> data dir for parquet shards (override with data_dir when calling)
DATASET_PATHS = {
    "c4": r"C:\large_storage\c4",
}


def iter_docs(data_dir: str):
    """Yield document strings from all parquet shards in data_dir (sorted by filename)."""
    paths = list_parquet_files(data_dir=data_dir)
    for filepath in paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                yield text


def iter_docs_by_split(
    split: str,
    data_dir: str,
    doc_cap: int | None = None,
):
    """
    Same iteration as tok_train: parquets_iter_batched(split=..., data_dir=...),
    flattened to one doc per yield. split is "train" (all but last file) or "val" (last file).
    If doc_cap is set, each doc is cropped to that many characters (like tok_train).
    """
    for batch in parquets_iter_batched(split=split, data_dir=data_dir):
        for doc in batch:
            if doc_cap is not None and len(doc) > doc_cap:
                doc = doc[:doc_cap]
            yield doc


def _byte_lens_from_file(path):
    """Load a tiktoken Encoding .pkl and return an array of per-token byte lengths."""
    with open(path, "rb") as f:
        enc = pickle.load(f)
    id_to_bytes = {rank: token for token, rank in enc._mergeable_ranks.items()}
    byte_lens = [len(id_to_bytes[i]) for i in range(len(id_to_bytes))]
    return np.array(byte_lens)


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


def _effective_token_count(flat_ids: list[int], byte_lens: np.ndarray, vocab_size: int) -> float:
    """Token count when using only the first vocab_size tokens; OOV (id >= vocab_size) counts as its byte length."""
    n = len(byte_lens)
    total = 0
    for tid in flat_ids:
        if tid < vocab_size:
            total += 1
        else:
            total += byte_lens[tid] if tid < n else 1
    return total


def plot_bytes_per_token_vs_vocab(
    dataset_name: str,
    tokenizer_dirs: list[str],
    n_docs: int,
    vocab_sizes: list[int],
    data_dir: str | None = None,
    doc_cap: int | None = None,
    figsize=(10, 4),
):
    """
    Use the same iterator as tok_train: train = first n_docs from split "train",
    val = first n_docs from split "val" (last parquet file). For each tokenizer
    encode and compute bytes/token at each vocab size V (first V tokens). One
    subplot per tokenizer, stacked vertically.

    Args:
        dataset_name: Key for DATASET_PATHS (e.g. "c4"); ignored if data_dir set.
        tokenizer_dirs: List of tokenizer directories (each contains tokenizer.pkl).
        n_docs: Number of docs for train and for val.
        vocab_sizes: Vocab sizes to evaluate (first V tokens) per tokenizer.
        data_dir: Override dataset path. If None, use DATASET_PATHS[dataset_name].
        doc_cap: If set, crop each doc to this many chars (same as tok_train).
        figsize: (width, height_per_subplot); total height = figsize[1] * len(tokenizer_dirs).

    Returns:
        matplotlib Figure.
    """
    data_dir = data_dir or DATASET_PATHS.get(dataset_name)
    if not data_dir or not os.path.isdir(data_dir):
        raise ValueError(f"data_dir must be an existing directory (got dataset_name={dataset_name!r}, data_dir={data_dir!r})")

    train_iter = iter_docs_by_split("train", data_dir, doc_cap=doc_cap)
    val_iter = iter_docs_by_split("val", data_dir, doc_cap=doc_cap)
    train_docs = list(itertools.islice(train_iter, n_docs))
    val_docs = list(itertools.islice(val_iter, n_docs))
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
        byte_lens = _byte_lens_from_file(pkl_path)
        tokenizer = RustBPETokenizer.from_directory(tok_dir)
        n_tokens = len(byte_lens)
        vs_list = [v for v in vocab_sizes if v <= n_tokens]
        if not vs_list:
            raise ValueError(f"No vocab_sizes <= tokenizer size {n_tokens} at {tok_dir}")

        train_ids = tokenizer.encode(train_docs)
        val_ids = tokenizer.encode(val_docs)
        flat_train = [tid for doc_ids in train_ids for tid in doc_ids]
        flat_val = [tid for doc_ids in val_ids for tid in doc_ids]

        train_bpt_list = []
        val_bpt_list = []
        for v in vs_list:
            eff_train = _effective_token_count(flat_train, byte_lens, v)
            eff_val = _effective_token_count(flat_val, byte_lens, v)
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
        [2**k for k in range(8, 18)],
    )
    bytes_per_token_vs_vocab.show()
    plt.show()
    bytes_per_token_vs_vocab.savefig('docs/plots/bytes_per_token_vs_vocab.png')

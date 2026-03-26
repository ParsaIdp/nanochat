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
# Resolve paths relative to the project root (repo root containing this script).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATASET_PATHS = {
    "c4": os.path.join(PROJECT_ROOT, "c4"),
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


def _scan_dictionary_groups(dictionaries_path: str) -> dict[str, list[str]]:
    """Map subdir name -> sorted list of .pkl filenames under dictionaries_path."""
    groups: dict[str, list[str]] = {}
    if not os.path.isdir(dictionaries_path):
        return groups
    for entry in sorted(os.scandir(dictionaries_path), key=lambda e: e.name):
        if entry.is_dir() and not entry.name.startswith("."):
            files = sorted(
                f.name for f in os.scandir(entry.path)
                if f.is_file() and f.name.endswith(".pkl") and not f.name.startswith(".")
            )
            if files:
                groups[entry.name] = files
    return groups


def plot_bytes_per_token(
    dataset_paths: list[tuple[str, str]],
    vocab_sizes: list[int],
):
    """
    For each (dataset_label, dictionaries_path), scan tokenizer .pkl files under
    subdirs (e.g. bpe/, bpe_chunk/). Plot vocab size vs. mean bytes-per-token.

    Single dataset: line style cycles per file within a subdir (unchanged).
    Multiple datasets: same color per tokenizer subdir; first dataset solid,
    second dotted (third+ alternate for clarity).

    With exactly two datasets, draws a translucent band between the two curves
    for every common ``(subdir, .pkl)`` pair.
    """
    if not dataset_paths:
        raise ValueError("dataset_paths must be non-empty")

    linestyles_multi = ["-", ":", "--", "-."]
    linestyles_single = ["-", "--", "-.", ":"]

    # Union of subdirs across all dataset roots
    all_groups = [_scan_dictionary_groups(p) for _, p in dataset_paths]
    all_subdirs = sorted(set().union(*(g.keys() for g in all_groups)))

    subdir_colors = {
        name: color
        for name, color in zip(all_subdirs, cm.tab10(np.linspace(0, 1, max(1, len(all_subdirs)))))
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    n_ds = len(dataset_paths)

    for subdir in all_subdirs:
        color = subdir_colors[subdir]
        if n_ds == 1:
            ds_label, root = dataset_paths[0]
            groups = all_groups[0]
            if subdir not in groups:
                continue
            for i, fname in enumerate(groups[subdir]):
                fpath = os.path.join(root, subdir, fname)
                byte_lens = _byte_lens_from_file(fpath)
                n_tokens = len(byte_lens)
                xs = [vs for vs in vocab_sizes if vs <= n_tokens]
                ys = [byte_lens[:vs].mean() for vs in xs]
                stem = os.path.splitext(fname)[0].lstrip("_")
                label = f"{subdir} / {stem}"
                ls = linestyles_single[i % len(linestyles_single)]
                ax.plot(xs, ys, marker="o", linewidth=2, linestyle=ls, label=label, color=color)
        else:
            # Only compare a subdir when every dataset has it; then intersect filenames
            if not all(subdir in g for g in all_groups):
                continue
            common_files = set(all_groups[0][subdir])
            for g in all_groups[1:]:
                common_files &= set(g[subdir])

            for fname in sorted(common_files):
                series_xs_ys = []
                for j, (ds_label, root) in enumerate(dataset_paths):
                    g = all_groups[j]
                    if subdir not in g or fname not in g[subdir]:
                        series_xs_ys.append(None)
                        continue
                    fpath = os.path.join(root, subdir, fname)
                    byte_lens = _byte_lens_from_file(fpath)
                    n_tokens = len(byte_lens)
                    xs = [vs for vs in vocab_sizes if vs <= n_tokens]
                    ys = [byte_lens[:vs].mean() for vs in xs]
                    series_xs_ys.append((ds_label, xs, ys))

                for idx, item in enumerate(series_xs_ys):
                    if item is None:
                        continue
                    ds_label, xs, ys = item
                    ls = linestyles_multi[idx % len(linestyles_multi)]
                    stem = os.path.splitext(fname)[0].lstrip("_")
                    label = f"{ds_label} / {subdir} / {stem}"
                    ax.plot(xs, ys, marker="o", linewidth=2, linestyle=ls, label=label, color=color)

                if (
                    n_ds == 2
                    and series_xs_ys[0] is not None
                    and series_xs_ys[1] is not None
                ):
                    _, xs0, y0 = series_xs_ys[0]
                    _, xs1, y1 = series_xs_ys[1]
                    common_x = [x for x in vocab_sizes if x in set(xs0) & set(xs1)]
                    if common_x:
                        m0 = {x: y for x, y in zip(xs0, y0)}
                        m1 = {x: y for x, y in zip(xs1, y1)}
                        yy0 = [m0[x] for x in common_x]
                        yy1 = [m1[x] for x in common_x]
                        ax.fill_between(
                            common_x,
                            yy0,
                            yy1,
                            alpha=0.25,
                            color=color,
                            linewidth=0,
                        )

    title_suffix = ", ".join(label for label, _ in dataset_paths)
    ax.set_xscale("log", base=2)
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels([f"{v // 1000}k" if v >= 1000 else str(v) for v in vocab_sizes])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Avg bytes / token")
    ax.set_title(f"Tokenizer efficiency: avg bytes per token vs. vocab size ({title_suffix})")
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


def _chunk_counts_from_encoding(enc: tiktoken.Encoding) -> np.ndarray:
    """Per-rank GPT-4 regex chunk counts for a tiktoken Encoding (same as _chunk_counts_from_file)."""
    mr = enc._mergeable_ranks
    if not mr:
        return np.zeros(0, dtype=np.int64)
    max_rank = max(mr.values())
    chunk_counts = np.zeros(max_rank + 1, dtype=np.int64)
    for token_bytes, rank in mr.items():
        token_str = token_bytes.decode("utf-8", errors="replace")
        chunk_counts[rank] = len(GPT4_PATTERN.findall(token_str))
    return chunk_counts


def _max_chunk_allowed_at_pos(pos: int, T: int, max_c: int) -> int:
    """Output positions [0, T) allow 1 chunk; [T, 2T) allow 2; …  T<=0 means no cap (use max_c)."""
    if T <= 0:
        return max_c
    return min(1 + pos // T, max_c)


def _encode_cot_chunk_schedule(
    text: str,
    enc: tiktoken.Encoding,
    chunk_counts: np.ndarray,
    T: int,
) -> list[int]:
    """
    Encode CoT text with a per-output-position cap on GPT-4 chunk count.

    T <= 0: native tiktoken ``encode_ordinary`` (no progressive restriction).
    T > 0: greedy longest-byte-prefix over ``enc._mergeable_ranks`` so each
    emitted token respects the cap at that output index (same vocabulary as tiktoken).
    """
    if T <= 0:
        return enc.encode_ordinary(text)

    mr = enc._mergeable_ranks
    max_c = int(chunk_counts.max())
    max_len = max((len(k) for k in mr.keys()), default=1)
    b = text.encode("utf-8")
    n = len(b)
    out: list[int] = []
    pos = 0
    i = 0
    while i < n:
        ma = _max_chunk_allowed_at_pos(pos, T, max_c)
        tid = None
        L = 0
        upper = min(max_len, n - i)
        for L_try in range(upper, 0, -1):
            piece = bytes(b[i : i + L_try])
            if piece not in mr:
                continue
            r = mr[piece]
            if r >= len(chunk_counts):
                continue
            if int(chunk_counts[r]) <= ma:
                tid = r
                L = L_try
                break
        if tid is None:
            piece = bytes([b[i]])
            if piece not in mr:
                raise ValueError(f"No vocab token for byte {b[i]!r} at offset {i}")
            r = mr[piece]
            if int(chunk_counts[r]) > ma:
                raise ValueError(
                    f"Cannot satisfy chunk cap {ma} at output pos {pos} (byte {b[i]!r})"
                )
            tid = r
            L = 1
        out.append(tid)
        i += L
        pos += 1
    return out


def plot_bytes_per_cot_len(
    tokenizer_pkl_path: str,
    T_values: list[int],
    *,
    max_problems: int | None = None,
    max_x_chunk: int = 100,
    figsize=(10, 6),
):
    """
    NuminaMath-CoT: split each problem with ``GPT4_PATTERN.findall`` (GPT-4
    pretoken chunks). Plot **x** = 0 (empty prefix) and **10, 20, …** up to
    ``max_x_chunk``; **y** = dataset average of **token count** for that prefix
    (``x=0`` is always 0 tokens).

    One curve per schedule ``T`` (progressive cap on tokenizer-token chunk count
    while encoding). ``T <= 0`` uses ``encode_ordinary`` on each prefix.
    """
    enc = _load_encoding(tokenizer_pkl_path)
    chunk_counts = _chunk_counts_from_encoding(enc)
    if not enc._mergeable_ranks:
        raise ValueError("Empty tokenizer")

    chunk_stride = 10
    x_positions_stride = list(range(chunk_stride, max_x_chunk + 1, chunk_stride))
    if not x_positions_stride:
        raise ValueError(f"max_x_chunk ({max_x_chunk}) must be >= {chunk_stride}")
    x_plot = [0] + x_positions_stride

    fig, ax = plt.subplots(figsize=figsize)

    for T in T_values:
        totals = np.zeros(max_x_chunk + 1, dtype=np.float64)
        counts = np.zeros(max_x_chunk + 1, dtype=np.int64)
        for doc in iter_numina_math_cot_docs("train", max_problems=max_problems):
            pieces = GPT4_PATTERN.findall(doc)
            if not pieces:
                continue
            n_p = min(max_x_chunk, len(pieces))
            for x in range(chunk_stride, n_p + 1, chunk_stride):
                prefix = "".join(pieces[:x])
                try:
                    ids = _encode_cot_chunk_schedule(prefix, enc, chunk_counts, T)
                except ValueError:
                    continue
                totals[x] += len(ids)
                counts[x] += 1
        if not any(counts[x] > 0 for x in x_positions_stride):
            raise ValueError("No valid prefix encodings (dataset empty or all failed)")
        ys = [0.0] + [
            totals[x] / counts[x] if counts[x] > 0 else np.nan
            for x in x_positions_stride
        ]
        leg = "Full Vocab" if T == 0 else f"T={T}"
        ax.plot(x_plot, ys, marker="o", linewidth=2, label=leg)

    ax.set_xlabel("Text prefix length")
    ax.set_ylabel("Avg. tokens to encode that prefix")
    ax.set_title("NuminaMath-CoT: encoding length vs. pretoken chunk prefix (per T)")
    ax.legend(title="Schedule T")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def _iter_numina_problems(max_problems: int | None):
    from nanochat.dataset import load_numina_math_cot

    ds = load_numina_math_cot(split="train")
    for i, row in enumerate(ds):
        if max_problems is not None and i >= max_problems:
            break
        p = (row.get("problem") or "").strip()
        if p:
            yield p


def _entropy_from_logits(logits: "torch.Tensor") -> float:
    import torch
    import torch.nn.functional as F

    log_p = F.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    return float(-(p * log_p).sum().item())


def plot_cot_entropy(
    *,
    max_problems: int,
    model_name: str = "meta-llama/Llama-3.2-1B",
    max_new_tokens: int = 512,
    figsize=(10, 6),
):
    """
    NuminaMath-CoT: for each problem, generate a response with a Llama causal LM,
    then plot **token index** (0 = first generated token) vs. **entropy** of
    the next-token distribution at each step (softmax entropy over the vocab).

    All problems are drawn on one axes with semi-transparent lines.

    Default checkpoint ``meta-llama/Llama-3.2-1B``; uses chat template when
    present, else a plain ``Problem: / Solution:`` prompt. Gated models require
    HF login / ``HF_TOKEN``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    device = next(model.parameters()).device
    fig, ax = plt.subplots(figsize=figsize)

    for p_idx, problem in enumerate(_iter_numina_problems(max_problems)):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": problem}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            input_ids = tokenizer(
                f"Problem:\n{problem}\n\nSolution:",
                return_tensors="pt",
            ).input_ids.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        scores = out.scores
        if not scores:
            continue

        entropies = [_entropy_from_logits(s[0].float()) for s in scores]
        xs = np.arange(len(entropies))
        ax.plot(
            xs,
            entropies,
            alpha=0.35,
            linewidth=1.2,
            color=f"C{p_idx % 10}",
        )

    ax.set_xlabel("Generated token index")
    ax.set_ylabel("Entropy (nats, next-token distribution)")
    ax.set_title(
        f"Per-token entropy vs. position ({max_problems} problems, {model_name})"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BPE tokenizers with various plots.")
    parser.add_argument(
        "--eval",
        type=str,
        default="bytes_per_token_vs_vocab",
        choices=[
            "bytes_per_token",
            "chunk_distribution",
            "bytes_per_token_vs_vocab",
            "bytes_per_cot_len",
            "plot_cot_entropy",
        ],
        help="Which evaluation to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        help="Short dataset name (e.g. 'c4', 'nmc'). Used for chunk_distribution, bytes_per_token_vs_vocab, and as default for bytes_per_token when --datasets is omitted.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="For bytes_per_token only: one or more short names → {name}_dictionaries each. Default: single --dataset. With two names, shades between curves for each common tokenizer file.",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=50_000,
        help="For bytes_per_token_vs_vocab: number of documents/problems per split.",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="For bytes_per_cot_len: max NuminaMath-CoT problems (default: all). For plot_cot_entropy: number of problems (default: 5 if omitted).",
    )
    args = parser.parse_args()

    bpt_names = args.datasets if args.datasets is not None else [args.dataset]
    dictionaries_path = f"{args.dataset}_dictionaries"
    if args.eval not in ("bytes_per_token", "bytes_per_cot_len", "plot_cot_entropy"):
        os.makedirs(dictionaries_path, exist_ok=True)

    if args.eval in ("bytes_per_token", "bytes_per_cot_len", "plot_cot_entropy"):
        plots_dir = os.path.join("docs", "plots")
    else:
        plots_dir = os.path.join("docs", "plots", args.dataset)
    os.makedirs(plots_dir, exist_ok=True)

    if args.eval == "bytes_per_token":
        vocab_sizes = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
        dataset_paths = [(n, f"{n}_dictionaries") for n in bpt_names]
        fig = plot_bytes_per_token(dataset_paths, vocab_sizes)
        fig.show()
        plt.show()
        bpt_filename = f"bytes_per_token_{'_'.join(bpt_names)}.png"
        fig.savefig(os.path.join(plots_dir, bpt_filename))
    elif args.eval == "bytes_per_cot_len":
        tok_path = os.path.join(dictionaries_path, "bpe", "tokenizer.pkl")
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
        T_vals = [0, 2, 10, 50, 100]
        fig = plot_bytes_per_cot_len(
            tok_path,
            T_vals,
            max_problems=args.max_problems,
            max_x_chunk=100,
        )
        fig.show()
        plt.show()
        fig.savefig(os.path.join(plots_dir, f"bytes_per_cot_len_{args.dataset}.png"))
    elif args.eval == "plot_cot_entropy":
        fig = plot_cot_entropy(max_problems=args.max_problems)
        fig.show()
        plt.show()
        fig.savefig(os.path.join(plots_dir, "plot_cot_entropy.png"))
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

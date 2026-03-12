import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _byte_lens_from_file(path):
    """Load a tiktoken Encoding .pkl and return an array of per-token byte lengths."""
    with open(path, "rb") as f:
        enc = pickle.load(f)
    return np.array([len(b) for b in enc.token_byte_values()])


def plot_bytes_per_token(dictionaries_path):
    """
    For every tokenizer .pkl (tiktoken Encoding) found in subdirectories of
    dictionaries_path (e.g. bpe/, bpe_chunk/, bpe_superchunk/), plot vocab size
    vs. average bytes-per-token at 1k, 2k, 4k … 128k vocab sizes.
    Lines are colored by subfolder and styled by file.
    """
    vocab_sizes = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 127_000]
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

            xs = [vs for vs in vocab_sizes if vs <= len(byte_lens)]
            print(len(byte_lens))
            ys = [byte_lens[:vs].mean() for vs in xs]

            label = f"{subdir} / {os.path.splitext(fname)[0].lstrip('_')}"
            ls = linestyles[i % len(linestyles)]
            ax.plot(xs, ys, marker="o", linewidth=2, linestyle=ls, label=label, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels([f"{v // 1000}k" for v in vocab_sizes])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Avg bytes / token")
    ax.set_title("Tokenizer efficiency: avg bytes per token vs. vocab size")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    dictionaries_path = "c4_dictionaries"
    bytes_per_token = plot_bytes_per_token(dictionaries_path)
    bytes_per_token.show()
    plt.show()
    bytes_per_token.savefig('docs/plots/bytes_per_token.png')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
 
 
def plot_bytes_per_token(dictionaries_path):
    """
    For every tokenizer .pkl (tiktoken Encoding) in dictionaries_path, plot
    vocab size vs. average bytes-per-token at 1k, 2k, 4k … 128k vocab sizes.
    """
    vocab_sizes = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
 
    files = sorted(
        f for f in os.listdir(dictionaries_path)
        if os.path.isfile(os.path.join(dictionaries_path, f))
    )
 
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.tab10(np.linspace(0, 1, len(files)))
 
    for fname, color in zip(files, colors):
        enc = pickle.load(open(os.path.join(dictionaries_path, fname), "rb"))
        byte_lens = np.array([len(b) for b in enc.token_byte_values()])  # shape: (n_vocab,)
 
        xs = [vs for vs in vocab_sizes if vs <= len(byte_lens)]
        ys = [byte_lens[:vs].mean() for vs in xs]
 
        ax.plot(xs, ys, marker="o", linewidth=2, label=os.path.splitext(fname)[0], color=color)
 
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


"""Plot BPB results from LZ78 vs BPE ablation runs.

Parses training log files and generates comparison plots for tokenizer
ablations including BPB curves, bar charts, FLOP efficiency, and prefix
loss experiments. Outputs PNG files to the docs/ directory.
"""
import os
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(OUT_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'lz78': '#1f77b4',
    'freqgated': '#2ca02c',
    'trie2x': '#d62728',
    'bpe': '#ff7f0e',
}
EMB_STYLES = {'flat': '-', 'struct': '--', 'hier': ':'}
PREFIX_COLORS = {
    'standard': '#333333',
    'flat': '#333333',
    'prefix-d0.3': '#1f77b4',
    'prefix-d0.5': '#2ca02c',
    'prefix-d0.7': '#d62728',
    'prefix-interp0.2': '#ff7f0e',
    'prefix-bce': '#9467bd',
}


def get_color(name: str) -> str:
    for key, color in COLORS.items():
        if key in name:
            return color
    return '#333333'


def get_emb_style(name: str) -> str:
    for emb, style in EMB_STYLES.items():
        if emb in name:
            return style
    return '-'


def parse_logs() -> dict[str, dict]:
    """Parse all log files and extract BPB + loss + FLOPs data.

    For runs with multiple log files (from preempted restarts), keep only
    the log file with the most BPB data points.
    """
    # First pass: parse each log file separately, keyed by (run_name, job_id)
    per_file = {}

    for f in sorted(glob.glob(os.path.join(LOG_DIR, '*.out'))):
        basename = os.path.basename(f)
        # Extract run name and job ID: "lz78-32k-flat-c4-d12-1701433.out"
        m = re.match(r'^(.+)-(\d{7,})\.out$', basename)
        if not m:
            continue
        run_name = m.group(1)
        job_id = m.group(2)

        if run_name.startswith('pretok') or run_name.startswith('bpe-tok'):
            continue

        data = {'bpb': [], 'loss': [], 'flops_per_token': None,
                'num_params': None, 'batch_size': 524288, 'vocab_size': None}

        with open(f) as fh:
            for line in fh:
                bpb_m = re.search(r'Step (\d+) \| Validation bpb: ([\d.]+)', line)
                if bpb_m:
                    data['bpb'].append((int(bpb_m.group(1)), float(bpb_m.group(2))))

                loss_m = re.search(r'step (\d+)/\d+ .* loss: ([\d.]+)', line)
                if loss_m:
                    data['loss'].append((int(loss_m.group(1)), float(loss_m.group(2))))

                flops_m = re.search(r'Estimated FLOPs per token: ([\d.e+]+)', line)
                if flops_m:
                    data['flops_per_token'] = float(flops_m.group(1))

                params_m = re.search(r'Number of parameters: ([\d,]+)', line)
                if params_m:
                    data['num_params'] = int(params_m.group(1).replace(',', ''))

                vocab_m = re.search(r'Vocab size: ([\d,]+)', line)
                if vocab_m:
                    data['vocab_size'] = int(vocab_m.group(1).replace(',', ''))

                total_m = re.search(r'Total batch size ([\d,]+)', line)
                if total_m:
                    data['batch_size'] = int(total_m.group(1).replace(',', ''))

        per_file[(run_name, job_id)] = data

    # Second pass: for each run_name, keep only the file with most BPB points
    best_per_run = {}
    for (run_name, job_id), data in per_file.items():
        if run_name not in best_per_run or len(data['bpb']) > len(best_per_run[run_name]['bpb']):
            best_per_run[run_name] = data

    return best_per_run


def short_name(name: str) -> str:
    return name.replace('-c4-d12', '')


def is_prefix_run(name: str) -> bool:
    return 'prefix' in name or name.endswith('standard-c4-d12')


def is_main_run(name: str) -> bool:
    return not is_prefix_run(name)


def plot_bpb_comparison(runs: dict[str, dict]) -> None:
    """Plot BPB curves for all main ablation runs."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for name, data in sorted(runs.items()):
        if not is_main_run(name):
            continue
        bpb = data['bpb']
        if len(bpb) < 2:
            continue

        steps = [s for s, _ in bpb]
        vals = [v for _, v in bpb]
        color = get_color(name)
        style = get_emb_style(name)
        lw = 2.5 if 'flat' in name else 1.8
        label = short_name(name)

        ax.plot(steps, vals, style, color=color, linewidth=lw, marker='o',
                markersize=4, label=f'{label} (best: {min(vals):.4f})')

    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Validation BPB (bits per byte)', fontsize=13)
    ax.set_title('LZ78 vs BPE Tokenizer Ablations — Validation BPB on C4', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    # Zoom into the interesting range
    all_bpb = []
    for n, d in runs.items():
        if is_main_run(n) and len(d['bpb']) >= 2:
            all_bpb.extend([v for _, v in d['bpb']])
    if all_bpb:
        ax.set_ylim(min(all_bpb) - 0.05, min(max(all_bpb), 4.0))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'bpb_comparison.png')
    plt.savefig(path, dpi=150)
    print(f'Saved: {path}')
    plt.close()


def plot_by_tokenizer(runs: dict[str, dict]) -> None:
    """Plot BPB grouped by tokenizer type (main runs only)."""
    groups = [
        ('LZ78 (32K)', ['lz78-32k-flat-c4-d12', 'lz78-32k-struct-c4-d12']),
        ('FreqGated (32K)', ['freqgated-32k-flat-c4-d12', 'freqgated-32k-struct-c4-d12']),
        ('Trie 2x (44K)', ['trie2x-44k-flat-c4-d12', 'trie2x-44k-struct-c4-d12', 'trie2x-44k-hier-c4-d12']),
        ('BPE (32K)', ['bpe-32k-flat-c4-d12']),
    ]

    # Only include groups that have data
    active = [(t, ns) for t, ns in groups if any(n in runs and len(runs[n]['bpb']) >= 2 for n in ns)]
    if not active:
        return

    ncols = len(active)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    emb_colors = {'flat': '#1f77b4', 'struct': '#ff7f0e', 'hier': '#2ca02c'}

    for ax, (title, names) in zip(axes, active):
        for name in names:
            if name not in runs or len(runs[name]['bpb']) < 2:
                continue
            bpb = runs[name]['bpb']
            steps = [s for s, _ in bpb]
            vals = [v for _, v in bpb]
            emb = 'flat'
            for p in name.split('-'):
                if p in ('flat', 'struct', 'hier'):
                    emb = p
            ax.plot(steps, vals, '-o', color=emb_colors[emb], linewidth=2,
                    markersize=4, label=f'{emb} (best: {min(vals):.4f})')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Step', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Validation BPB', fontsize=13)
    fig.suptitle('Embedding Strategy Comparison by Tokenizer', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'bpb_by_tokenizer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def plot_best_per_run(runs: dict[str, dict]) -> None:
    """Horizontal bar chart of best BPB per run."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 2]})

    # Left: main ablation runs
    main_results = []
    for name, data in runs.items():
        if not is_main_run(name) or len(data['bpb']) < 2:
            continue
        best = min(v for _, v in data['bpb'])
        main_results.append((short_name(name), best, name))
    main_results.sort(key=lambda x: x[1])

    ax = axes[0]
    names = [r[0] for r in main_results]
    vals = [r[1] for r in main_results]
    bar_colors = [get_color(r[2]) for r in main_results]
    bars = ax.barh(range(len(names)), vals, color=bar_colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Best Validation BPB (lower is better)', fontsize=12)
    ax.set_title('Main Ablation Runs', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.4f}',
                va='center', fontsize=10, fontweight='bold')
    if vals:
        ax.set_xlim(min(vals) - 0.02, max(vals) + 0.03)
    ax.grid(True, alpha=0.3, axis='x')

    # Right: prefix loss runs
    prefix_results = []
    for name, data in runs.items():
        if not is_prefix_run(name) or len(data['bpb']) < 2:
            continue
        best = min(v for _, v in data['bpb'])
        prefix_results.append((short_name(name), best, name))
    prefix_results.sort(key=lambda x: x[1])

    ax = axes[1]
    if prefix_results:
        names = [r[0] for r in prefix_results]
        vals = [r[1] for r in prefix_results]
        bar_colors = [get_color(r[2]) for r in prefix_results]
        bars = ax.barh(range(len(names)), vals, color=bar_colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Best Validation BPB (lower is better)', fontsize=12)
        ax.set_title('Prefix Loss Runs', fontsize=13, fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.4f}',
                    va='center', fontsize=10, fontweight='bold')
        ax.set_xlim(min(vals) - 0.02, max(vals) + 0.03)
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No prefix run data yet', ha='center', va='center',
                fontsize=14, transform=ax.transAxes, color='gray')
        ax.set_title('Prefix Loss Runs', fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['lz78'], label='LZ78 (32K)'),
        Patch(facecolor=COLORS['freqgated'], label='FreqGated (32K)'),
        Patch(facecolor=COLORS['trie2x'], label='Trie 2x (44K)'),
        Patch(facecolor=COLORS['bpe'], label='BPE (32K)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Best BPB Achieved per Run (partial training @ ~39%)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(OUT_DIR, 'bpb_best_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def plot_loss_curves(runs: dict[str, dict]) -> None:
    """Plot training loss curves (subsampled, main runs only)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for name, data in sorted(runs.items()):
        if not is_main_run(name):
            continue
        loss = data['loss']
        if len(loss) < 50:
            continue

        # Subsample every 20 steps
        steps = [s for s, _ in loss[::20]]
        vals = [v for _, v in loss[::20]]

        color = get_color(name)
        lw = 2.0 if 'flat' in name else 1.2
        alpha = 1.0 if 'flat' in name else 0.7

        ax.plot(steps, vals, '-', color=color, linewidth=lw, alpha=alpha,
                label=f'{short_name(name)} (final: {loss[-1][1]:.3f})')

    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title('Training Loss Curves (Main Runs)', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'training_loss.png')
    plt.savefig(path, dpi=150)
    print(f'Saved: {path}')
    plt.close()


def plot_bpb_vs_flops(runs: dict[str, dict]) -> None:
    """Plot BPB vs cumulative training FLOPs (main runs only)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for name, data in sorted(runs.items()):
        if not is_main_run(name):
            continue
        bpb = data['bpb']
        if len(bpb) < 2 or data['flops_per_token'] is None:
            continue

        flops_per_token = data['flops_per_token']
        batch_size = data['batch_size']
        flops_exa = [s * batch_size * flops_per_token / 1e18 for s, _ in bpb]
        vals = [v for _, v in bpb]

        color = get_color(name)
        style = get_emb_style(name)
        lw = 2.5 if 'flat' in name else 1.8

        ax.plot(flops_exa, vals, style, color=color, linewidth=lw, marker='o',
                markersize=4, label=f'{short_name(name)} (best: {min(vals):.4f})')

    ax.set_xlabel('Cumulative Training FLOPs (exaFLOPs)', fontsize=13)
    ax.set_ylabel('Validation BPB (bits per byte)', fontsize=13)
    ax.set_title('BPB vs Training Compute', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    # Zoom into interesting range
    all_bpb = []
    for n, d in runs.items():
        if is_main_run(n) and len(d['bpb']) >= 2:
            all_bpb.extend([v for _, v in d['bpb']])
    if all_bpb:
        ax.set_ylim(min(all_bpb) - 0.05, min(max(all_bpb), 4.0))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'bpb_vs_flops.png')
    plt.savefig(path, dpi=150)
    print(f'Saved: {path}')
    plt.close()


def plot_bpb_vs_params(runs: dict[str, dict]) -> None:
    """Scatter: best BPB vs model parameters (main runs only)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for name, data in runs.items():
        if not is_main_run(name) or len(data['bpb']) < 2 or data['num_params'] is None:
            continue

        best = min(v for _, v in data['bpb'])
        label = short_name(name)
        params_m = data['num_params'] / 1e6
        color = get_color(name)
        marker = 'o' if 'flat' in name else ('s' if 'struct' in name else '^')

        ax.scatter(params_m, best, color=color, marker=marker, s=150, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(label, (params_m, best), textcoords="offset points",
                    xytext=(10, 6), fontsize=9, fontweight='bold')

    ax.set_xlabel('Total Parameters (M)', fontsize=13)
    ax.set_ylabel('Best Validation BPB', fontsize=13)
    ax.set_title('BPB vs Model Size', fontsize=14)
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS['lz78'], label='LZ78 (32K)'),
        Patch(facecolor=COLORS['freqgated'], label='FreqGated (32K)'),
        Patch(facecolor=COLORS['trie2x'], label='Trie 2x (44K)'),
        Patch(facecolor=COLORS['bpe'], label='BPE (32K)'),
        Line2D([0], [0], marker='o', color='gray', label='flat', markersize=8, linestyle=''),
        Line2D([0], [0], marker='s', color='gray', label='struct', markersize=8, linestyle=''),
        Line2D([0], [0], marker='^', color='gray', label='hier', markersize=8, linestyle=''),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'bpb_vs_params.png')
    plt.savefig(path, dpi=150)
    print(f'Saved: {path}')
    plt.close()


def plot_prefix_comparison(runs: dict[str, dict]) -> None:
    """Plot BPB for prefix loss experiments (LZ78 + BPE)."""
    # Collect LZ78 prefix runs + the LZ78 flat baseline
    lz78_runs = {}
    bpe_runs = {}
    for name, data in runs.items():
        sn = short_name(name)
        if len(data['bpb']) < 2:
            continue
        if 'lz78' in name and (is_prefix_run(name) or name == 'lz78-32k-flat-c4-d12'):
            lz78_runs[sn] = data
        if 'bpe' in name and (is_prefix_run(name) or name == 'bpe-32k-flat-c4-d12'):
            bpe_runs[sn] = data

    has_lz78 = len(lz78_runs) > 0
    has_bpe = len(bpe_runs) > 0
    ncols = has_lz78 + has_bpe
    if ncols == 0:
        # Nothing to plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No prefix run data yet', ha='center', va='center',
                fontsize=16, color='gray', transform=ax.transAxes)
        ax.set_title('Prefix Loss Experiments', fontsize=14)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, 'bpb_prefix_comparison.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f'Saved: {path}')
        return

    fig, axes = plt.subplots(1, max(ncols, 1), figsize=(8 * max(ncols, 1), 7))
    if ncols == 1:
        axes = [axes]

    panel_idx = 0
    for panel_title, group in [('LZ78 32K — Prefix Loss Ablation', lz78_runs),
                                ('BPE 32K — Prefix Loss Ablation', bpe_runs)]:
        if not group:
            continue
        ax = axes[panel_idx]
        panel_idx += 1

        for sn, data in sorted(group.items()):
            bpb = data['bpb']
            steps = [s for s, _ in bpb]
            vals = [v for _, v in bpb]

            # Determine color from suffix
            color = '#333333'
            for key, c in PREFIX_COLORS.items():
                if key in sn:
                    color = c
                    break

            lw = 2.5 if ('standard' in sn or sn.endswith('-flat')) else 2.0
            ax.plot(steps, vals, '-o', color=color, linewidth=lw, markersize=4,
                    label=f'{sn} (best: {min(vals):.4f})')

        ax.set_title(panel_title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Validation BPB', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Prefix Loss Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, 'bpb_prefix_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


if __name__ == '__main__':
    runs = parse_logs()

    print(f"\nFound {len(runs)} runs with data:")
    for name, data in sorted(runs.items()):
        bpb_count = len(data['bpb'])
        loss_count = len(data['loss'])
        best_bpb = min((v for _, v in data['bpb']), default=None)
        params = f"{data['num_params'] / 1e6:.1f}M" if data['num_params'] else "?"
        vocab = data.get('vocab_size', '?')
        print(f"  {name}: {bpb_count} BPB evals, {loss_count} loss steps, "
              f"best BPB={best_bpb}, params={params}, vocab={vocab}")

    print("\nGenerating plots...")
    plot_bpb_comparison(runs)
    plot_by_tokenizer(runs)
    plot_best_per_run(runs)
    plot_loss_curves(runs)
    plot_bpb_vs_flops(runs)
    plot_bpb_vs_params(runs)
    plot_prefix_comparison(runs)
    print("\nDone! Plots saved to docs/")

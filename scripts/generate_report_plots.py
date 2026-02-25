"""Generate plots for the LZ78 ablation study report.

Produces training curve comparisons, bar charts, embedding strategy plots,
prefix loss experiments, chunking ablations, and compute efficiency figures.
Outputs PNGs to both weezl and nanochat docs/plots directories.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directories
WEEZL_PLOTS = "/home/parsaidp/weezl/docs/plots"
NANOCHAT_PLOTS = "/home/parsaidp/nanochat/docs/plots"
os.makedirs(WEEZL_PLOTS, exist_ok=True)
os.makedirs(NANOCHAT_PLOTS, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# Data: BPB training curves (step -> BPB) from logs
# ============================================================

steps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
steps_extended = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5160]

# Full convergence curves — ALL runs now completed to full chinchilla-20 training
curves = {
    "FreqGated-flat":    [3.6870, 1.5010, 1.3548, 1.2965, 1.2614, 1.2416, 1.2275, 1.2192, 1.2116, 1.2077, 1.2028, 1.2000, 1.1998, 1.1967, 1.1956, 1.1966, 1.1957, 1.1932, 1.1868, 1.1820, 1.1767, 1.1756],  # 5156 steps
    "FreqGated-struct":  [3.687, 1.5234, 1.3702, 1.3078, 1.2714, 1.2512, 1.2356, 1.2264, 1.2186, 1.2148, 1.2092, 1.2068, 1.2050, 1.2047, 1.2001, 1.2002, 1.1984, 1.1955, 1.1905, 1.1844, 1.1788, 1.1768],  # 5163 steps
    "LZ78-flat":         [3.8059, 1.5557, 1.3989, 1.3331, 1.2943, 1.2721, 1.2554, 1.2447, 1.2376, 1.2314, 1.2277, 1.2263, 1.2214, 1.2225, 1.2185, 1.2213, 1.2169, 1.2142, 1.2087, 1.2018, 1.1965, 1.1952],  # 5133 steps
    "LZ78-struct":       [3.8057, 1.5754, 1.4078, 1.3382, 1.3005, 1.2759, 1.2602, 1.2492, 1.2418, 1.2367, 1.2318, 1.2288, 1.2257, 1.2226, 1.2202, 1.2214, 1.2201, 1.2177, 1.2082, 1.2019, 1.1964, 1.1944],  # 5141 steps
    "Trie2x-flat":       [3.5662, 1.5258, 1.3776, 1.3148, 1.2791, 1.2599, 1.2471, 1.2363, 1.2311, 1.2248, 1.2239, 1.2192, 1.2209, 1.2171, 1.2167, 1.2194, 1.2165, 1.2167, 1.2207, 1.2211, 1.2155, 1.2107],  # 5846 steps, converged
    "Trie2x-struct":     [3.5661, 1.6012, 1.4337, 1.3656, 1.3262, 1.3043, 1.2881, 1.2782, 1.2691, 1.2639, 1.2604, 1.2569, 1.2549, 1.2505, 1.2513, 1.2513, 1.2518, 1.2494, 1.2484, 1.2502, 1.2413, 1.2307],  # 5853 steps
    "Trie2x-hier":       [3.5661, 1.5985, 1.4346, 1.3659, 1.3270, 1.3051, 1.2892, 1.2782, 1.2723, 1.2645, 1.2613, 1.2560, 1.2560, 1.2513, 1.2509, 1.2520, 1.2520, 1.2496, 1.2500, 1.2458, 1.2418, 1.2306],  # 5853 steps
}

# BPE standard baseline — COMPLETED job 1703897, converged at 0.9433 BPB
bpe_baseline = [3.2415, 1.2395, 1.1300, 1.0884, 1.0617, 1.0461, 1.0332, 1.0229, 1.0143, 1.0072, 1.0015, 0.9971, 0.9926, 0.9889, 0.9848, 0.9819, 0.9786, 0.9726, 0.9638, 0.9543, 0.9462, 0.9433]

# BPE unchunked — COMPLETED job 1704607, 0.9434 BPB (confirms BPE baseline)
bpe_unchunked = [3.2415, 1.2394, 1.1299, 1.0882, 1.0619, 1.0465, 1.0331, 1.0230, 1.0147, 1.0075, 1.0019, 0.9976, 0.9924, 0.9893, 0.9850, 0.9820, 0.9789, 0.9729, 0.9636, 0.9545, 0.9464, 0.9434]

# Tuple embedding curves — ALL converged
tuple_curves = {
    "LZ78-tuple":        [3.8058, 1.6218, 1.4318, 1.3546, 1.3123, 1.2863, 1.2696, 1.2577, 1.2484, 1.2405, 1.2363, 1.2331, 1.2282, 1.2275, 1.2260, 1.2240, 1.2160, 1.2064, 1.1979, 1.1945],  # converged at step 4686 (20 points)
    "Trie2x-tuple":      [3.5662, 1.6548, 1.4549, 1.3806, 1.3372, 1.3132, 1.2973, 1.2852, 1.2781, 1.2688, 1.2673, 1.2623, 1.2584, 1.2547, 1.2557, 1.2540, 1.2537, 1.2498, 1.2420, 1.2381, 1.2309, 1.2279],  # 5220 steps
    "FreqGated-tuple":   [3.6870, 1.5759, 1.3914, 1.3219, 1.2831, 1.2602, 1.2441, 1.2341, 1.2255, 1.2193, 1.2138, 1.2107, 1.2078, 1.2067, 1.2033, 1.2034, 1.1950, 1.1882, 1.1800, 1.1762],  # 4703 steps (20 points)
}

# Prefix-smooth CE curves — ALL fully converged
prefsmooth_curves = {
    "LZ78-pw0.1":        [3.8059, 1.6093, 1.4615, 1.4034, 1.3586, 1.3419, 1.3257, 1.3149, 1.3082, 1.3010, 1.2975, 1.2893, 1.2864, 1.2893, 1.2871, 1.2807, 1.2799, 1.2778, 1.2716, 1.2661, 1.2596, 1.2581],  # 5133 steps
    "LZ78-pw0.5":        [3.8059, 1.8106, 1.6503, 1.5871, 1.5621, 1.5279, 1.5109, 1.4992, 1.5046, 1.4871, 1.4838, 1.4900, 1.4859, 1.4852, 1.4829, 1.4813, 1.4723, 1.4704, 1.4639, 1.4604, 1.4542, 1.4529],  # 5133 steps
    "LZ78-pw1.0":        [3.8059, 1.9251, 1.8074, 1.7230, 1.7034, 1.6655, 1.6637, 1.6486, 1.6301, 1.6335, 1.6197, 1.6167, 1.6265, 1.6246, 1.6222, 1.6091, 1.6198, 1.6086, 1.6008, 1.5998, 1.5930, 1.5924],  # 5133 steps
    "FG-pw0.1":          [3.6870, 1.5772, 1.4307, 1.3680, 1.3322, 1.3166, 1.2968, 1.2935, 1.2855, 1.2808, 1.2753, 1.2694, 1.2654, 1.2640, 1.2669, 1.2650, 1.2639, 1.2617, 1.2523, 1.2469, 1.2414, 1.2397],  # 5156 steps
    "FG-pw0.5":          [3.6870, 1.7615, 1.6327, 1.5700, 1.5222, 1.5046, 1.4998, 1.4808, 1.4830, 1.4781, 1.4719, 1.4608, 1.4578, 1.4577, 1.4557, 1.4560, 1.4627, 1.4581, 1.4512, 1.4438, 1.4382, 1.4367],  # 5156 steps
    "FG-pw1.0":          [3.6870, 1.9440, 1.7508, 1.7098, 1.6604, 1.6365, 1.6251, 1.6151, 1.6205, 1.6136, 1.6111, 1.6079, 1.5941, 1.6013, 1.5931, 1.5925, 1.5895, 1.5883, 1.5849, 1.5815, 1.5766, 1.5754],  # 5156 steps
    "BPE-pw0.1":         [3.2415, 1.3051, 1.1934, 1.1537, 1.1276, 1.1118, 1.0949, 1.0892, 1.0806, 1.0726, 1.0673, 1.0618, 1.0549, 1.0545, 1.0507, 1.0452, 1.0421, 1.0372, 1.0286, 1.0198, 1.0113, 1.0093],  # 5160 steps
    "BPE-pw0.5":         [3.2415, 1.4882, 1.3574, 1.3278, 1.3005, 1.2846, 1.2704, 1.2599, 1.2513, 1.2440, 1.2297, 1.2267, 1.2233, 1.2185, 1.2206, 1.2116, 1.2100, 1.2038, 1.2003, 1.1895, 1.1809, 1.1785],  # 5160 steps
    "BPE-pw1.0":         [3.2415, 1.5819, 1.4943, 1.4508, 1.4226, 1.4053, 1.3904, 1.3805, 1.3712, 1.3631, 1.3576, 1.3533, 1.3506, 1.3453, 1.3309, 1.3380, 1.3274, 1.3297, 1.3196, 1.3096, 1.3003, 1.2973],  # 5160 steps
}

# BPE-nochunk (retrained without regex chunking) — COMPLETED, converged at 0.9691 BPB
bpe_nochunk = [2.6921, 1.2736, 1.1626, 1.1177, 1.0918, 1.0744, 1.0606, 1.0509, 1.0420, 1.0343, 1.0290, 1.0238, 1.0193, 1.0149, 1.0107, 1.0076, 1.0050, 0.9994, 0.9896, 0.9801, 0.9722, 0.9691]  # 5160 steps, converged

# Chunking ablation curves — ALL converged
chunked_curves = {
    "LZ78-chunked":     [4.7440, 1.3995, 1.2748, 1.2264, 1.1989, 1.1805, 1.1668, 1.1573, 1.1511, 1.1449, 1.1414, 1.1375, 1.1350, 1.1312, 1.1307, 1.1294, 1.1298, 1.1243, 1.1166, 1.1095, 1.1032, 1.1016],  # 5133 steps
    "FreqGated-chunked":[4.4804, 1.3861, 1.2642, 1.2158, 1.1897, 1.1726, 1.1608, 1.1508, 1.1449, 1.1393, 1.1364, 1.1327, 1.1313, 1.1288, 1.1272, 1.1273, 1.1256, 1.1228, 1.1158, 1.1091, 1.1024, 1.0999],  # 5156 steps
    "Trie2x-chunked":   [4.9212, 1.4005, 1.2758, 1.2246, 1.1985, 1.1788, 1.1661, 1.1564, 1.1506, 1.1462, 1.1412, 1.1382, 1.1354, 1.1330, 1.1311, 1.1299, 1.1301, 1.1289, 1.1276, 1.1270, 1.1200, 1.1035],  # 5846 steps
}

prefix_curves = {
    "LZ78-flat (baseline)":  [3.806, 1.553, 1.400, 1.333, 1.294, 1.272, 1.256, 1.246, 1.239],
    "prefix-interp0.2":      [3.806, 1.566, 1.414, 1.351, 1.313, 1.290, 1.273, 1.263, 1.255],
    "prefix-d0.3":           [3.806, 1.626, 1.470, 1.409, 1.368, 1.345, 1.329, 1.319, 1.312],
    "prefix-d0.5":           [3.806, 1.704, 1.541, 1.485, 1.442, 1.427, 1.404, 1.401, 1.393],
    "prefix-d0.7":           [3.806, 1.770, 1.641, 1.576, 1.532, 1.510, 1.493, 1.494, 1.477],
}

# Colors
COLORS = {
    "FreqGated": "#2ecc71",
    "LZ78":      "#3498db",
    "Trie2x":    "#e74c3c",
    "BPE":       "#f39c12",
}

EMB_STYLES = {
    "flat": "-",
    "struct": "--",
    "hier": ":",
    "tuple": "-.",
}

def save(fig: "plt.Figure", name: str) -> None:
    for d in [WEEZL_PLOTS, NANOCHAT_PLOTS]:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight')
    print(f"Saved: {name}")


# ============================================================
# Plot 1: Main training curves (all tokenizer x embedding) — FULL CONVERGENCE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
for name, bpb in curves.items():
    tok = name.split("-")[0]
    emb = name.split("-")[1]
    color = COLORS.get(tok, "#999")
    style = EMB_STYLES.get(emb, "-")
    s = steps_extended[:len(bpb)]
    ax.plot(s, bpb, style, color=color, linewidth=2, label=name, marker='o', markersize=3)

# Add BPE baseline for reference
ax.plot(steps_extended[:len(bpe_baseline)], bpe_baseline, '-', color=COLORS["BPE"], linewidth=2.5, label='BPE-std CE', marker='o', markersize=3)

ax.set_xlabel("Training Step")
ax.set_ylabel("Validation BPB (bits per byte)")
ax.set_title("Training Curves: Tokenizer × Embedding (Full Convergence)")
ax.set_xlim(0, 5500)
ax.set_ylim(0.92, 1.65)
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
fig.tight_layout()
save(fig, "training_curves.png")
plt.close()


# ============================================================
# Plot 2: Final BPB bar chart — CONVERGED VALUES
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
all_final = {name: bpb[-1] for name, bpb in curves.items()}
all_final["BPE-std CE"] = bpe_baseline[-1]
names_sorted = sorted(all_final.keys(), key=lambda k: all_final[k])
final_bpb = [all_final[n] for n in names_sorted]
bar_colors = []
for n in names_sorted:
    tok = n.split("-")[0]
    bar_colors.append(COLORS.get(tok, "#999"))

bars = ax.barh(range(len(names_sorted)), final_bpb, color=bar_colors, edgecolor='white', height=0.6)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted)
ax.set_xlabel("Converged Validation BPB (lower is better)")
ax.set_title("Final BPB — All Runs Converged")
ax.set_xlim(0.92, 1.26)

for i, (bar, val) in enumerate(zip(bars, final_bpb)):
    ax.text(val + 0.002, i, f"{val:.4f}", va='center', fontsize=9)

fig.tight_layout()
save(fig, "bpb_comparison_bar.png")
plt.close()


# ============================================================
# Plot 3: Embedding method comparison (grouped by tokenizer)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

tokenizer_groups = {
    "LZ78 32K": {"flat": 1.1952, "structured": 1.1944, "tuple": 1.1945},
    "FreqGated 32K": {"flat": 1.1756, "structured": 1.1768, "tuple": 1.1762},
    "Trie2x 44K": {"flat": 1.2107, "structured": 1.2307, "tuple": 1.2279, "hierarchical": 1.2306},
}

for ax, (tok_name, emb_data) in zip(axes, tokenizer_groups.items()):
    tok_key = tok_name.split()[0]
    color = COLORS.get(tok_key, "#999")
    emb_names = list(emb_data.keys())
    emb_vals = list(emb_data.values())
    bars = ax.bar(emb_names, emb_vals, color=color, edgecolor='white', width=0.5, alpha=0.85)
    ax.set_title(tok_name, fontweight='bold')
    ax.set_ylim(1.17, 1.25)
    for bar, val in zip(bars, emb_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.4f}",
                ha='center', va='bottom', fontsize=9)

axes[0].set_ylabel("Converged Validation BPB")
fig.suptitle("Embedding Strategy Comparison by Tokenizer", fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "embedding_comparison.png")
plt.close()


# ============================================================
# Plot 4: Old prefix loss comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

prefix_colors = {
    "LZ78-flat (baseline)": "#3498db",
    "prefix-interp0.2":     "#9b59b6",
    "prefix-d0.3":          "#e67e22",
    "prefix-d0.5":          "#e74c3c",
    "prefix-d0.7":          "#c0392b",
}

for name, bpb in prefix_curves.items():
    color = prefix_colors[name]
    lw = 2.5 if "baseline" in name else 1.8
    style = "-" if "baseline" in name else "--"
    ax.plot(steps, bpb, style, color=color, linewidth=lw, label=name, marker='o', markersize=4)

ax.set_xlabel("Training Step")
ax.set_ylabel("Validation BPB (bits per byte)")
ax.set_title("Old Prefix Loss Experiments (all deprecated)")
ax.set_xlim(0, 2100)
ax.set_ylim(1.20, 1.52)
ax.legend(loc='upper right', framealpha=0.9)

# Add annotation
ax.annotate("More prefix weight\n= worse performance",
            xy=(1500, 1.42), fontsize=9, fontstyle='italic', color='#666',
            ha='center')

fig.tight_layout()
save(fig, "old_prefix_loss.png")
plt.close()


# ============================================================
# Plot 5: Prefix-smooth CE — what the label looks like
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Example: token "hello" with 4 prefix ancestors
token_names = ['"h"', '"he"', '"hel"', '"hell"', '"hello"']
n_tokens = len(token_names)

pw_configs = [
    ("Standard CE\n(pw=0, one-hot)", 0.0),
    ("Prefix Smooth\n(pw=0.1, mild)", 0.1),
    ("Prefix Smooth\n(pw=1.0, uniform)", 1.0),
]

for ax, (title, pw) in zip(axes, pw_configs):
    if pw == 0:
        weights = [0, 0, 0, 0, 1.0]
    else:
        raw = [pw, pw, pw, pw, 1.0]
        total = sum(raw)
        weights = [w/total for w in raw]

    colors = ['#bdc3c7'] * 4 + ['#2ecc71']
    if pw > 0:
        colors = ['#85c1e9'] * 4 + ['#2ecc71']

    bars = ax.bar(range(n_tokens), weights, color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(token_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=11)

    for bar, val in zip(bars, weights):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.2f}", ha='center', va='bottom', fontsize=8)

axes[0].set_ylabel("Label weight")
fig.suptitle('Target Label Distribution: "hello" as next token', fontsize=13, y=1.02)
fig.tight_layout()
save(fig, "prefix_smooth_label.png")
plt.close()


# ============================================================
# Plot 6: Tokenizer ranking summary
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# All converged results — horizontal bar chart sorted by BPB
tok_data = [
    ("BPE std CE",        0.9433, COLORS["BPE"],       0.85),
    ("BPE unchunked",     0.9434, COLORS["BPE"],       0.60),
    ("BPE pw=0.1",        1.0093, COLORS["BPE"],       0.40),
    ("BPE nochunk",       0.9691, COLORS["BPE"],       0.40),
    ("FG chunked",        1.0999, COLORS["FreqGated"], 0.85),
    ("LZ78 chunked",      1.1016, COLORS["LZ78"],      0.85),
    ("Trie2x chunked",    1.1035, COLORS["Trie2x"],    0.85),
    ("FreqGated flat",    1.1756, COLORS["FreqGated"], 0.55),
    ("FG tuple",          1.1762, COLORS["FreqGated"], 0.40),
    ("FG struct",         1.1768, COLORS["FreqGated"], 0.40),
    ("LZ78 struct",       1.1944, COLORS["LZ78"],      0.40),
    ("LZ78 tuple",        1.1945, COLORS["LZ78"],      0.40),
    ("LZ78 flat",         1.1952, COLORS["LZ78"],      0.55),
    ("Trie2x flat",       1.2107, COLORS["Trie2x"],    0.55),
    ("Trie2x tuple",      1.2279, COLORS["Trie2x"],    0.40),
    ("Trie2x hier",       1.2306, COLORS["Trie2x"],    0.55),
    ("Trie2x struct",     1.2307, COLORS["Trie2x"],    0.40),
]

tok_names = [d[0] for d in tok_data]
tok_bpb = [d[1] for d in tok_data]
tok_colors_list = [d[2] for d in tok_data]
tok_alphas = [d[3] for d in tok_data]

bars = ax.barh(range(len(tok_names)), tok_bpb, color=tok_colors_list, edgecolor='white', height=0.6)
for bar, alpha in zip(bars, tok_alphas):
    bar.set_alpha(alpha)

ax.set_yticks(range(len(tok_names)))
ax.set_yticklabels(tok_names, fontsize=9)
ax.set_xlabel("Converged Validation BPB (lower is better)")
ax.set_title("All Runs Ranked by Converged BPB")
ax.set_xlim(0.92, 1.26)
ax.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, tok_bpb)):
    ax.text(val + 0.002, i, f"{val:.4f}", va='center', fontsize=9)

fig.tight_layout()
save(fig, "tokenizer_ranking.png")
plt.close()


# ============================================================
# Plot 7: Convergence speed (steps to reach BPB thresholds)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

thresholds = [1.40, 1.35, 1.30, 1.28, 1.26, 1.25, 1.24, 1.23, 1.22, 1.20]

for name, bpb in curves.items():
    tok = name.split("-")[0]
    emb = name.split("-")[1]
    if emb != "flat":
        continue
    color = COLORS.get(tok, "#999")
    curve_steps = steps_extended[:len(bpb)]
    steps_to_thresh = []
    for thr in thresholds:
        reached = None
        for s, b in zip(curve_steps, bpb):
            if b <= thr:
                reached = s
                break
        steps_to_thresh.append(reached)

    valid_thresh = [t for t, s in zip(thresholds, steps_to_thresh) if s is not None]
    valid_steps = [s for s in steps_to_thresh if s is not None]
    ax.plot(valid_thresh, valid_steps, '-o', color=color, linewidth=2, markersize=5, label=name)

ax.set_xlabel("BPB Threshold")
ax.set_ylabel("Steps to Reach Threshold")
ax.set_title("Convergence Speed: Steps to Reach BPB Milestones (flat embedding only)")
ax.invert_xaxis()
ax.legend(loc='upper left', framealpha=0.9)
fig.tight_layout()
save(fig, "convergence_speed.png")
plt.close()


# ============================================================
# Plot 8: Tuple Embedding Comparison
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, tok in zip(axes, ["LZ78", "FreqGated", "Trie2x"]):
    color = COLORS[tok]
    flat_key = f"{tok}-flat"
    tuple_key = f"{tok}-tuple"
    struct_key = f"{tok}-struct"

    # flat baseline
    fc = curves[flat_key]
    ax.plot(steps_extended[:len(fc)], fc, '-o', color=color, linewidth=2, markersize=3, label='flat')
    # structured
    sc = curves[struct_key]
    ax.plot(steps_extended[:len(sc)], sc, '--s', color=color, linewidth=1.5, markersize=3, alpha=0.7, label='structured')
    # tuple
    tc = tuple_curves[tuple_key]
    t_steps = steps_extended[:len(tc)] if len(tc) > len(steps) else steps[:len(tc)]
    ax.plot(t_steps, tc, ':^', color=color, linewidth=2, markersize=4, label='tuple')

    if tok == "Trie2x":
        hier_key = f"{tok}-hier"
        hc = curves[hier_key]
        ax.plot(steps_extended[:len(hc)], hc, '-.d', color=color, linewidth=1.5, markersize=3, alpha=0.5, label='hierarchical')

    ax.set_xlabel("Step")
    ax.set_title(f"{tok}")
    ax.set_ylim(1.16, 1.35)
    ax.set_xlim(0, 5500)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

axes[0].set_ylabel("Validation BPB")
fig.suptitle("Embedding Strategy Comparison: flat vs structured vs tuple", fontsize=13, y=1.02)
fig.tight_layout()
save(fig, "tuple_comparison.png")
plt.close()


# ============================================================
# Plot 9: Prefix-Smooth CE Results
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, (tok, prefix) in zip(axes, [("LZ78", "LZ78"), ("FreqGated", "FG"), ("BPE", "BPE")]):
    color = COLORS[tok]

    # baseline (flat, standard CE)
    if tok in ["LZ78", "FreqGated"]:
        flat_key = f"{tok}-flat"
        fc = curves[flat_key]
        ax.plot(steps_extended[:len(fc)], fc, '-o', color=color, linewidth=2, markersize=3, label='standard CE')
    elif tok == "BPE":
        ax.plot(steps_extended[:len(bpe_baseline)], bpe_baseline, '-o', color=color, linewidth=2, markersize=3, label='standard CE')

    # prefix-smooth variants
    for pw, ls, alpha in [("0.1", "--", 0.8), ("0.5", "-.", 0.6), ("1.0", ":", 0.5)]:
        key = f"{prefix}-pw{pw}"
        if key in prefsmooth_curves:
            pc = prefsmooth_curves[key]
            ps = steps_extended[:len(pc)]
            ax.plot(ps, pc, ls, color=color, linewidth=1.5, alpha=alpha,
                    markersize=4, label=f'pw={pw}')

    ax.set_xlabel("Step")
    ax.set_title(f"{tok}")
    ax.set_ylim(0.92, 1.75)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

axes[0].set_ylabel("Validation BPB")
fig.suptitle("Prefix-Smooth CE: Effect of prefix_weight on BPB", fontsize=13, y=1.02)
fig.tight_layout()
save(fig, "prefsmooth_results.png")
plt.close()


# ============================================================
# Plot 10: Compute Efficiency — BPB vs cumulative FLOPs
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Data: FLOPs per step for each tokenizer (flat embedding only)
flops_per_step = {
    "FreqGated-flat": 4.65e14,
    "Trie2x-flat":    4.93e14,
    "LZ78-flat":      4.64e14,
}

# Corpus-level bytes per token (measured on C4 train set, NOT vocabulary average)
bytes_per_token = {
    "FreqGated-flat": 4.02,
    "Trie2x-flat":    4.28,
    "LZ78-flat":      3.90,
    "BPE-flat":       4.53,
}

batch_tokens = 524288

# Left panel: BPB vs cumulative FLOPs (training curves in FLOP-space)
ax = axes[0]
for name in ["FreqGated-flat", "Trie2x-flat", "LZ78-flat"]:
    bpb = curves[name]
    fps = flops_per_step[name]
    curve_steps = steps_extended[:len(bpb)]
    cumulative_flops = [s * fps for s in curve_steps]
    tok = name.split("-")[0]
    color = COLORS.get(tok, "#999")
    ax.plot([f/1e18 for f in cumulative_flops], bpb, '-o', color=color,
            linewidth=2, markersize=3, label=name)

ax.set_xlabel("Cumulative FLOPs (×10¹⁸)")
ax.set_ylabel("Validation BPB")
ax.set_title("BPB vs Compute (FLOP-space)")
ax.set_ylim(1.10, 1.65)
ax.legend(loc='upper right', framealpha=0.9)

# Right panel: Bytes processed per FLOP — bar chart
ax = axes[1]
tok_names = ["BPE 32K", "Trie2x 44K", "FreqGated 32K", "LZ78 32K"]
tok_keys = ["BPE-flat", "Trie2x-flat", "FreqGated-flat", "LZ78-flat"]
flops_per_token_vals = [8.9e8, 9.41e8, 8.87e8, 8.85e8]

bytes_per_flop = []
for key, fpt in zip(tok_keys, flops_per_token_vals):
    bpt = bytes_per_token[key]
    bytes_per_flop.append(bpt / fpt * 1e9)  # bytes per 1e9 FLOPs (GFLOPs)

bar_colors_eff = [COLORS["BPE"], COLORS["Trie2x"], COLORS["FreqGated"], COLORS["LZ78"]]
bars = ax.barh(range(len(tok_names)), bytes_per_flop, color=bar_colors_eff,
               edgecolor='white', height=0.5, alpha=0.85)
ax.set_yticks(range(len(tok_names)))
ax.set_yticklabels(tok_names, fontsize=11)
ax.set_xlabel("Bytes per GFLOP (higher = more efficient)")
ax.set_title("Text Throughput per FLOP")

for i, (bar, val) in enumerate(zip(bars, bytes_per_flop)):
    ax.text(val + 0.05, i, f"{val:.2f}", va='center', fontsize=10)

fig.tight_layout()
save(fig, "compute_efficiency.png")
plt.close()


# ============================================================
# Plot 9: Bytes per step comparison (stacked context)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

tok_labels = ["BPE 32K\n(4.53 B/tok)", "Trie2x 44K\n(4.28 B/tok)",
              "FreqGated 32K\n(4.02 B/tok)", "LZ78 32K\n(3.90 B/tok)"]
bytes_per_step_vals = [
    batch_tokens * 4.53,
    batch_tokens * 4.28,
    batch_tokens * 4.02,
    batch_tokens * 3.90,
]
bar_colors_bps = [COLORS["BPE"], COLORS["Trie2x"], COLORS["FreqGated"], COLORS["LZ78"]]

bars = ax.bar(range(len(tok_labels)), [b/1e6 for b in bytes_per_step_vals],
              color=bar_colors_bps, edgecolor='white', width=0.5, alpha=0.85)
ax.set_xticks(range(len(tok_labels)))
ax.set_xticklabels(tok_labels, fontsize=10)
ax.set_ylabel("Megabytes of text per training step")
ax.set_title("Text Processed per Training Step (batch_size=524K tokens)")

for bar, val in zip(bars, bytes_per_step_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val/1e6:.1f} MB", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add relative labels
ref = bytes_per_step_vals[-1]
for i, (bar, val) in enumerate(zip(bars, bytes_per_step_vals)):
    pct = (val / ref - 1) * 100
    if pct > 0.5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f"+{pct:.0f}%", ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')

fig.tight_layout()
save(fig, "bytes_per_step.png")
plt.close()


# ============================================================
# Plot 12: BPE Full Convergence Curves (dedicated)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# BPE standard baseline
ax.plot(steps_extended, bpe_baseline, '-o', color='#c0392b', linewidth=2.5, markersize=4, label=f'standard CE → {bpe_baseline[-1]:.4f}')

bpe_colors = {"BPE-pw0.1": "#e67e22", "BPE-pw0.5": "#f39c12", "BPE-pw1.0": "#f1c40f"}
bpe_labels = {"BPE-pw0.1": "pw=0.1 → 1.0093", "BPE-pw0.5": "pw=0.5 → 1.1785", "BPE-pw1.0": "pw=1.0 → 1.2973"}

for key in ["BPE-pw0.1", "BPE-pw0.5", "BPE-pw1.0"]:
    pc = prefsmooth_curves[key]
    ps = steps_extended[:len(pc)]
    ax.plot(ps, pc, '--o', color=bpe_colors[key], linewidth=1.8, markersize=4, label=bpe_labels[key])

# BPE-nochunk (retrained without regex)
ax.plot(steps_extended[:len(bpe_nochunk)], bpe_nochunk, ':s', color='#8e44ad', linewidth=1.8, markersize=4, label=f'nochunk → {bpe_nochunk[-1]:.4f}')

# Reference lines for LZ78-family best results
ax.axhline(y=1.0999, color=COLORS["FreqGated"], alpha=0.4, linestyle=':', linewidth=1.5)
ax.text(100, 1.1049, "FG-chunked: 1.0999", fontsize=8, color=COLORS["FreqGated"], alpha=0.8)

ax.set_xlabel("Training Step")
ax.set_ylabel("Validation BPB (bits per byte)")
ax.set_title("BPE: Standard CE vs Prefix-Smooth CE (5160 steps)")
ax.set_xlim(0, 5400)
ax.set_ylim(0.92, 1.65)
ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

# Mark convergence points
ax.plot(5160, 0.9433, '*', color='#c0392b', markersize=14, zorder=5)
for key, marker_y in [("BPE-pw0.1", 1.0093), ("BPE-pw0.5", 1.1785), ("BPE-pw1.0", 1.2973)]:
    ax.plot(5160, marker_y, '*', color=bpe_colors[key], markersize=12, zorder=5)

fig.tight_layout()
save(fig, "bpe_convergence.png")
plt.close()


# ============================================================
# Plot 13: Prefix Weight Sensitivity (all tokenizers)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

pw_values = [0, 0.1, 0.5, 1.0]

# All converged baselines (pw=0 is standard CE)
lz78_baseline = 1.1952  # converged
fg_baseline = 1.1756  # converged
bpe_baseline_val = 0.9433

lz78_pw = [lz78_baseline, 1.2581, 1.4529, 1.5924]  # ALL converged
fg_pw = [fg_baseline, 1.2397, 1.4367, 1.5754]  # ALL converged
bpe_pw = [bpe_baseline_val, 1.0093, 1.1785, 1.2973]  # converged

ax.plot(pw_values, lz78_pw, '-o', color=COLORS["LZ78"], linewidth=2, markersize=8, label='LZ78')
ax.plot(pw_values, fg_pw, '-s', color=COLORS["FreqGated"], linewidth=2, markersize=8, label='FreqGated')
ax.plot(pw_values, bpe_pw, '-^', color=COLORS["BPE"], linewidth=2, markersize=8, label='BPE')

ax.set_xlabel("Prefix Weight (0 = standard CE)", fontsize=12)
ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
ax.set_title("Prefix Weight Sensitivity: BPB vs prefix_weight")
ax.set_xticks(pw_values)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.90, 1.70)
ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

# Annotate key insight
ax.annotate("BPE std CE best overall:\n0.9433 BPB",
            xy=(0, 0.9433), xytext=(0.35, 0.97),
            fontsize=9, fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#666'),
            color='#666')

ax.annotate("Prefix smoothing hurts\nall tokenizers",
            xy=(0.5, 1.3), fontsize=9, fontstyle='italic', color='#999',
            ha='center')

fig.tight_layout()
save(fig, "prefix_weight_sensitivity.png")
plt.close()


# ============================================================
# Plot 14: Grand Comparison — Best per Tokenizer Family
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# All runs now converged — grand comparison with full curves
grand = {
    "BPE-std CE":         (bpe_baseline, steps_extended, '#c0392b', '-', 2.5),
    "BPE-nochunk":        (bpe_nochunk, steps_extended, '#e67e22', '-.', 1.5),
    "FG-chunked":         (chunked_curves["FreqGated-chunked"], steps_extended, COLORS["FreqGated"], '-', 2.5),
    "LZ78-chunked":       (chunked_curves["LZ78-chunked"], steps_extended, COLORS["LZ78"], '-', 2),
    "Trie2x-chunked":     (chunked_curves["Trie2x-chunked"], steps_extended, COLORS["Trie2x"], '-', 2),
    "FreqGated-flat":     (curves["FreqGated-flat"], steps_extended, COLORS["FreqGated"], '--', 1.5),
    "LZ78-flat":          (curves["LZ78-flat"], steps_extended, COLORS["LZ78"], '--', 1.5),
    "Trie2x-flat":        (curves["Trie2x-flat"], steps_extended, COLORS["Trie2x"], '--', 1.5),
}

for name, (bpb, st, color, ls, lw) in grand.items():
    s = st[:len(bpb)]
    ax.plot(s, bpb, ls, color=color, linewidth=lw, label=name, marker='o', markersize=3)

# Mark final converged values
final_marks = [
    ("BPE", 0.9433, '#c0392b'),
    ("FG-chunked", 1.0999, COLORS["FreqGated"]),
    ("LZ78-chunked", 1.1016, COLORS["LZ78"]),
    ("T2x-chunked", 1.1035, COLORS["Trie2x"]),
    ("FG-flat", 1.1756, COLORS["FreqGated"]),
    ("LZ78-flat", 1.1952, COLORS["LZ78"]),
    ("T2x-flat", 1.2107, COLORS["Trie2x"]),
]
for name, val, color in final_marks:
    ax.plot(5160, val, '*', color=color, markersize=12, zorder=5)
    ax.text(5200, val, f"{val:.4f}", fontsize=8, fontweight='bold', color=color, va='center')

ax.set_xlabel("Training Step")
ax.set_ylabel("Validation BPB (bits per byte)")
ax.set_title("Grand Comparison: All Tokenizers (Full Convergence)")
ax.set_xlim(0, 5800)
ax.set_ylim(0.90, 1.60)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

fig.tight_layout()
save(fig, "grand_comparison.png")
plt.close()


# ============================================================
# Plot 15: Chunking Ablation — chunked vs unchunked
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, tok in zip(axes, ["LZ78", "FreqGated", "Trie2x"]):
    color = COLORS[tok]
    flat_key = f"{tok}-flat"
    chunked_key = f"{tok}-chunked"

    # Unchunked baseline (flat) — full convergence
    fc = curves[flat_key]
    ax.plot(steps_extended[:len(fc)], fc, '-o', color=color, linewidth=2, markersize=3, label='unchunked (flat)')
    # Chunked — full convergence
    cc = chunked_curves[chunked_key]
    ax.plot(steps_extended[:len(cc)], cc, '--s', color=color, linewidth=2, markersize=3, alpha=0.8, label='chunked (flat)')

    # Show converged improvement
    unchunked_val = fc[-1]
    chunked_val = cc[-1]
    improvement = (unchunked_val - chunked_val) / unchunked_val * 100
    ax.annotate(f'{improvement:.1f}% better\n({chunked_val:.4f} vs {unchunked_val:.4f})',
                xy=(5000, chunked_val), xytext=(2500, chunked_val + 0.03),
                fontsize=8, fontstyle='italic', color=color,
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.5))

    ax.set_xlabel("Step")
    ax.set_title(f"{tok}")
    ax.set_ylim(1.08, 1.40)
    ax.set_xlim(0, 5500)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

axes[0].set_ylabel("Validation BPB")
fig.suptitle("Chunking Ablation: Chunked vs Unchunked (Full Convergence)", fontsize=13, y=1.02)
fig.tight_layout()
save(fig, "chunking_ablation.png")
plt.close()


# ============================================================
# Plot 16: Chunking + BPE comparison bar chart
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

# All results at CONVERGENCE, including BPE unchunked
bar_data = [
    ("BPE std CE", bpe_baseline[-1], COLORS["BPE"], 0.85),
    ("BPE unchunked", bpe_unchunked[-1], COLORS["BPE"], 0.60),
    ("FG chunked", chunked_curves["FreqGated-chunked"][-1], COLORS["FreqGated"], 0.85),
    ("LZ78 chunked", chunked_curves["LZ78-chunked"][-1], COLORS["LZ78"], 0.85),
    ("Trie2x chunked", chunked_curves["Trie2x-chunked"][-1], COLORS["Trie2x"], 0.85),
    ("FG unchunked", curves["FreqGated-flat"][-1], COLORS["FreqGated"], 0.45),
    ("LZ78 unchunked", curves["LZ78-flat"][-1], COLORS["LZ78"], 0.45),
    ("Trie2x unchunked", curves["Trie2x-flat"][-1], COLORS["Trie2x"], 0.45),
]

bar_data_sorted = sorted(bar_data, key=lambda x: x[1])
names = [d[0] for d in bar_data_sorted]
vals = [d[1] for d in bar_data_sorted]
colors = [d[2] for d in bar_data_sorted]
alphas = [d[3] for d in bar_data_sorted]

bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white', height=0.6)
for bar, alpha in zip(bars, alphas):
    bar.set_alpha(alpha)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("Converged Validation BPB (lower is better)")
ax.set_title("Chunking Impact: All Tokenizers (Converged)")
ax.set_xlim(0.92, 1.26)

for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.text(val + 0.002, i, f"{val:.4f}", va='center', fontsize=9)

fig.tight_layout()
save(fig, "chunking_comparison_bar.png")
plt.close()


print("\nAll plots generated!")
print(f"  Weezl:   {WEEZL_PLOTS}/")
print(f"  Nanochat: {NANOCHAT_PLOTS}/")

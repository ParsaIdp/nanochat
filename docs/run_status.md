# Run Status & Maintenance

**Last updated**: 2026-02-21 02:00
**Status**: ALL 26 EXPERIMENTS COMPLETE
**Target steps**: ~5,133 per run (Chinchilla-20 ratio)
**Model**: depth=12, n_embd=768, n_head=6, ~134M params

---

## Study Complete — No Active Runs

All 26 experiments have fully converged to Chinchilla-20 training. No runs pending or in progress.

---

## Final Results Summary (sorted by BPB)

| Rank | Run | BPB | vs BPE | Category |
|------|-----|-----|--------|----------|
| 1 | BPE std CE | **0.9433** | — | Baseline |
| 2 | BPE unchunked | 0.9434 | +0.01% | Control |
| 3 | BPE nochunk | 0.9691 | +2.7% | No-regex ablation |
| 4 | BPE pw=0.1 | 1.0093 | +7.0% | Prefix-smooth |
| 5 | FG chunked | 1.0999 | +16.6% | Chunking |
| 6 | LZ78 chunked | 1.1016 | +16.8% | Chunking |
| 7 | Trie2x chunked | 1.1035 | +17.0% | Chunking |
| 8 | FG flat | 1.1756 | +24.6% | Baseline |
| 9 | FG tuple | 1.1762 | +24.7% | Embedding |
| 10 | FG struct | 1.1768 | +24.7% | Embedding |
| 11 | BPE pw=0.5 | 1.1785 | +24.9% | Prefix-smooth |
| 12 | LZ78 struct | 1.1944 | +26.6% | Embedding |
| 13 | LZ78 tuple | 1.1945 | +26.6% | Embedding |
| 14 | LZ78 flat | 1.1952 | +26.7% | Baseline |
| 15 | Trie2x flat | 1.2107 | +28.3% | Baseline |
| 16 | Trie2x tuple | 1.2279 | +30.2% | Embedding |
| 17 | Trie2x hier | 1.2306 | +30.5% | Embedding |
| 18 | Trie2x struct | 1.2307 | +30.5% | Embedding |
| 19 | FG pw=0.1 | 1.2397 | +31.4% | Prefix-smooth |
| 20 | LZ78 pw=0.1 | 1.2581 | +33.4% | Prefix-smooth |
| 21 | BPE pw=1.0 | 1.2973 | +37.5% | Prefix-smooth |
| 22 | FG pw=0.5 | 1.4367 | +52.3% | Prefix-smooth |
| 23 | LZ78 pw=0.5 | 1.4529 | +54.0% | Prefix-smooth |
| 24 | FG pw=1.0 | 1.5754 | +67.0% | Prefix-smooth |
| 25 | LZ78 pw=1.0 | 1.5924 | +68.8% | Prefix-smooth |

---

## Key Findings

1. **BPE standard CE = 0.9433** — best overall by a wide margin
2. **BPE nochunk = 0.9691** — regex chunking adds only 2.7% for BPE; the merge algorithm alone still dominates
3. **Chunking cuts LZ78 gap nearly in half** — unchunked FG 24.6% behind BPE → chunked FG only 16.6% behind
4. **Prefix smoothing hurts all tokenizers** — negative result, more weight = worse
5. **Embedding strategies converge** — flat/struct/tuple within 0.1% for LZ78/FG at full training
6. **FreqGated is the best LZ78 variant** — both chunked (1.0999) and unchunked (1.1756)

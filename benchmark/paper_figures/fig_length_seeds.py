"""Figure 4 candidates from the three-seed real-world runs (mean over seeds 42/43/44, error bars = std).

Input: benchmark/paper_figures/data/realworld_sampling3_summary.json (built on the pod by
benchmark/score_seeds_realworld.py). Writes PNG previews to --out (not wired to the manuscript).
Panels: (a) RealKIE header-field VA by length, (b) RealKIE line-item field VA by length,
(c) ExtractBench 131k parse success by length, (d) ExtractBench 131k value accuracy by length."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmark" / "paper_figures" / "data" / "realworld_sampling3_summary.json"
W, H = 2.70, 1.95
GREY, INK, C_BASE, C_STAGE = "#6B7580", "#1B1B1E", "#A7B0CA", "#8EDCE6"
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
      "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300}


def lab(b):
    return b.replace("<=", "$\\leq$").replace("-", "–")


def panel(ax, buckets, ns, series, ylabel, ylim=(0, 100), shade=None):
    x = list(range(len(buckets)))
    for label, means, stds, color in series:
        ax.errorbar(x, means, yerr=stds, color=color, linewidth=1.1, marker="o", ms=3.4, mec=INK, mew=0.45, capsize=1.6, elinewidth=0.6, label=label, zorder=3)
    if shade:
        ax.axvspan(*shade, color="#EEEEEE", zorder=0, linewidth=0)
    ax.set_xticks(x); ax.set_xticklabels([f"{lab(b)}\n(n={n})" for b, n in zip(buckets, ns)], fontsize=5.6)
    ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0); ax.legend(frameon=False, fontsize=5.6, loc="best")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    d = json.loads(DATA.read_text())
    with plt.rc_context(RC):
        rk = d["realkie_74"]; order = ["<=4k", "4-8k", "8-16k", ">16k"]; bk = [b for b in order if b in rk["base_nothink"]["buckets"]]
        ns = [rk["base_nothink"]["buckets"][b]["n"] for b in bk]
        for key, fname, ylabel in [("header_va", "4a_realkie_header_seeds.png", "Header-field value accuracy (%)"), ("item_field_va", "4b_realkie_items_seeds.png", "Line-item field value accuracy (%)")]:
            fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
            panel(ax, bk, ns, [("Qwen3-4B", [rk["base_nothink"]["buckets"][b][key]["mean"] for b in bk], [rk["base_nothink"]["buckets"][b][key]["std"] for b in bk], C_BASE),
                                ("+ STAGE", [rk["sft"]["buckets"][b][key]["mean"] for b in bk], [rk["sft"]["buckets"][b][key]["std"] for b in bk], C_STAGE)], ylabel)
            fig.savefig(a.out / fname, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
        eb = d["extractbench_131k_237"]; order = ["<=4k", "4-8k", "8-16k", "16-32k", "32-64k", ">64k"]; bk = [b for b in order if b in eb["base_nothink_yarn"]["buckets"]]
        ns = [eb["base_nothink_yarn"]["buckets"][b]["n"] for b in bk]
        for key, fname, ylabel in [("PFR", "4c_extractbench_parse_seeds.png", "Parse success (%)"), ("VA", "4d_extractbench_va_seeds.png", "Value accuracy (%)"), ("SCR", "4e_extractbench_scr_seeds.png", "Schema compliance (%)")]:
            fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
            panel(ax, bk, ns, [("Qwen3-4B", [eb["base_nothink_yarn"]["buckets"][b][key]["mean"] for b in bk], [eb["base_nothink_yarn"]["buckets"][b][key]["std"] for b in bk], C_BASE),
                                ("+ STAGE", [eb["sft_yarn"]["buckets"][b][key]["mean"] for b in bk], [eb["sft_yarn"]["buckets"][b][key]["std"] for b in bk], C_STAGE)], ylabel, (0, 104), shade=(3.5, 5.5))
            fig.savefig(a.out / fname, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", sorted(p.name for p in a.out.glob("*.png")))


if __name__ == "__main__":
    main()

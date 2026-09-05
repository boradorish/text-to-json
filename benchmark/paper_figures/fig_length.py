"""Appendix figure: does the STAGE advantage depend on document length?

Two independent panels at final print size (2.70 x 1.95 in) for a 5.5 in column,
composed with subcaption.
  (a) RealKIE-FCC (74 invoices): header-field value accuracy per prompt-length bucket
  (b) ExtractBench (237 digital documents, 131k YaRN context): parse success per prompt-length bucket
Inputs: benchmark/paper_figures/data/length_buckets/{realkie_header,extractbench_long}.json,
built on the pod with benchmark/score_realkie.py and benchmark/length_bucket_analysis.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmark" / "paper_figures" / "data" / "length_buckets"
OUT = ROOT / "overleaf-paper" / "figures"
W, H = 2.70, 1.95
RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 400,
}
GREY, INK = "#6B7580", "#1B1B1E"
C_BASE, C_STAGE = "#A7B0CA", "#8EDCE6"  # author palette (2026-09-05): grey-blue = untrained, aqua = STAGE
# label, run key, colour, marker, fill  (STAGE-Dialog removed from the main-text figure at the author's request)
ARMS_A = [
    ("Qwen3-4B", "qwen3_4b_base_nothink", C_BASE, "o", C_BASE),
    ("+ STAGE", "qwen3_4b_stage_sft", C_STAGE, "o", C_STAGE),
]
ARMS_B = [("Qwen3-4B", "base", C_BASE, "o", C_BASE), ("+ STAGE", "sft", C_STAGE, "o", C_STAGE)]
MS, MEW, LW = 11, 0.45, 1.1  # marker area, marker edge width, line width


def finish(fig, path):
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    page = Bbox.from_bounds(0, 0, W, H)
    assert page.contains(tb.x0, tb.y0) and page.contains(tb.x1, tb.y1), f"clipping: {tb} outside {page}"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def xlabels(buckets, ns):
    return [f"{b.replace('0-4k', '$\\leq$4k').replace('0-2k', '$\\leq$2k')}\n(n={n})" for b, n in zip(buckets, ns)]


def panel_a():
    d = json.loads((DATA / "realkie_header.json").read_text())
    buckets = ["0-4k", "4-8k", "8-16k", ">16k"]
    ns = [d["runs"]["qwen3_4b_base_nothink"][b]["n"] for b in buckets]
    assert sum(ns) == 74, ns
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    x = list(range(len(buckets)))
    for label, key, color, marker, face in ARMS_A:
        y = [d["runs"][key][b]["header_va"] for b in buckets]
        ax.plot(x, y, color=color, linewidth=LW, zorder=2, solid_capstyle="round")
        ax.scatter(x, y, marker=marker, s=MS, facecolors=face, edgecolors=INK, linewidths=MEW, zorder=3, label=label)
    ax.set_xticks(x); ax.set_xticklabels(xlabels(buckets, ns))
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Header-field value accuracy (%)")
    ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    finish(fig, OUT / "fig_len_a.pdf")
    return {label: [d["runs"][key][b]["header_va"] for b in buckets] for label, key, *_ in ARMS_A}, ns


def panel_b():
    """ExtractBench 237 @131k: parse success by length, greedy (solid) and temperature-0.6 sampling (dashed)."""
    dg = json.loads((DATA / "extractbench_long_greedy.json").read_text())
    ds = json.loads((DATA / "extractbench_long.json").read_text())  # sampling run (temperature 0.6)
    buckets = ["<=4k", "4-8k", "8-16k", "16-32k", "32-64k", ">64k"]
    ns = [dg["runs"]["base"][b]["n"] for b in buckets]
    assert sum(ns) == 237, ns
    assert [ds["runs"]["base"][b]["n"] for b in buckets] == ns
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    x = list(range(len(buckets)))
    for label, key, color in [("Qwen3-4B", "base", C_BASE), ("+ STAGE", "sft", C_STAGE)]:
        yg = [100 * dg["runs"][key][b]["PFR"] for b in buckets]; ys = [100 * ds["runs"][key][b]["PFR"] for b in buckets]
        ax.plot(x, yg, color=color, linewidth=LW, zorder=2, solid_capstyle="round", label=f"{label}, greedy")
        ax.scatter(x, yg, marker="o", s=MS, facecolors=color, edgecolors=INK, linewidths=MEW, zorder=3, clip_on=False)
        ax.plot(x, ys, color=color, linewidth=LW, linestyle=(0, (3, 2)), zorder=2, label=f"{label}, sampling")
        ax.scatter(x, ys, marker="o", s=MS, facecolors="white", edgecolors=color, linewidths=0.8, zorder=3, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels([f"{b.replace('<=4k', '$\\leq$4k')}\n(n={n})" for b, n in zip(buckets, ns)], fontsize=5.6)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Parse success (%)")
    ax.set_ylim(0, 104); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axvspan(3.5, 5.5, color="#EEEEEE", zorder=0, linewidth=0)
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderaxespad=0.2, fontsize=5.4, handlelength=2.2)
    finish(fig, OUT / "fig_len_b.pdf")
    return {f"{lab} {dec}": [100 * dd["runs"][key][b]["PFR"] for b in buckets] for lab, key in [("Qwen3-4B", "base"), ("+ STAGE", "sft")] for dec, dd in [("greedy", dg), ("sampling", ds)]}, ns


def main():
    with plt.rc_context(RC):
        a, na = panel_a(); b, nb = panel_b()
    print("self-check (a) RealKIE header VA by bucket", na)
    for k, v in a.items():
        print(f"  {k:26}", " ".join(f"{x:5.1f}" for x in v))
    print("self-check (b) ExtractBench 237 @131k parse success by bucket (greedy / sampling)", nb)
    for k, v in b.items():
        print(f"  {k:26}", " ".join(f"{x:5.1f}" for x in v))
    for name in ("fig_len_a.pdf", "fig_len_b.pdf"):
        f = OUT / name; print(f"  wrote {f} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

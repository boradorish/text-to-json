"""Appendix figures replacing the two real-world metric tables.
  fig_eb131.pdf : ExtractBench 237 @131k YaRN, per prompt-length bucket, parse success / schema compliance / value accuracy,
                  untrained vs STAGE, 3-seed mean with std error bars (data/realworld_sampling3_summary.json).
  fig_rk_budget.pdf : RealKIE-FCC 74, all metrics as grouped bars for base / STAGE under the 3,100 and 16,384-token budgets,
                  3-seed mean with std error bars (data/realkie_budget_summary.json).
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "benchmark/paper_figures/data"; OUT = ROOT / "overleaf-paper/figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6.2,
                     "xtick.labelsize": 6.0, "ytick.labelsize": 6.2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C_BASE, C_STAGE, INK, BAND = "#A7B0CA", "#8EDCE6", "#1B1B1E", "#EEEEEE"
LW, MS, MEW = 1.4, 22, 0.6

def fig_eb131():
    d = json.loads((DATA / "realworld_sampling3_summary.json").read_text())["extractbench_131k_237"]
    B = ["<=4k", "4-8k", "8-16k", "16-32k", "32-64k", ">64k"]; ns = [d["base_nothink_yarn"]["buckets"][b]["n"] for b in B]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9), layout="constrained", sharey=True)
    for ax, (key, title) in zip(axes, [("PFR", "Parse success (%)"), ("SCR", "Schema compliance (%)"), ("VA", "Value accuracy (%)")]):
        x = np.arange(len(B))
        arms = [("base_nothink_yarn", C_BASE, "Qwen3-4B", "-", "o"), ("sft_yarn", C_STAGE, "Qwen3-4B + STAGE", "-", "o"),
                ("q25_base_yarn", C_BASE, "Qwen2.5-3B", (0, (2.2, 1.4)), "s"), ("q25_sft_yarn", C_STAGE, "Qwen2.5-3B + STAGE", (0, (2.2, 1.4)), "s")]
        for arm, color, label, ls, marker in arms:
            if arm not in d: continue
            y = [d[arm]["buckets"][b][key]["mean"] for b in B]; e = [d[arm]["buckets"][b][key]["std"] for b in B]
            ax.errorbar(x, y, yerr=e, color=color, linewidth=LW, linestyle=ls, capsize=1.6, capthick=0.5, elinewidth=0.5, zorder=2)
            ax.scatter(x, y, s=MS if marker == "o" else MS * 0.85, marker=marker, facecolors=color, edgecolors=INK, linewidths=MEW, zorder=3, label=label, clip_on=False)
        ax.axvspan(3.5, 5.5, color=BAND, zorder=0, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels([f"{b.replace('<=4k', '$\\leq$4k')}\n({n})" for b, n in zip(B, ns)], fontsize=5.4)
        ax.set_xlim(-0.4, len(B) - 0.6); ax.set_ylim(0, 104); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title(title, fontsize=7)
        ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0)
    axes[0].set_ylabel("Score (%)"); axes[1].set_xlabel("Prompt length (tokens, bucket size)")
    axes[0].legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.2, fontsize=5.4)
    fig.savefig(OUT / "fig_eb131.pdf"); plt.close(fig)

def fig_rk_budget():
    """RealKIE-FCC 74 by prompt-length bucket, three panels (schema compliance, header-field accuracy, line-item field
    accuracy); 16,384-token budget (Figure 4a protocol); 3-seed mean with std bars."""
    d = json.loads((DATA / "realkie_budget_summary.json").read_text())
    B = ["<=4k", "4-8k", "8-16k", ">16k"]; ns = [d["16384"]["base"]["buckets"][b]["n"] for b in B]
    arms = [("16384", "base", C_BASE, "Qwen3-4B", "-", "o"), ("16384", "stage", C_STAGE, "+ STAGE", "-", "o")]  # 16,384-token budget only
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9), layout="constrained", sharey=True)
    for ax, (key, title) in zip(axes, [("SCR", "Schema compliance (%)"), ("header_va", "Header-field accuracy (%)"), ("item_field_va", "Line-item field accuracy (%)")]):
        x = np.arange(len(B))
        for budget, arm, color, label, ls, marker in arms:
            y = [d[budget][arm]["buckets"][b][key]["mean"] for b in B]; e = [d[budget][arm]["buckets"][b][key]["std"] for b in B]
            ax.errorbar(x, y, yerr=e, color=color, linewidth=LW, linestyle=ls, capsize=1.6, capthick=0.5, elinewidth=0.5, zorder=2)
            ax.scatter(x, y, s=MS if marker == "o" else MS * 0.85, marker=marker, facecolors=color, edgecolors=INK, linewidths=MEW, zorder=3, label=label, clip_on=False)
        ax.set_xticks(x); ax.set_xticklabels([f"{b.replace('<=4k', '$\\leq$4k')}\n({n})" for b, n in zip(B, ns)], fontsize=5.6)
        ax.set_xlim(-0.4, len(B) - 0.6); ax.set_ylim(0, 104); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title(title, fontsize=7)
        ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0)
    axes[0].set_ylabel("Score (%)"); axes[1].set_xlabel("Prompt length (tokens, bucket size)")
    axes[2].legend(loc="upper right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(OUT / "fig_rk_budget.pdf"); plt.close(fig)

def fig_short():
    """Short real-world documents (SWDE, FDA 510(k), PMC): untrained vs STAGE, single seed, grouped bars with dataset brackets."""
    d = json.loads((DATA / "paper_data.json").read_text())["realworld_short"]
    groups = [("SWDE", [("swde_va_norm", "normalised"), ("swde_va_strict", "exact")]),
              ("FDA 510(k)", [("fda_va_norm", "normalised"), ("fda_va_strict", "exact")]),
              ("PMC articles", [("pmc_scalar_va", "scalar\nfields"), ("pmc_keyword_recall", "keyword\nrecall"), ("pmc_ref_record_recall", "reference\nrecall"), ("pmc_author_field_recall", "author-field\nrecall")])]
    keys = [(k, lab) for _, ms in groups for k, lab in ms]; ns = {"SWDE": d["base"]["swde_n"], "FDA 510(k)": d["base"]["fda_n"], "PMC articles": d["base"]["pmc_n"]}
    fig, ax = plt.subplots(figsize=(5.5, 2.1), layout="constrained")
    x = np.arange(len(keys)); w = 0.36
    for i, (arm, color, label) in enumerate([("base", C_BASE, "Qwen3-4B"), ("sft", C_STAGE, "+ STAGE")]):
        vals = [d[arm][k] for k, _ in keys]; xx = x + (i - 0.5) * w
        ax.bar(xx, vals, w, color=color, edgecolor=INK, linewidth=0.5, zorder=3, label=label)
        for xi, v in zip(xx, vals): ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in keys], fontsize=6.0); ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0, 116); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("Value accuracy / recall (%)")
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)
    pos = 0
    for name, ms in groups:
        lo, hi = pos - 0.4, pos + len(ms) - 0.6
        ax.plot([lo, hi], [106, 106], color="#7F8590", linewidth=0.6, clip_on=False); ax.text((lo + hi) / 2, 107.5, f"{name} ({ns[name]:,})", ha="center", va="bottom", fontsize=6.0, color="#7F8590")
        if pos: ax.axvline(pos - 0.5, color="#DDDDDD", linewidth=0.5, zorder=0)
        pos += len(ms)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.0, 0.02))
    fig.savefig(OUT / "fig_short.pdf"); plt.close(fig)

if __name__ == "__main__":
    fig_eb131(); fig_rk_budget(); fig_short(); print("wrote fig_eb131.pdf, fig_rk_budget.pdf, fig_short.pdf")

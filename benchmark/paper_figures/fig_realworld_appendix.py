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
        for arm, color, label in [("base_nothink_yarn", C_BASE, "Qwen3-4B"), ("sft_yarn", C_STAGE, "+ STAGE")]:
            y = [d[arm]["buckets"][b][key]["mean"] for b in B]; e = [d[arm]["buckets"][b][key]["std"] for b in B]
            ax.errorbar(x, y, yerr=e, color=color, linewidth=LW, capsize=1.6, capthick=0.5, elinewidth=0.5, zorder=2)
            ax.scatter(x, y, s=MS, facecolors=color, edgecolors=INK, linewidths=MEW, zorder=3, label=label, clip_on=False)
        ax.axvspan(3.5, 5.5, color=BAND, zorder=0, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels([f"{b.replace('<=4k', '$\\leq$4k')}\n({n})" for b, n in zip(B, ns)], fontsize=5.4)
        ax.set_xlim(-0.4, len(B) - 0.6); ax.set_ylim(0, 104); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title(title, fontsize=7)
        ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0)
    axes[0].set_ylabel("Score (%)"); axes[1].set_xlabel("Prompt length (tokens, bucket size)")
    axes[0].legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(OUT / "fig_eb131.pdf"); plt.close(fig)

def fig_rk_budget():
    d = json.loads((DATA / "realkie_budget_summary.json").read_text())
    metrics = [("PFR", "Parse\nsuccess"), ("SCR", "Schema\ncompliance"), ("header_va", "Header-field\naccuracy"), ("item_field_va", "Line-item\nfield accuracy"), ("item_recall", "Line-item\nrecall")]
    series = [("3100", "base", C_BASE, "", "Qwen3-4B, 3,100-token budget"), ("3100", "stage", C_STAGE, "", "+ STAGE, 3,100-token budget"),
              ("16384", "base", C_BASE, "//", "Qwen3-4B, 16,384-token budget"), ("16384", "stage", C_STAGE, "//", "+ STAGE, 16,384-token budget")]
    plt.rcParams["hatch.linewidth"] = 0.4
    fig, ax = plt.subplots(figsize=(5.5, 2.1), layout="constrained")
    x = np.arange(len(metrics)); w = 0.19
    for i, (budget, arm, color, hatch, label) in enumerate(series):
        vals = [d[budget][arm]["all"][k]["mean"] for k, _ in metrics]; errs = [d[budget][arm]["all"][k]["std"] for k, _ in metrics]
        xx = x + (i - 1.5) * w
        ax.bar(xx, vals, w, color=color, edgecolor=INK, linewidth=0.5, zorder=3, label=label)
        if hatch: ax.bar(xx, vals, w, facecolor="none", edgecolor="#7F8590", linewidth=0, hatch=hatch, zorder=4)
        ax.errorbar(xx, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=5)
        for xi, v, e in zip(xx, vals, errs): ax.text(xi, v + e + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
    ax.set_xticks(x); ax.set_xticklabels([m for _, m in metrics], fontsize=6.2); ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(0, 112); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("Score on RealKIE-FCC (%)")
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerTuple
    handles = []
    for budget, arm, color, hatch, label in series:
        solid = mpatches.Patch(facecolor=color, edgecolor=INK, linewidth=0.5)
        handles.append((solid, mpatches.Patch(facecolor="none", edgecolor="#7F8590", linewidth=0, hatch="////")) if hatch else solid)
    ax.legend(handles, [s[4] for s in series], handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, loc="upper right", frameon=False, ncol=2, fontsize=5.6, handlelength=1.6, handleheight=0.9, columnspacing=1.0, bbox_to_anchor=(1.0, 1.02))
    fig.savefig(OUT / "fig_rk_budget.pdf"); plt.close(fig)

if __name__ == "__main__":
    fig_eb131(); fig_rk_budget(); print("wrote fig_eb131.pdf, fig_rk_budget.pdf")

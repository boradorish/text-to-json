"""Preview: RealKIE-FCC header / line-item value accuracy by prompt-length bucket, base vs STAGE, 3,100 vs 16,384-token
generation budget (3 seeds, error bars = std). Input: data/realkie_budget_summary.json (built on the pod by benchmark/score_realkie_budget.py)."""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "benchmark/paper_figures/data/realkie_budget_summary.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.4,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
C_BASE, C_STAGE, INK = "#A7B0CA", "#8EDCE6", "#1B1B1E"
B = ["<=4k", "4-8k", "8-16k", ">16k"]
def panel(ax, budget, key, ylabel, title):
    x = list(range(len(B)))
    for arm, color, label in [("base", C_BASE, "Qwen3-4B"), ("stage", C_STAGE, "+ STAGE")]:
        v = D[budget][arm]["buckets"]; y = [v[b][key]["mean"] for b in B]; e = [v[b][key]["std"] for b in B]
        ax.errorbar(x, y, yerr=e, color=color, linewidth=1.4, capsize=2, capthick=0.6, elinewidth=0.6, zorder=2)
        ax.scatter(x, y, s=22, facecolors=color, edgecolors=INK, linewidths=0.6, zorder=3, label=label, clip_on=False)
    ns = [D[budget]["base"]["buckets"][b]["n"] for b in B]
    ax.set_xticks(x); ax.set_xticklabels([f"{b.replace('<=4k', '$\\leq$4k')}\n(n={n})" for b, n in zip(B, ns)])
    ax.set_ylim(0, 104); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=7.5)
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0); ax.set_xlabel("Prompt length (tokens)")
a = argparse.ArgumentParser(); a.add_argument("--out", required=True); a = a.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for key, fname, ylabel in [("header_va", "4f_realkie_header_budget.png", "Header-field value accuracy (%)"), ("item_field_va", "4g_realkie_items_budget.png", "Line-item field value accuracy (%)"), ("SCR", "4h_realkie_scr_budget.png", "Schema compliance (%)")]:
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.1), layout="constrained", sharey=True)
    panel(axes[0], "3100", key, ylabel, "Generation budget 3,100 tokens (paper)"); panel(axes[1], "16384", key, "", "Generation budget 16,384 tokens")
    axes[0].legend(loc="lower left", frameon=False, handletextpad=0.3)
    fig.savefig(out / fname, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("wrote", [p.name for p in sorted(out.glob("4[fgh]_*.png"))])

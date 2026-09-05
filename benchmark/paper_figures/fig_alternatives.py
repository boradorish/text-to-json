"""Alternative renderings of the main-text Figure 3 (data construction / constrained decoding) and
Figure 4 (real-world length buckets) for the author to choose from. Writes PNGs (300 dpi, print size
2.70 x 1.95 in per panel unless noted) to --out. Not used by the manuscript."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_tables_to_plots as F  # noqa: E402  (loads paper_data.json and manuscript tables)

TEAL, OCHRE, GREY = F.TEAL, F.OCHRE, F.GREY
LIGHT_TEAL, LIGHT_OCHRE = "#8FBFCF", "#C9B37A"
W, H = 2.70, 1.95
RC = dict(F.RC); RC["savefig.dpi"] = 300
DATA = F.DATA
ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "benchmark" / "paper_figures" / "data" / "length_buckets"


def save(fig, name, out):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, linewidth=0.3, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)


# ------------------------------------------------------------------ data
MATCHED = F.MATCHED  # name -> (EMR, SV, VA)
se = DATA["stage_eval"]
cost = {(r["label"], r["pass"], str(r["batch_size"])): r for r in DATA["inference_cost"]}
COND = [("Qwen3-4B\nfree", "base_nothink_free", "base_nothink_free", OCHRE, False),
        ("Qwen3-4B\nxgrammar", "base_nothink_xgrammar", "base_nothink_xgrammar", OCHRE, True),
        ("+ STAGE\nfree", "stage_sft_free", "sft_free", TEAL, False),
        ("+ STAGE\nxgrammar", "stage_sft_xgrammar", "sft_xgrammar", TEAL, True)]
def cm(key, metric): return se[key]["compat798"][metric]
def lat(ck): return float(cost[(ck, "warm", "1")]["latency_median_seconds"])

rk = json.loads((LB / "realkie_header.json").read_text())["runs"]
RK_B = ["0-4k", "4-8k", "8-16k", ">16k"]; RK_LAB = ["≤4k", "4–8k", "8–16k", ">16k"]
RK = {"Qwen3-4B": "qwen3_4b_base_nothink", "+ STAGE": "qwen3_4b_stage_sft", "+ STAGE + STAGE-Dialog": "qwen3_4b_stage_dialog_v2"}
rk_n = [rk["qwen3_4b_base_nothink"][b]["n"] for b in RK_B]
eb = json.loads((LB / "extractbench_long.json").read_text())["runs"]
EB_B = ["<=4k", "4-8k", "8-16k", "16-32k", "32-64k", ">64k"]; EB_LAB = ["≤4k", "4–8k", "8–16k", "16–32k", "32–64k", ">64k"]
eb_n = [eb["base"][b]["n"] for b in EB_B]


def wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


# ------------------------------------------------------------------ Figure 3(a): data construction
def a_dumbbell(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); names = list(MATCHED); y = range(len(names))
    for yi, n in zip(y, names):
        emr, sv, va = MATCHED[n]; c = TEAL if "STAGE" in n else OCHRE
        ax.plot([emr, va], [yi, yi], color=GREY, lw=0.8, zorder=1)
        ax.scatter([emr], [yi], s=18, facecolors="white", edgecolors=c, lw=0.9, zorder=3); ax.scatter([va], [yi], s=18, color=c, zorder=3)
        ax.annotate(f"{va:.1f}", (va, yi), xytext=(4, 0), textcoords="offset points", fontsize=5.6, va="center")
    ax.set_yticks(list(y)); ax.set_yticklabels(names); ax.set_xlim(0, 100); ax.set_xlabel("Score on STAGE-Eval (%)"); grid(ax, "x"); ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=18, facecolors="white", edgecolors=GREY, label="exact match"); ax.scatter([], [], s=18, color=GREY, label="value accuracy")
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", handletextpad=0.3); save(fig, "3a_1_dumbbell", out)

def a_grouped_bars(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); names = list(MATCHED); x = np.arange(len(names)); w = 0.26
    for i, (metric, label, alpha) in enumerate([(0, "exact match", 0.45), (1, "schema validity", 0.7), (2, "value accuracy", 1.0)]):
        vals = [MATCHED[n][metric] for n in names]; cols = [TEAL if "STAGE" in n else OCHRE for n in names]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=cols, alpha=alpha, zorder=3, label=label)
        for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (ours)", "\n(ours)") for n in names], fontsize=6.0); ax.set_ylim(0, 105); ax.set_ylabel("Score on STAGE-Eval (%)"); grid(ax)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", ncol=1, handlelength=1.0); save(fig, "3a_2_grouped_bars", out)

def a_hbars(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); names = list(MATCHED)[::-1]; y = np.arange(len(names)); h = 0.36
    for i, (metric, label, alpha) in enumerate([(0, "exact match", 0.5), (2, "value accuracy", 1.0)]):
        vals = [MATCHED[n][metric] for n in names]; cols = [TEAL if "STAGE" in n else OCHRE for n in names]
        bars = ax.barh(y + (0.5 - i) * h, vals, h, color=cols, alpha=alpha, zorder=3, label=label)
        for b, v in zip(bars, vals): ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=5.4)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 105); ax.set_xlabel("Score on STAGE-Eval (%)"); grid(ax, "x"); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.0); save(fig, "3a_3_hbars", out)

def a_scatter(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    for n, (emr, sv, va) in MATCHED.items():
        c = TEAL if "STAGE" in n else OCHRE; ax.scatter(emr, va, s=28 if "STAGE" in n else 22, color=c, zorder=3, marker="s" if "STAGE" in n else "o")
        dx, dy, ha = (-4, 4, "right") if "STAGE" in n else (4, -2, "left")
        ax.annotate(n.replace(" (ours)", ""), (emr, va), xytext=(dx, dy), textcoords="offset points", fontsize=5.6, ha=ha)
    ax.plot([0, 100], [0, 100], color="#DDDDDD", lw=0.6, zorder=0); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Exact match (%)"); ax.set_ylabel("Value accuracy (%)"); grid(ax, "both"); save(fig, "3a_4_scatter_emr_va", out)

def a_heatmap(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); names = list(MATCHED); M = np.array([MATCHED[n] for n in names])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.2, color="white" if M[i, j] > 60 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["exact match", "schema valid", "value acc."]); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; save(fig, "3a_5_heatmap", out)

def a_lollipop(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); order = sorted(MATCHED, key=lambda n: MATCHED[n][2]); y = np.arange(len(order))
    for yi, n in zip(y, order):
        emr, sv, va = MATCHED[n]; c = TEAL if "STAGE" in n else OCHRE
        ax.hlines(yi, 0, va, color=c, lw=2.2, zorder=2); ax.scatter([va], [yi], s=30, color=c, zorder=3)
        ax.scatter([emr], [yi], s=14, color="white", edgecolors=c, lw=0.9, zorder=4)
        ax.text(va + 2, yi, f"VA {va:.1f} · EM {emr:.1f}", va="center", fontsize=5.4)
    ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(0, 100); ax.set_xlabel("Value accuracy on STAGE-Eval (%); open dot = exact match"); grid(ax, "x"); ax.tick_params(axis="y", length=0)
    save(fig, "3a_6_lollipop", out)


# ------------------------------------------------------------------ Figure 3(b): training vs constrained decoding
def b_dumbbell(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); y = range(len(COND))
    for yi, (lab, sk, ck, c, _) in zip(y, COND):
        e, v = cm(sk, "EMR"), cm(sk, "VA"); ax.plot([e, v], [yi, yi], color=GREY, lw=0.8, zorder=1)
        ax.scatter([e], [yi], s=18, facecolors="white", edgecolors=c, lw=0.9, zorder=3); ax.scatter([v], [yi], s=18, color=c, zorder=3)
        ax.annotate(f"{v:.1f}  ({lat(ck):.2f} s)", (v, yi), xytext=(4, 0), textcoords="offset points", fontsize=5.6, va="center")
    ax.set_yticks(list(y)); ax.set_yticklabels([c[0].replace("\n", ", ") for c in COND]); ax.set_xlim(0, 100); ax.set_xlabel("Score on STAGE-Eval, 798 schemas (%)"); grid(ax, "x"); ax.tick_params(axis="y", length=0)
    save(fig, "3b_1_dumbbell", out)

def b_bars2x2(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = np.array([0, 1]); w = 0.34
    for i, (model, keys, c) in enumerate([("Qwen3-4B", ("base_nothink_free", "base_nothink_xgrammar"), OCHRE), ("+ STAGE", ("stage_sft_free", "stage_sft_xgrammar"), TEAL)]):
        va = [cm(k, "VA") for k in keys]; em = [cm(k, "EMR") for k in keys]
        bars = ax.bar(x + (i - 0.5) * w, va, w, color=c, alpha=0.35, zorder=2, label=f"{model}: value acc.")
        ax.bar(x + (i - 0.5) * w, em, w, color=c, zorder=3, label=f"{model}: exact match")
        for b, v, e in zip(bars, va, em):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=5.4); ax.text(b.get_x() + b.get_width() / 2, e - 2, f"{e:.1f}", ha="center", va="top", fontsize=5.0, color="white")
    ax.set_xticks(x); ax.set_xticklabels(["free decoding", "xgrammar"]); ax.set_ylim(0, 105); ax.set_ylabel("Score on STAGE-Eval (%)"); grid(ax)
    ax.legend(frameon=False, fontsize=5.0, loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.8); save(fig, "3b_2_bars_2x2", out)

def b_slope(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    for model, keys, c in [("Qwen3-4B", ("base_nothink_free", "base_nothink_xgrammar"), OCHRE), ("+ STAGE", ("stage_sft_free", "stage_sft_xgrammar"), TEAL)]:
        for metric, ls, mk in [("VA", "-", "o"), ("EMR", "--", "s")]:
            v = [cm(k, metric) for k in keys]; ax.plot([0, 1], v, color=c, ls=ls, marker=mk, ms=3.5, lw=1.0, mfc="white" if metric == "EMR" else c, zorder=3)
            ax.text(-0.05, v[0], f"{v[0]:.1f}", ha="right", va="center", fontsize=5.4, color=c); ax.text(1.05, v[1], f"{v[1]:.1f}", ha="left", va="center", fontsize=5.4, color=c)
    ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["free decoding", "xgrammar"]); ax.set_ylim(30, 95); ax.set_ylabel("Score on STAGE-Eval (%)"); grid(ax)
    ax.plot([], [], color=OCHRE, label="Qwen3-4B"); ax.plot([], [], color=TEAL, label="+ STAGE"); ax.plot([], [], color=GREY, ls="-", marker="o", ms=3, label="value acc."); ax.plot([], [], color=GREY, ls="--", marker="s", ms=3, mfc="white", label="exact match")
    ax.legend(frameon=False, fontsize=5.0, loc="center right", handlelength=1.6); save(fig, "3b_3_slope", out)

def b_latency_scatter(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    for lab, sk, ck, c, xg in COND:
        ax.scatter(lat(ck), cm(sk, "VA"), s=18 + cm(sk, "SCR") * 0.5, color=c if not xg else "white", edgecolors=c, lw=1.0, zorder=3)
        dx, dy, ha = (5, 0, "left") if "STAGE" not in lab else (5, 0, "left")
        ax.annotate(lab.replace("\n", ", "), (lat(ck), cm(sk, "VA")), xytext=(dx, dy), textcoords="offset points", fontsize=5.4, ha=ha, va="center")
    ax.set_xlabel("Batch-1 latency, warm cache (s)"); ax.set_ylabel("Value accuracy (%)"); ax.set_xlim(1.6, 2.5); ax.set_ylim(60, 92); grid(ax, "both")
    ax.text(0.02, 0.04, "marker size = schema validity; open = xgrammar", transform=ax.transAxes, fontsize=5.2, color=GREY); save(fig, "3b_4_latency_scatter", out)

def b_heatmap(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); metrics = ["PFR", "SCR", "EMR", "VA"]
    M = np.array([[cm(sk, m) for m in metrics] + [lat(ck)] for _, sk, ck, _, _ in COND])
    Mn = M.copy(); Mn[:, :4] = M[:, :4]; Mn[:, 4] = 100 - (M[:, 4] - M[:, 4].min()) / (M[:, 4].max() - M[:, 4].min() + 1e-9) * 100
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(5): ax.text(j, i, f"{M[i, j]:.1f}" if j < 4 else f"{M[i, j]:.2f} s", ha="center", va="center", fontsize=5.8, color="white" if Mn[i, j] > 60 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(["parse", "schema", "exact", "value", "latency"]); ax.set_yticks(range(4)); ax.set_yticklabels([c[0].replace("\n", ", ") for c in COND])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; save(fig, "3b_5_heatmap", out)

def b_metric_groups(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); metrics = [("EMR", "exact match"), ("SCR", "schema valid"), ("VA", "value acc.")]; x = np.arange(3); w = 0.2
    for i, (lab, sk, ck, c, xg) in enumerate(COND):
        vals = [cm(sk, m) for m, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=c, alpha=1.0 if not xg else 0.45, hatch="" if not xg else "////", edgecolor=c, lw=0.4, zorder=3, label=lab.replace("\n", ", "))
        for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=4.8)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics]); ax.set_ylim(0, 108); ax.set_ylabel("Score on STAGE-Eval (%)"); grid(ax)
    ax.legend(frameon=False, fontsize=4.8, loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.6); save(fig, "3b_6_metric_groups", out)


# ------------------------------------------------------------------ Figure 4(a): RealKIE header VA by length
def c_lines(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = range(4)
    for (label, key), (c, mk, face) in zip(RK.items(), [(OCHRE, "o", "white"), (TEAL, "s", "white"), (TEAL, "s", TEAL)]):
        y = [rk[key][b]["header_va"] for b in RK_B]; ax.plot(x, y, color=c, lw=1.0, zorder=2); ax.scatter(x, y, marker=mk, s=22, facecolors=face, edgecolors=c, lw=0.9, zorder=3, label=label)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(RK_LAB, rk_n)]); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("Header-field value accuracy (%)"); ax.set_ylim(0, 100); grid(ax)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left"); save(fig, "4a_1_lines", out)

def c_bars(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = np.arange(4); w = 0.26
    for i, ((label, key), (c, alpha)) in enumerate(zip(RK.items(), [(OCHRE, 1.0), (TEAL, 1.0), (TEAL, 0.45)])):
        y = [rk[key][b]["header_va"] for b in RK_B]; bars = ax.bar(x + (i - 1) * w, y, w, color=c, alpha=alpha, zorder=3, label=label)
        for b, v in zip(bars, y): ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=5.0)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(RK_LAB, rk_n)]); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("Header-field value accuracy (%)"); ax.set_ylim(0, 100); grid(ax)
    ax.legend(frameon=False, fontsize=5.2, loc="upper right", handlelength=1.0); save(fig, "4a_2_grouped_bars", out)

def c_diff(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = np.arange(4)
    d = [rk["qwen3_4b_stage_sft"][b]["header_va"] - rk["qwen3_4b_base_nothink"][b]["header_va"] for b in RK_B]
    bars = ax.bar(x, d, 0.55, color=[TEAL if v >= 0 else OCHRE for v in d], zorder=3)
    for b, v in zip(bars, d): ax.text(b.get_x() + b.get_width() / 2, v + (2 if v >= 0 else -2), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.6)
    ax.axhline(0, color="black", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(RK_LAB, rk_n)]); ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("STAGE − untrained, header-field acc. (pts)"); ax.set_ylim(-15, 85); grid(ax); save(fig, "4a_3_difference", out)

def c_paired(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = np.arange(4)
    for xi, b in zip(x, RK_B):
        bv, sv = rk["qwen3_4b_base_nothink"][b]["header_va"], rk["qwen3_4b_stage_sft"][b]["header_va"]
        ax.annotate("", (xi, sv), (xi, bv), arrowprops=dict(arrowstyle="-|>", color=TEAL if sv >= bv else OCHRE, lw=1.0, mutation_scale=7), zorder=2)
        ax.scatter([xi], [bv], s=22, facecolors="white", edgecolors=OCHRE, lw=1.0, zorder=3); ax.scatter([xi], [sv], s=22, color=TEAL, marker="s", zorder=3)
        ax.text(xi + 0.12, bv, f"{bv:.0f}", fontsize=5.4, va="center", color=OCHRE); ax.text(xi + 0.12, sv, f"{sv:.0f}", fontsize=5.4, va="center", color=TEAL)
    ax.scatter([], [], s=22, facecolors="white", edgecolors=OCHRE, label="Qwen3-4B"); ax.scatter([], [], s=22, color=TEAL, marker="s", label="+ STAGE")
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(RK_LAB, rk_n)]); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("Header-field value accuracy (%)"); ax.set_ylim(0, 100); ax.set_xlim(-0.5, 3.7); grid(ax)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left"); save(fig, "4a_4_paired_arrows", out)

def c_heatmap(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); M = np.array([[rk[k][b]["header_va"] for b in RK_B] for k in RK.values()])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.4, color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(RK_LAB, rk_n)]); ax.set_yticks(range(3)); ax.set_yticklabels(list(RK)); ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xlabel("Prompt length (tokens)"); save(fig, "4a_5_heatmap", out)

def c_two_metrics(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = range(4)
    for key, c, lab in [("qwen3_4b_base_nothink", OCHRE, "Qwen3-4B"), ("qwen3_4b_stage_sft", TEAL, "+ STAGE")]:
        ax.plot(x, [rk[key][b]["header_va"] for b in RK_B], color=c, lw=1.1, marker="o", ms=3.2, label=f"{lab}: header fields")
        ax.plot(x, [rk[key][b]["item_field_va"] for b in RK_B], color=c, lw=1.0, ls="--", marker="s", ms=3.0, mfc="white", label=f"{lab}: line-item fields")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(RK_LAB, rk_n)]); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("Value accuracy (%)"); ax.set_ylim(0, 100); grid(ax)
    ax.legend(frameon=False, fontsize=5.0, loc="upper right", handlelength=1.6); save(fig, "4a_6_header_and_items", out)


# ------------------------------------------------------------------ Figure 4(b): ExtractBench 131k by length
def d_lines(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = range(6)
    for key, c, mk, lab in [("base", OCHRE, "o", "Qwen3-4B"), ("sft", TEAL, "s", "+ STAGE")]:
        y = [100 * eb[key][b]["PFR"] for b in EB_B]; ax.plot(x, y, color=c, lw=1.0, zorder=2); ax.scatter(x, y, marker=mk, s=22, facecolors="white", edgecolors=c, lw=0.9, zorder=3, label=lab, clip_on=False)
    ax.axvspan(3.5, 5.5, color="#EEEEEE", zorder=0, lw=0); ax.text(3.6, 45, "beyond native\n32k context", fontsize=5.6, color=GREY, va="top")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(EB_LAB, eb_n)], fontsize=5.6); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("Parse success (%)"); ax.set_ylim(0, 104); grid(ax)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left"); save(fig, "4b_1_lines", out)

def d_bars_pfr_va(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = np.arange(6); w = 0.36
    for i, (key, c, lab) in enumerate([("base", OCHRE, "Qwen3-4B"), ("sft", TEAL, "+ STAGE")]):
        pfr = [100 * eb[key][b]["PFR"] for b in EB_B]; va = [100 * eb[key][b]["VA"] for b in EB_B]
        ax.bar(x + (i - 0.5) * w, pfr, w, color=c, alpha=0.3, zorder=2, label=f"{lab}: parse success"); ax.bar(x + (i - 0.5) * w, va, w, color=c, zorder=3, label=f"{lab}: value accuracy")
    ax.axvspan(3.5, 5.5, color="#F3F3F3", zorder=0, lw=0)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(EB_LAB, eb_n)], fontsize=5.4); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("%"); ax.set_ylim(0, 105); grid(ax)
    ax.legend(frameon=False, fontsize=4.8, loc="upper right", ncol=2, handlelength=1.0, columnspacing=0.6); save(fig, "4b_2_bars_pfr_va", out)

def d_diff(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = np.arange(6); w = 0.36
    dp = [100 * (eb["sft"][b]["PFR"] - eb["base"][b]["PFR"]) for b in EB_B]; dv = [100 * (eb["sft"][b]["VA"] - eb["base"][b]["VA"]) for b in EB_B]
    b1 = ax.bar(x - w / 2, dp, w, color=TEAL, alpha=0.45, zorder=3, label="parse success"); b2 = ax.bar(x + w / 2, dv, w, color=TEAL, zorder=3, label="value accuracy")
    for bars, vals in ((b1, dp), (b2, dv)):
        for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + (1.5 if v >= 0 else -1.5), f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.0)
    ax.axhline(0, color="black", lw=0.6); ax.axvspan(3.5, 5.5, color="#F3F3F3", zorder=0, lw=0)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(EB_LAB, eb_n)], fontsize=5.4); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("STAGE − untrained (pts)"); ax.set_ylim(-20, 65); grid(ax)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", handlelength=1.0); save(fig, "4b_3_difference", out)

def d_wilson(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = np.arange(6)
    for i, (key, c, mk, lab) in enumerate([("base", OCHRE, "o", "Qwen3-4B"), ("sft", TEAL, "s", "+ STAGE")]):
        p = [eb[key][b]["PFR"] for b in EB_B]; n = eb_n; lo = [100 * wilson(pp, nn)[0] for pp, nn in zip(p, n)]; hi = [100 * wilson(pp, nn)[1] for pp, nn in zip(p, n)]
        xi = x + (i - 0.5) * 0.12; ax.errorbar(xi, [100 * v for v in p], yerr=[[max(0.0, 100 * v - l) for v, l in zip(p, lo)], [max(0.0, h - 100 * v) for v, h in zip(p, hi)]], fmt=mk, color=c, ms=3.4, mfc="white", lw=0.8, capsize=1.5, zorder=3, label=lab)
    ax.axvspan(3.5, 5.5, color="#EEEEEE", zorder=0, lw=0); ax.text(3.6, 50, "beyond native\n32k context", fontsize=5.6, color=GREY, va="top")
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(EB_LAB, eb_n)], fontsize=5.6); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("Parse success (%), 95% Wilson CI"); ax.set_ylim(0, 104); grid(ax)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left"); save(fig, "4b_4_wilson_ci", out)

def d_heatmap(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); rows = [("Qwen3-4B parse", "base", "PFR"), ("+ STAGE parse", "sft", "PFR"), ("Qwen3-4B value", "base", "VA"), ("+ STAGE value", "sft", "VA")]
    M = np.array([[100 * eb[k][b][m] for b in EB_B] for _, k, m in rows]); ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(4):
        for j in range(6): ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.0, color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(EB_LAB, eb_n)], fontsize=5.4); ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xlabel("Prompt length (tokens)"); save(fig, "4b_5_heatmap", out)

def d_two_metrics(out):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained"); x = range(6)
    for key, c, lab in [("base", OCHRE, "Qwen3-4B"), ("sft", TEAL, "+ STAGE")]:
        ax.plot(x, [100 * eb[key][b]["PFR"] for b in EB_B], color=c, lw=1.1, marker="o", ms=3.2, label=f"{lab}: parse success")
        ax.plot(x, [100 * eb[key][b]["VA"] for b in EB_B], color=c, lw=1.0, ls="--", marker="s", ms=3.0, mfc="white", label=f"{lab}: value accuracy")
    ax.axvspan(3.5, 5.5, color="#EEEEEE", zorder=0, lw=0)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(EB_LAB, eb_n)], fontsize=5.4); ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("%"); ax.set_ylim(0, 104); grid(ax)
    ax.legend(frameon=False, fontsize=4.8, loc="lower left", handlelength=1.6); save(fig, "4b_6_parse_and_value", out)


# ------------------------------------------------------------------ combined / wide alternatives
def combo_fig3_single(out):
    """One wide panel: all seven training/decoding arms on one axis (EMR open, VA filled), grouped."""
    fig, ax = plt.subplots(figsize=(5.5, 2.1), layout="constrained")
    rows = [("JSONSchemaBench (full FT)", MATCHED["JSONSchemaBench"][0], MATCHED["JSONSchemaBench"][2], OCHRE), ("Glaive (full FT)", MATCHED["Glaive"][0], MATCHED["Glaive"][2], OCHRE),
            ("ScrapeGraphAI (full FT)", MATCHED["ScrapeGraphAI"][0], MATCHED["ScrapeGraphAI"][2], OCHRE), ("STAGE (full FT, ours)", MATCHED["STAGE (ours)"][0], MATCHED["STAGE (ours)"][2], TEAL),
            (None, 0, 0, None),
            ("Qwen3-4B, free decoding", cm("base_nothink_free", "EMR"), cm("base_nothink_free", "VA"), OCHRE), ("Qwen3-4B, xgrammar", cm("base_nothink_xgrammar", "EMR"), cm("base_nothink_xgrammar", "VA"), OCHRE),
            ("+ STAGE, free decoding", cm("stage_sft_free", "EMR"), cm("stage_sft_free", "VA"), TEAL), ("+ STAGE, xgrammar", cm("stage_sft_xgrammar", "EMR"), cm("stage_sft_xgrammar", "VA"), TEAL)]
    y = list(range(len(rows)))[::-1]; labels = []
    for yi, (lab, e, v, c) in zip(y, rows):
        labels.append(lab or "")
        if lab is None: continue
        ax.plot([e, v], [yi, yi], color=GREY, lw=0.8, zorder=1); ax.scatter([e], [yi], s=20, facecolors="white", edgecolors=c, lw=0.9, zorder=3); ax.scatter([v], [yi], s=20, color=c, zorder=3)
        ax.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=5.8)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, 100); ax.set_xlabel("Score on STAGE-Eval (%): open = exact match, filled = value accuracy"); grid(ax, "x"); ax.tick_params(axis="y", length=0)
    ax.text(101, y[0] + 0.6, "(a) data construction, one full-FT recipe (851 schemas)", fontsize=5.8, ha="right", color=GREY); ax.text(101, y[5] + 0.6, "(b) training vs. grammar-constrained decoding (798 schemas)", fontsize=5.8, ha="right", color=GREY)
    save(fig, "3_combo_single_axis", out)

def combo_fig4_small_multiples(out):
    """2x2 small multiples: RealKIE header/items and ExtractBench parse/value, base vs STAGE."""
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.4), layout="constrained")
    panels = [(axes[0, 0], "RealKIE-FCC: header-field accuracy", RK_B, RK_LAB, rk_n, lambda k, b: rk[k][b]["header_va"], ("qwen3_4b_base_nothink", "qwen3_4b_stage_sft")),
              (axes[0, 1], "RealKIE-FCC: line-item field accuracy", RK_B, RK_LAB, rk_n, lambda k, b: rk[k][b]["item_field_va"], ("qwen3_4b_base_nothink", "qwen3_4b_stage_sft")),
              (axes[1, 0], "ExtractBench (131k): parse success", EB_B, EB_LAB, eb_n, lambda k, b: 100 * eb[k][b]["PFR"], ("base", "sft")),
              (axes[1, 1], "ExtractBench (131k): value accuracy", EB_B, EB_LAB, eb_n, lambda k, b: 100 * eb[k][b]["VA"], ("base", "sft"))]
    for ax, title, B, LAB, N, f, (kb, ks) in panels:
        x = range(len(B)); ax.plot(x, [f(kb, b) for b in B], color=OCHRE, marker="o", ms=3, mfc="white", lw=1.0, label="Qwen3-4B"); ax.plot(x, [f(ks, b) for b in B], color=TEAL, marker="s", ms=3, mfc="white", lw=1.0, label="+ STAGE")
        ax.set_xticks(list(x)); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(LAB, N)], fontsize=5.2); ax.set_ylim(0, 100); ax.set_title(title, fontsize=6.6, loc="left"); grid(ax)
        if len(B) == 6: ax.axvspan(3.5, 5.5, color="#EEEEEE", zorder=0, lw=0)
    axes[0, 0].legend(frameon=False, fontsize=5.6, loc="lower left"); axes[1, 0].set_xlabel("Prompt length (tokens)"); axes[1, 1].set_xlabel("Prompt length (tokens)")
    save(fig, "4_combo_small_multiples", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        for fn in [a_dumbbell, a_grouped_bars, a_hbars, a_scatter, a_heatmap, a_lollipop, b_dumbbell, b_bars2x2, b_slope, b_latency_scatter, b_heatmap, b_metric_groups,
                   c_lines, c_bars, c_diff, c_paired, c_heatmap, c_two_metrics, d_lines, d_bars_pfr_va, d_diff, d_wilson, d_heatmap, d_two_metrics, combo_fig3_single, combo_fig4_small_multiples]:
            fn(a.out); print("ok", fn.__name__)


if __name__ == "__main__":
    main()

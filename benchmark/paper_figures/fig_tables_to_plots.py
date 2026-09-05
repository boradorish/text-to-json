"""Turn magnitude-grid tables of the manuscript into dot/dumbbell figures.

Figures (each panel authored at 2.70 x 1.95 in for the 5.5 in column):
  fig_main_a  : data construction under one full-FT recipe (EMR, VA per training set)
  fig_main_b  : training vs grammar-constrained decoding (EMR, VA; latency noted)
  fig_dje_a/b : DeepJSONEval Medium / Hard, base -> STAGE per model (detailed, strict)
  fig_eb_a/b  : ExtractBench parse success and value accuracy, base -> STAGE per model

Sources: benchmark/paper_figures/data/paper_data.json (STAGE-Eval, cost, ExtractBench,
rebuilt from raw outputs) and the manuscript tables parsed from overleaf-paper/neurips2026.tex
(Table 1 / DeepJSONEval appendix / matched full-FT numbers supplied by the authors).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "benchmark" / "paper_figures" / "data" / "paper_data.json").read_text())
TEX = (ROOT / "overleaf-paper" / "neurips2026.tex").read_text(encoding="utf-8")
OUT = ROOT / "overleaf-paper" / "figures"
W, H = 2.70, 1.95
TEAL, OCHRE, GREY = "#1E6E8A", "#8A6A1A", "#6B7580"
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
      "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.linewidth": 0.6,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False,
      "pdf.fonttype": 42, "savefig.dpi": 400}


# ------------------------------------------------------------------ parse manuscript tables
def meanstd_row(label_regex: str, block: str) -> list[float]:
    """Return the numeric \\meanstd / \\bestmeanstd values of the first row whose label matches."""
    for row in re.split(r"\\\\", block):
        lines = [l.strip() for l in row.splitlines()
                 if re.sub(r"^(%%\s*)+", "", l.strip()) and not re.match(r"(%%\s*)*\\(mid|cmid|top|bottom)rule", l.strip())]
        if lines and re.search(label_regex, re.sub(r"^(%%\s*)+", "", lines[0])):  # rows commented out in the manuscript keep their numbers
            vals = re.findall(r"\\(?:best|second)?meanstd\{([0-9.]+)\}", row)
            if vals:
                return [float(v) for v in vals]
    raise KeyError(label_regex)


def table_block(label: str) -> str:
    i = TEX.index(f"\\label{{{label}}}"); j = TEX.rfind("\\begin{tabular}", 0, i)
    block = TEX[j:i]
    # join rows that were wrapped after a column separator ("Qwen3-4B &" NEWLINE "\\meanstd...")
    out = []
    for line in block.splitlines():
        if out and out[-1].rstrip().endswith("&") and not out[-1].lstrip().startswith("%"):
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


T1 = table_block("tab:results")            # cols: dje format, detailed, strict | PFR EMR SCR NR VA
TMH = table_block("tab:deepjsoneval_medium_hard")  # medium format/detailed/strict, hard format/detailed/strict
MODELS = [("Qwen3-4B", r"^Qwen3-4B &|^Qwen3-4B$|^Qwen3-4B\s*$", r"Qwen3-4B SFT\s*\+ \\method"),
          ("Qwen2.5-3B", r"^Qwen2\.5-3B", r"Qwen2\.5-3B(-Instruct)? \+ \\method"),
          ("Llama-3.2-1B", r"^Llama-3\.2-1B-Instruct\s*$|^Llama-3\.2-1B-Instruct &", r"Llama-3\.2-1B-Instruct \+ \\method"),
          ("Llama-3.2-3B", r"^Llama-3\.2-3B-Instruct\s*$|^Llama-3\.2-3B-Instruct &", r"Llama-3\.2-3B-Instruct \+ \\method")]


def parse_mh() -> dict:
    out = {}
    for name, base_re, stage_re in MODELS:
        b = meanstd_row(base_re if "Qwen2.5" not in name else r"^Qwen2\.5-3B-Instruct\s*$", TMH)
        s = meanstd_row(stage_re, TMH)
        out[name] = {"base": {"medium": b[0:3], "hard": b[3:6]}, "stage": {"medium": s[0:3], "hard": s[3:6]}}
    return out


MATCHED = {  # numbers supplied by the authors (rebuttal), also in tables/tab_matched_ft.tex
    "JSONSchemaBench": (30.67, 76.38, 53.75), "Glaive": (4.23, 20.21, 11.03),
    "ScrapeGraphAI": (27.97, 66.63, 51.05), "STAGE (ours)": (74.27, 93.54, 90.69)}


METRIC_COLORS = {"exact match": "#8EDCE6", "schema validity": "#D5DCF9", "value accuracy": "#A7B0CA"}  # author-chosen palette
BAR_EDGE = "#1B1B1E"
BAND = "#EFEFEF"  # background band behind the STAGE-trained groups
HATCH_COLOR = "#7F8590"  # quiet grey hatch for grammar-constrained bars


# ------------------------------------------------------------------ helpers
def finish(fig, path):
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer()); page = Bbox.from_bounds(0, 0, W, H)
    assert page.contains(tb.x0, tb.y0) and page.contains(tb.x1, tb.y1), f"clipping {path.name}: {tb}"
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)


def check_texts(fig, ax):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    boxes = [(t, t.get_window_extent(r)) for t in ax.texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i][1].overlaps(boxes[j][1]), f"collision {boxes[i][0].get_text()!r}/{boxes[j][0].get_text()!r}"


def dumbbell(ax, rows, base, stage, color_stage=TEAL, color_base=OCHRE, xlim=(0, 100), xlabel="", note=None):
    """rows: labels top-to-bottom; base/stage: values; segment from base to stage."""
    y = list(range(len(rows)))[::-1]
    for yi, b, s in zip(y, base, stage):
        ax.plot([b, s], [yi, yi], color=GREY, linewidth=0.8, zorder=1)
        ax.scatter([b], [yi], s=18, facecolors="white", edgecolors=color_base, linewidths=0.9, zorder=3)
        ax.scatter([s], [yi], s=18, facecolors=color_stage, edgecolors=color_stage, linewidths=0.9, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(rows); ax.set_xlim(*xlim); ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", linewidth=0.3, color="#DDDDDD", zorder=0); ax.tick_params(axis="y", length=0)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=5.6, va="top", ha="left", color=GREY)


# ------------------------------------------------------------------ main-text figure
def fig_main():
    # (a) data construction, matched full FT: grouped bars (exact match, schema validity, value accuracy) per training set
    #     -- style chosen by the author from fig_alternatives.py (3a_2_grouped_bars)
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    names = list(MATCHED); x = np.arange(len(names)); w = 0.26
    for xi, n in zip(x, names):
        if "STAGE" in n:
            ax.axvspan(xi - 0.5, xi + 0.5, color=BAND, zorder=0, linewidth=0)
    for i, (metric, label) in enumerate([(0, "exact match"), (1, "schema validity"), (2, "value accuracy")]):
        vals = [MATCHED[n][metric] for n in names]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=METRIC_COLORS[label], edgecolor=BAR_EDGE, linewidth=0.5, zorder=3, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (ours)", "\n(ours)") for n in names], fontsize=6.0); ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylim(0, 118); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("Score on STAGE-Eval (%)")
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.4, loc="upper center", ncol=3, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)
    check_texts(fig, ax); finish(fig, OUT / "fig_main_a.pdf")

    # (b) training vs constrained decoding on the 798 schemas: grouped bars (exact match, schema validity, value accuracy)
    #     per decoding condition, same shading scheme as (a); latency in the tick labels
    se = DATA["stage_eval"]; cost = {(r["label"], r["pass"], str(r["batch_size"])): r for r in DATA["inference_cost"]}
    rows = [("Qwen3-4B\nfree", "base_nothink_free", "base_nothink_free"), ("Qwen3-4B\nxgrammar", "base_nothink_xgrammar", "base_nothink_xgrammar"),
            ("+ STAGE\nfree", "stage_sft_free", "sft_free"), ("+ STAGE\nxgrammar", "stage_sft_xgrammar", "sft_xgrammar")]
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    x = np.arange(len(rows)); w = 0.26
    for xi, (lab, _, _) in zip(x, rows):
        if "STAGE" in lab:
            ax.axvspan(xi - 0.5, xi + 0.5, color=BAND, zorder=0, linewidth=0)
    for i, (metric, label) in enumerate([("EMR", "exact match"), ("SCR", "schema validity"), ("VA", "value accuracy")]):
        vals = [se[sk]["compat798"][metric] for _, sk, _ in rows]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=METRIC_COLORS[label], edgecolor=BAR_EDGE, linewidth=0.5, zorder=3, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
    ticks = [f"{lab}\n{float(cost[(ck, 'warm', '1')]['latency_median_seconds']):.2f} s" for lab, _, ck in rows]
    ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=5.6); ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0, 118); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("Score on STAGE-Eval, 798 schemas (%)")
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.4, loc="upper center", ncol=3, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)
    check_texts(fig, ax); finish(fig, OUT / "fig_main_b.pdf")


# ------------------------------------------------------------------ main-text figure, single panel (author request 2026-09-05)
def fig_main_combined():
    """One 5.5 in x 2.3 in panel: untrained Qwen3-4B (free / xgrammar), Qwen3-4B trained on the three comparison
    datasets, and Qwen3-4B trained on STAGE (free / xgrammar). Bars = exact match / schema validity / value accuracy
    (author palette); hatched bars = xgrammar-constrained decoding; grey band = the three alternative datasets."""
    se = DATA["stage_eval"]
    def se_m(key, metric): return se[key]["compat798"][metric]
    groups = [  # label, (EMR, SCR, VA), xgrammar?, section
        ("Qwen3-4B\n(untrained)", (se_m("base_nothink_free", "EMR"), se_m("base_nothink_free", "SCR"), se_m("base_nothink_free", "VA")), False, "none"),
        ("Qwen3-4B\n+ xgrammar", (se_m("base_nothink_xgrammar", "EMR"), se_m("base_nothink_xgrammar", "SCR"), se_m("base_nothink_xgrammar", "VA")), True, "none"),
        ("+ JSONSchema-\nBench data", MATCHED["JSONSchemaBench"], False, "other"),
        ("+ Glaive\ndata", MATCHED["Glaive"], False, "other"),
        ("+ ScrapeGraphAI\ndata", MATCHED["ScrapeGraphAI"], False, "other"),
        ("+ STAGE data\n(ours)", MATCHED["STAGE (ours)"], False, "stage"),
        ("+ STAGE data\n+ xgrammar", (se_m("stage_sft_xgrammar", "EMR"), se_m("stage_sft_xgrammar", "SCR"), se_m("stage_sft_xgrammar", "VA")), True, "stage"),
    ]
    WW, HH = 5.5, 2.5
    fig, ax = plt.subplots(figsize=(WW, HH), layout="constrained")
    plt.rcParams["hatch.linewidth"] = 0.4
    x = np.arange(len(groups)); w = 0.26
    # background band for the three alternative datasets
    ax.axvspan(1.5, 4.5, color=BAND, zorder=0, linewidth=0)
    for i, (label, alpha) in enumerate([("exact match", 1), ("schema validity", 1), ("value accuracy", 1)]):
        vals = [g[1][i] for g in groups]
        for xi, (g, v) in enumerate(zip(groups, vals)):
            ax.bar(xi + (i - 1) * w, v, w, color=METRIC_COLORS[label], edgecolor=BAR_EDGE, linewidth=0.5, zorder=3, label=label if xi == 0 else None)
            if g[2]:  # xgrammar: quiet, sparse grey hatch drawn over the solid bar (hatch colour decoupled from the border)
                ax.bar(xi + (i - 1) * w, v, w, facecolor="none", edgecolor=HATCH_COLOR, linewidth=0, hatch="//", zorder=4)
            ax.text(xi + (i - 1) * w, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.4)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=6.0); ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, 116); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("Score on STAGE-Eval (%)")
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)
    # section brackets
    for (lo, hi, text) in [(-0.4, 1.4, "no training"), (1.6, 4.4, "full fine-tuning on other data"), (4.6, 6.4, "full fine-tuning on STAGE data")]:
        ax.plot([lo, hi], [106, 106], color=GREY, linewidth=0.6, clip_on=False); ax.text((lo + hi) / 2, 108, text, ha="center", va="bottom", fontsize=5.8, color=GREY)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=METRIC_COLORS[k], edgecolor=BAR_EDGE, linewidth=0.5, label=k) for k in METRIC_COLORS]
    from matplotlib.legend_handler import HandlerTuple
    xg_swatch = (mpatches.Patch(facecolor="white", edgecolor=BAR_EDGE, linewidth=0.5),
                 mpatches.Patch(facecolor="none", edgecolor=HATCH_COLOR, linewidth=0, hatch="////"))  # bordered box with the grey hatch
    labels = list(METRIC_COLORS) + ["xgrammar-constrained decoding"]
    ax.legend(handles + [xg_swatch], labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, frameon=False, fontsize=5.6,
              loc="lower center", ncol=4, handlelength=1.6, handleheight=0.9, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    fig.canvas.draw(); tb = fig.get_tightbbox(fig.canvas.get_renderer()); page = Bbox.from_bounds(0, 0, WW, HH)
    assert page.contains(tb.x0, tb.y0) and page.contains(tb.x1, tb.y1), f"clipping fig_main: {tb}"
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "fig_main.pdf"); plt.close(fig)


# ------------------------------------------------------------------ appendix: DeepJSONEval medium / hard
def fig_dje():
    mh = parse_mh()
    for split, fname in (("medium", "fig_dje_a.pdf"), ("hard", "fig_dje_b.pdf")):
        fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
        rows, base, stage = [], [], []
        for name, _, _ in MODELS:
            for mi, metric in ((1, "detailed"), (2, "strict")):
                rows.append(f"{name} {metric}"); base.append(mh[name]["base"][split][mi]); stage.append(mh[name]["stage"][split][mi])
        dumbbell(ax, rows, base, stage, xlim=(0, 100), xlabel=f"DeepJSONEval {split} score (%)")
        for i, (b, s) in enumerate(zip(base, stage)):
            ax.annotate(f"{s:.1f}", (max(b, s), len(rows) - 1 - i), xytext=(4, 0), textcoords="offset points", fontsize=5.2, va="center", ha="left")
        check_texts(fig, ax); finish(fig, OUT / fname)


# ------------------------------------------------------------------ appendix: ExtractBench
def fig_eb():
    eb = DATA["extractbench_194"]
    pairs = [("Qwen3-4B", "qwen3_base_nothink_free", "qwen3_sft_free"), ("Qwen2.5-3B", "qwen25_base_free", "qwen25_sft_free")]
    # (a) schema compliance, all 194 and medium (parse success is ~100% for every Qwen3 condition after re-parsing)
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    rows, base, stage = [], [], []
    for name, bk, sk in pairs:
        for split, key in (("all", "all194"), ("medium", "medium")):
            rows.append(f"{name} {split}"); base.append(eb[bk][key]["SCR"]); stage.append(eb[sk][key]["SCR"])
    dumbbell(ax, rows, base, stage, xlim=(0, 100), xlabel="Schema compliance (%)")
    for i, s in enumerate(stage):
        ax.annotate(f"{s:.1f}", (max(base[i], s), len(rows) - 1 - i), xytext=(4, 0), textcoords="offset points", fontsize=5.4, va="center", ha="left")
    check_texts(fig, ax); finish(fig, OUT / "fig_eb_a.pdf")
    # (b) value accuracy, all / short / medium
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    rows, base, stage = [], [], []
    for name, bk, sk in pairs:
        for split, key in (("all", "all194"), ("short", "short"), ("medium", "medium")):
            rows.append(f"{name} {split}"); base.append(eb[bk][key]["VA"]); stage.append(eb[sk][key]["VA"])
    dumbbell(ax, rows, base, stage, xlim=(0, 50), xlabel="Value accuracy (%)")
    for i, s in enumerate(stage):
        ax.annotate(f"{s:.1f}", (max(base[i], s), len(rows) - 1 - i), xytext=(4, 0), textcoords="offset points", fontsize=5.4, va="center", ha="left")
    check_texts(fig, ax); finish(fig, OUT / "fig_eb_b.pdf")


def main():
    with plt.rc_context(RC):
        fig_main(); fig_main_combined(); fig_dje(); fig_eb()
    mh = parse_mh()
    print("self-check DeepJSONEval parsed (base -> STAGE, hard strict):", {k: (v["base"]["hard"][2], v["stage"]["hard"][2]) for k, v in mh.items()})
    for f in sorted(OUT.glob("fig_*.pdf")):
        print(f"  {f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()

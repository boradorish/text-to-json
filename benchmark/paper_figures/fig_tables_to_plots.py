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
    # (a) data construction, matched full FT: EMR (open) and VA (filled) per training set
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    names = list(MATCHED); y = list(range(len(names)))
    for yi, n in zip(y, names):
        emr, sv, va = MATCHED[n]; c = TEAL if "STAGE" in n else OCHRE
        ax.plot([emr, va], [yi, yi], color=GREY, linewidth=0.8, zorder=1)
        ax.scatter([emr], [yi], s=18, facecolors="white", edgecolors=c, linewidths=0.9, zorder=3)
        ax.scatter([va], [yi], s=18, facecolors=c, edgecolors=c, linewidths=0.9, zorder=3)
        ax.annotate(f"{va:.1f}", (va, yi), xytext=(4, 0), textcoords="offset points", fontsize=5.6, va="center", ha="left")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 100); ax.set_xlabel("Score on STAGE-Eval (%)")
    ax.grid(True, axis="x", linewidth=0.3, color="#DDDDDD", zorder=0); ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=18, facecolors="white", edgecolors=GREY, label="exact match"); ax.scatter([], [], s=18, facecolors=GREY, edgecolors=GREY, label="value accuracy")
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", handletextpad=0.3)
    check_texts(fig, ax); finish(fig, OUT / "fig_main_a.pdf")

    # (b) training vs constrained decoding on the 798 schemas: EMR (open) and VA (filled), latency noted
    se = DATA["stage_eval"]; cost = {(r["label"], r["pass"], r["batch_size"]): r for r in DATA["inference_cost"]}
    rows = [("Qwen3-4B, free", "base_nothink_free", "base_nothink_free"), ("Qwen3-4B, xgrammar", "base_nothink_xgrammar", "base_nothink_xgrammar"),
            ("+ STAGE, free", "stage_sft_free", "sft_free"), ("+ STAGE, xgrammar", "stage_sft_xgrammar", "sft_xgrammar")]
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    y = list(range(len(rows)))
    for yi, (lab, sk, ck) in zip(y, rows):
        m = se[sk]["compat798"]; c = TEAL if "STAGE" in lab else OCHRE
        ax.plot([m["EMR"], m["VA"]], [yi, yi], color=GREY, linewidth=0.8, zorder=1)
        ax.scatter([m["EMR"]], [yi], s=18, facecolors="white", edgecolors=c, linewidths=0.9, zorder=3)
        ax.scatter([m["VA"]], [yi], s=18, facecolors=c, edgecolors=c, linewidths=0.9, zorder=3)
        lat = float(cost[(ck, "warm", "1")]["latency_median_seconds"])
        ax.annotate(f"{m['VA']:.1f}  ({lat:.2f} s)", (m["VA"], yi), xytext=(4, 0), textcoords="offset points", fontsize=5.6, va="center", ha="left")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 100); ax.set_xlabel("Score on STAGE-Eval, 798 schemas (%)")
    ax.grid(True, axis="x", linewidth=0.3, color="#DDDDDD", zorder=0); ax.tick_params(axis="y", length=0)
    check_texts(fig, ax); finish(fig, OUT / "fig_main_b.pdf")


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
        fig_main(); fig_dje(); fig_eb()
    mh = parse_mh()
    print("self-check DeepJSONEval parsed (base -> STAGE, hard strict):", {k: (v["base"]["hard"][2], v["stage"]["hard"][2]) for k, v in mh.items()})
    for f in sorted(OUT.glob("fig_*.pdf")):
        print(f"  {f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()

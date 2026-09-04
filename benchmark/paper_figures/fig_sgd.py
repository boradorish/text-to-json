"""Appendix figure: dialogue-state tracking (SGD) vs in-distribution extraction.

Two independent panels, each authored at final print size (2.70 x 1.95 in) for a
5.5 in single column, composed in LaTeX with subcaption.
  (a) SGD joint goal accuracy vs hallucinated-slot rate (explicit format, 2,000 turns)
  (b) SGD joint goal accuracy vs STAGE-Eval value accuracy (851 examples)
Every printed number is read from benchmark/paper_figures/data/paper_data.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmark" / "paper_figures" / "data" / "paper_data.json"
OUT = ROOT / "overleaf-paper" / "figures"
W, H = 2.70, 1.95  # inches, per panel

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 400,
}
TEAL, OCHRE, GREY = "#1E6E8A", "#8A6A1A", "#6B7580"

# (label, sgd run key, stage-eval key, init, continuation)
MODELS = [
    ("Qwen3-4B (thinking)", "qwen3_4b_base", "base_think_free", "base", False),
    ("Qwen3-4B", "qwen3_4b_base_nothink", "base_nothink_free", "base", False),
    ("+ STAGE", "qwen3_4b_sft", "stage_sft_free", "stage", False),
    ("+ STAGE + STAGE-Dialog", "qwen3_4b_stage_dialog_v2", "stage_sft_dialog_v2", "stage", True),
    ("Qwen3-4B + STAGE-Dialog", "qwen3_4b_base_dialog_v2", "base_dialog_v2", "base", True),
]


def load():
    d = json.loads(DATA.read_text())
    pts = []
    for label, sgd_key, se_key, init, cont in MODELS:
        e = d["sgd_full_2000"][f"{sgd_key}_explicit_full"]["all"]
        se = d["stage_eval"][se_key]["all"]
        assert e["samples"] == 2000, (label, e["samples"])
        assert se["n"] == 851, (label, se["n"])
        pts.append({
            "label": label, "init": init, "cont": cont,
            "jga": e["joint_goal_accuracy"] * 100, "halluc": e["hallucinated_slot_rate"] * 100,
            "va": se["VA"],
        })
    return pts


def style(p):
    marker = "s" if p["init"] == "stage" else "o"
    color = TEAL if p["init"] == "stage" else OCHRE
    face = color if p["cont"] else "white"
    return dict(marker=marker, s=26, facecolors=face, edgecolors=color, linewidths=0.9, zorder=3)


def place_labels(ax, pts, xkey, ykey, offsets):
    """Direct labels with per-point offsets (in points); assert no overlaps."""
    fig = ax.figure
    texts = []
    for p in pts:
        dx, dy, ha = offsets[p["label"]]
        t = ax.annotate(p["label"], (p[xkey], p[ykey]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=5.6, ha=ha, va="center", color="black", zorder=4)
        texts.append(t)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), f"label collision: {texts[i].get_text()!r} / {texts[j].get_text()!r}"
    axbox = ax.get_window_extent(r)
    for t, b in zip(texts, boxes):
        assert axbox.x0 - 2 <= b.x0 and b.x1 <= axbox.x1 + 2, f"label outside axes: {t.get_text()!r}"


def finish(fig, ax, path):
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    page = Bbox.from_bounds(0, 0, W, H)
    assert page.contains(tb.x0, tb.y0) and page.contains(tb.x1, tb.y1), f"clipping: tight bbox {tb} outside page {page}"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)  # no bbox_inches="tight": keep the authored page size
    plt.close(fig)


def panel_a(pts):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    for p in pts:
        ax.scatter(p["halluc"], p["jga"], **style(p))
    ax.set_xlabel("Hallucinated-slot rate (%), lower is better")
    ax.set_ylabel("SGD joint goal accuracy (%)")
    ax.set_xlim(0, 45); ax.set_ylim(10, 50)
    ax.set_xticks([0, 10, 20, 30, 40]); ax.set_yticks([10, 20, 30, 40, 50])
    ax.grid(True, linewidth=0.3, color="#DDDDDD", zorder=0)
    offsets = {
        "Qwen3-4B (thinking)": (5, -5, "left"),
        "Qwen3-4B": (5, -5, "left"),
        "+ STAGE": (-5, 0, "right"),
        "+ STAGE + STAGE-Dialog": (-3, 8, "left"),
        "Qwen3-4B + STAGE-Dialog": (5, -6, "left"),
    }
    place_labels(ax, pts, "halluc", "jga", offsets)
    finish(fig, ax, OUT / "fig_sgd_a.pdf")


def panel_b(pts):
    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    for p in pts:
        ax.scatter(p["jga"], p["va"], **style(p))
    ax.set_xlabel("SGD joint goal accuracy (%)")
    ax.set_ylabel("STAGE-Eval value accuracy (%)")
    ax.set_xlim(10, 50); ax.set_ylim(40, 95)
    ax.set_xticks([10, 20, 30, 40, 50]); ax.set_yticks([40, 50, 60, 70, 80, 90])
    ax.grid(True, linewidth=0.3, color="#DDDDDD", zorder=0)
    offsets = {
        "Qwen3-4B (thinking)": (-5, 0, "right"),
        "Qwen3-4B": (5, 0, "left"),
        "+ STAGE": (5, 0, "left"),
        "+ STAGE + STAGE-Dialog": (-5, 4, "right"),
        "Qwen3-4B + STAGE-Dialog": (-5, -4, "right"),
    }
    place_labels(ax, pts, "jga", "va", offsets)
    finish(fig, ax, OUT / "fig_sgd_b.pdf")


def main():
    pts = load()
    with plt.rc_context(RC):
        panel_a(pts); panel_b(pts)
    print("self-check (label: JGA_explicit, halluc, STAGE-Eval VA):")
    for p in pts:
        print(f"  {p['label']:26} {p['jga']:5.1f} {p['halluc']:5.1f} {p['va']:5.1f}")
    for name in ("fig_sgd_a.pdf", "fig_sgd_b.pdf"):
        f = OUT / name
        print(f"  wrote {f} ({f.stat().st_size} bytes); expected page {W*72:.0f}x{H*72:.0f} pt")


if __name__ == "__main__":
    main()

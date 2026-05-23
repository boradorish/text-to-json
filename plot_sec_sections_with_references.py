#!/usr/bin/env python3
"""
Plot SEC 10-K section token counts with reference dataset token-count boxes.

Example usage:
    python plot_sec_sections_with_references.py \
      --sec_sections_csv outputs/sec_10k_sections_lengths.csv \
      --reference_csvs outputs/realkie_fcc_verified_lengths.csv outputs/ours_text_to_json_lengths.csv outputs/deepjsoneval_lengths.csv \
      --reference_labels "RealKIE FCC" "Ours" "DeepJSONEval" \
      --output outputs/sec_sections_with_references_boxplot.png

    python plot_sec_sections_with_references.py \
      --sec_sections_csv outputs/sec_10k_sections_lengths.csv \
      --reference_csvs outputs/realkie_fcc_verified_lengths.csv outputs/ours_text_to_json_lengths.csv \
      --reference_labels "RealKIE FCC" "Ours" \
      --plot_style summary \
      --output outputs/sec_sections_with_references_summary.png
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "text_to_json_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("sec_sections_references")

SECTION_ORDER = [
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "9C",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
]

SECTION_SHORT_LABELS = {
    "1": "1 Business",
    "1A": "1A Risk",
    "1B": "1B Comments",
    "1C": "1C Cyber",
    "2": "2 Properties",
    "3": "3 Legal",
    "4": "4 Mine",
    "5": "5 Market",
    "6": "6 Data",
    "7": "7 MD&A",
    "7A": "7A Risk",
    "8": "8 Financials",
    "9": "9 Acct.",
    "9A": "9A Controls",
    "9B": "9B Other",
    "9C": "9C Foreign",
    "10": "10 Directors",
    "11": "11 Comp.",
    "12": "12 Ownership",
    "13": "13 Related",
    "14": "14 Fees",
    "15": "15 Exhibits",
}

SECTION_COLOR = "#8EC7E8"
REFERENCE_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SEC 10-K section token counts with reference dataset token counts."
    )
    parser.add_argument("--sec_sections_csv", default="outputs/sec_10k_sections_lengths.csv")
    parser.add_argument("--reference_csvs", nargs="+", required=True)
    parser.add_argument("--reference_labels", nargs="+", required=True)
    parser.add_argument("--token_column", default="token_count")
    parser.add_argument("--output", default="outputs/sec_sections_with_references_boxplot.png")
    parser.add_argument(
        "--plot_style",
        choices=("box", "summary"),
        default="box",
        help="box: standard boxplot. summary: median point + IQR band + p90/p95 markers.",
    )
    parser.add_argument("--log_y", action="store_true", default=True)
    parser.add_argument("--no_log_y", action="store_true", help="Disable log-scale y-axis.")
    parser.add_argument("--show_fliers", action="store_true")
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--title", default="SEC 10-K Sections vs. Text-to-JSON Reference Datasets")
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if len(args.reference_csvs) != len(args.reference_labels):
        raise ValueError("--reference_csvs and --reference_labels must have the same length")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max_rows must be positive when provided")
    if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
        raise ValueError("--y_min must be smaller than --y_max")


def read_token_column(path: Path, token_column: str, max_rows: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path, usecols=[token_column], nrows=max_rows)
    except ValueError as exc:
        header = pd.read_csv(path, nrows=0)
        raise ValueError(f"{path} missing {token_column!r}; columns={list(header.columns)}") from exc
    values = pd.to_numeric(frame[token_column], errors="coerce").dropna()
    values = values[values > 0]
    arr = values.to_numpy(dtype=np.float64)
    LOGGER.info("%s: n=%s median=%.1f p95=%.1f", path, arr.size, np.median(arr), np.percentile(arr, 95))
    return arr


def read_section_data(path: Path, token_column: str, max_rows: int | None) -> tuple[list[str], list[np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path, usecols=["section", token_column], nrows=max_rows)
    except ValueError as exc:
        header = pd.read_csv(path, nrows=0)
        raise ValueError(f"{path} must contain section and {token_column!r}; columns={list(header.columns)}") from exc
    frame[token_column] = pd.to_numeric(frame[token_column], errors="coerce")
    frame = frame.dropna(subset=["section", token_column])
    frame = frame[frame[token_column] > 0]
    frame["section"] = frame["section"].astype(str).str.upper()

    labels: list[str] = []
    arrays: list[np.ndarray] = []
    for section in SECTION_ORDER:
        values = frame.loc[frame["section"] == section, token_column].to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        labels.append(SECTION_SHORT_LABELS.get(section, f"Item {section}"))
        arrays.append(values)
        LOGGER.info("SEC Item %s: n=%s median=%.1f p95=%.1f", section, values.size, np.median(values), np.percentile(values, 95))
    return labels, arrays


def style_boxplot(parts: dict, colors: list[str]) -> None:
    for patch, color in zip(parts["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333333")
    for median in parts["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.7)
    for whisker in parts["whiskers"]:
        whisker.set_color("#555555")
    for cap in parts["caps"]:
        cap.set_color("#555555")


def add_group_labels(
    ax: plt.Axes,
    positions: list[float],
    sec_count: int,
    gap: float,
    *,
    horizontal: bool,
) -> None:
    if sec_count == 0 or sec_count >= len(positions):
        return
    if horizontal:
        separator = len(positions) - sec_count + 0.5
        ax.axhline(separator, color="#777777", linestyle="--", linewidth=1)
        ax.text(
            0.01,
            (np.mean(positions[:sec_count]) - 0.5) / len(positions),
            "SEC 10-K item sections",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color="#333333",
        )
        ax.text(
            0.01,
            (np.mean(positions[sec_count:]) - 0.5) / len(positions),
            "Reference datasets",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color="#333333",
        )
        return

    separator = sec_count + gap / 2
    ax.axvline(separator, color="#777777", linestyle="--", linewidth=1)
    ax.text(
        np.mean(positions[:sec_count]),
        0.98,
        "SEC 10-K item sections",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    ax.text(
        np.mean(positions[sec_count:]),
        0.98,
        "Reference datasets",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )


def plot_box(
    args: argparse.Namespace,
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    positions: list[float],
    sec_count: int,
    gap: float,
) -> plt.Figure:
    fig_width = max(13, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    parts = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        showfliers=args.show_fliers,
        patch_artist=True,
        tick_labels=labels,
    )
    style_boxplot(parts, colors)
    add_group_labels(ax, positions, sec_count, gap, horizontal=False)

    if args.log_y and not args.no_log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Token count (log scale)")
    else:
        ax.set_ylabel("Token count")
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(bottom=args.y_min, top=args.y_max)

    ax.set_title(args.title)
    ax.set_xlabel("SEC item section / dataset")
    ax.grid(True, axis="y", which="major", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    return fig


def percentile_summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    q1, median, q3, p90, p95 = np.percentile(values, [25, 50, 75, 90, 95])
    return float(q1), float(median), float(q3), float(p90), float(p95)


def plot_summary(
    args: argparse.Namespace,
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    sec_count: int,
) -> plt.Figure:
    y_positions = list(np.arange(len(labels), 0, -1, dtype=float))
    fig_height = max(7, 0.34 * len(labels) + 2.2)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    for y, label, values, color in zip(y_positions, labels, data, colors, strict=True):
        q1, median, q3, p90, p95 = percentile_summary(values)
        ax.hlines(y, q1, q3, color=color, linewidth=8, alpha=0.42)
        ax.hlines(y, q3, p95, color=color, linewidth=1.5, alpha=0.9)
        ax.scatter(median, y, color=color, edgecolor="#111111", linewidth=0.6, s=55, zorder=4)
        ax.scatter(p90, y, marker="|", color="#111111", s=180, linewidths=1.8, zorder=5)
        ax.scatter(p95, y, marker="x", color="#111111", s=50, linewidths=1.6, zorder=5)
        LOGGER.info(
            "%s: n=%s q1=%.1f median=%.1f q3=%.1f p90=%.1f p95=%.1f",
            label,
            values.size,
            q1,
            median,
            q3,
            p90,
            p95,
        )

    add_group_labels(ax, y_positions, sec_count, gap=0, horizontal=True)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_title(args.title)
    ax.set_xlabel("Token count (log scale)" if args.log_y and not args.no_log_y else "Token count")
    ax.grid(True, axis="x", which="major", alpha=0.25)
    if args.log_y and not args.no_log_y:
        ax.set_xscale("log")
    if args.y_min is not None or args.y_max is not None:
        ax.set_xlim(left=args.y_min, right=args.y_max)

    ax.hlines([], [], [], color="#555555", linewidth=8, alpha=0.42, label="IQR (Q1-Q3)")
    ax.scatter([], [], color="#555555", edgecolor="#111111", s=55, label="Median")
    ax.scatter([], [], marker="|", color="#111111", s=180, linewidths=1.8, label="p90")
    ax.scatter([], [], marker="x", color="#111111", s=50, linewidths=1.6, label="p95")
    ax.legend(loc="lower right", frameon=True)
    return fig


def plot(args: argparse.Namespace) -> None:
    sec_labels, sec_arrays = read_section_data(Path(args.sec_sections_csv), args.token_column, args.max_rows)
    reference_arrays = [
        read_token_column(Path(path), args.token_column, args.max_rows) for path in args.reference_csvs
    ]
    reference_labels = list(args.reference_labels)

    data = [*sec_arrays, *reference_arrays]
    labels = [*sec_labels, *reference_labels]
    colors = [SECTION_COLOR] * len(sec_arrays) + [
        REFERENCE_COLORS[idx % len(REFERENCE_COLORS)] for idx in range(len(reference_arrays))
    ]
    positions = list(range(1, len(sec_arrays) + 1))
    gap = 1.4
    ref_start = len(sec_arrays) + gap + 1
    positions.extend(ref_start + idx for idx in range(len(reference_arrays)))

    if args.plot_style == "summary":
        fig = plot_summary(args, data, labels, colors, len(sec_arrays))
    else:
        fig = plot_box(args, data, labels, colors, positions, len(sec_arrays), gap)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)
    LOGGER.info("Saved plot: %s", output)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        validate_args(args)
        plot(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare token-length distributions across CSV files.

Example usage:
    python plot_length_comparison.py \
      --inputs outputs/sec_10k_lengths.csv outputs/generated_report_lengths.csv \
      --labels "SEC 10-K" "Generated reports" \
      --output outputs/token_length_comparison.png

    python plot_length_comparison.py \
      --inputs outputs/sec_10k_lengths.csv outputs/ours_text_to_json_lengths.csv \
      --labels "SEC 10-K" "Ours" \
      --plot_type dashboard \
      --output outputs/token_length_comparison_dashboard.png
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


LOGGER = logging.getLogger("length_comparison")
COLOR_CYCLE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token_count distributions from multiple CSV files."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV files with a token_count column.")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels matching --inputs.")
    parser.add_argument("--output", default="outputs/token_length_comparison.png", help="Output PNG path.")
    parser.add_argument("--token_column", default="token_count", help="Token-count column name.")
    parser.add_argument(
        "--plot_type",
        choices=("dashboard", "hist", "ecdf", "survival", "box", "violin", "percentile", "ridgeline"),
        default="dashboard",
        help="Visualization style. dashboard creates one multi-panel figure.",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bin count.")
    parser.add_argument(
        "--density",
        action="store_true",
        help="Normalize histograms to density for easier shape comparison.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Plot normalized sample fractions from 0 to 1 instead of raw counts where applicable.",
    )
    parser.add_argument(
        "--hist_style",
        choices=("filled", "step", "stepfilled"),
        default="filled",
        help="Histogram rendering style. stepfilled uses colored outlines plus translucent fills.",
    )
    parser.add_argument(
        "--no_log_x",
        action="store_true",
        help="Disable automatic log-scale x-axis for skewed distributions.",
    )
    parser.add_argument("--x_min", type=float, default=None, help="Optional minimum token count shown on x-axis.")
    parser.add_argument("--x_max", type=float, default=None, help="Optional maximum token count shown on x-axis.")
    parser.add_argument("--y_min", type=float, default=None, help="Optional minimum shown on y-axis.")
    parser.add_argument("--y_max", type=float, default=None, help="Optional maximum shown on y-axis.")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional maximum rows to read from each CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per pandas chunk when reading CSV files.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must have the same number of values as --inputs")
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max_rows must be positive when provided")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    if args.x_min is not None and args.x_min <= 0:
        raise ValueError("--x_min must be positive when provided")
    if args.x_max is not None and args.x_max <= 0:
        raise ValueError("--x_max must be positive when provided")
    if args.x_min is not None and args.x_max is not None and args.x_min >= args.x_max:
        raise ValueError("--x_min must be smaller than --x_max")
    if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
        raise ValueError("--y_min must be smaller than --y_max")


def read_token_counts(path: Path, token_column: str, max_rows: int | None, chunksize: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    values_list: list[np.ndarray] = []
    rows_seen = 0
    try:
        reader = pd.read_csv(path, usecols=[token_column], chunksize=chunksize)
        for chunk in reader:
            if max_rows is not None:
                remaining = max_rows - rows_seen
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)
            rows_seen += len(chunk)
            values = pd.to_numeric(chunk[token_column], errors="coerce").dropna()
            values = values[values >= 0]
            if not values.empty:
                values_list.append(values.to_numpy(dtype=np.float64))
    except ValueError as exc:
        header = pd.read_csv(path, nrows=0)
        raise ValueError(
            f"{path} does not contain column {token_column!r}. "
            f"Available columns: {list(header.columns)}"
        ) from exc

    arr = np.concatenate(values_list) if values_list else np.asarray([], dtype=np.float64)
    LOGGER.info("Loaded %s token counts from %s", arr.size, path)
    return arr


def should_use_log_x(arrays: list[np.ndarray]) -> bool:
    positive_arrays = [arr[arr > 0] for arr in arrays if np.any(arr > 0)]
    if not positive_arrays:
        return False
    combined = np.concatenate(positive_arrays)
    if combined.size < 2:
        return False
    p50, p95 = np.percentile(combined, [50, 95])
    return bool(p50 > 0 and p95 / p50 >= 5)


def positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0)]


def non_empty_positive(arrays: list[np.ndarray], labels: list[str]) -> list[tuple[str, np.ndarray]]:
    items = [(label, positive(arr)) for label, arr in zip(labels, arrays, strict=True)]
    return [(label, arr) for label, arr in items if arr.size > 0]


def log_bins(arrays: list[np.ndarray], bins: int, x_min: float | None = None, x_max: float | None = None) -> np.ndarray | int:
    combined = np.concatenate(arrays)
    lower = max(1.0, float(combined.min()) if x_min is None else x_min)
    upper = float(combined.max()) if x_max is None else x_max
    if lower >= upper:
        return bins
    return np.logspace(np.log10(lower), np.log10(upper), bins)


def linear_bins(
    arrays: list[np.ndarray],
    bins: int,
    x_min: float | None = None,
    x_max: float | None = None,
) -> np.ndarray | int:
    combined = np.concatenate(arrays)
    lower = float(combined.min()) if x_min is None else x_min
    upper = float(combined.max()) if x_max is None else x_max
    if lower >= upper:
        return bins
    return np.linspace(lower, upper, bins + 1)


def summarize(label: str, values: np.ndarray) -> None:
    if values.size == 0:
        LOGGER.warning("%s has no valid token counts.", label)
        return
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    LOGGER.info(
        "%s | n=%s mean=%.1f p50=%.1f p90=%.1f p95=%.1f p99=%.1f max=%.1f",
        label,
        values.size,
        float(np.mean(values)),
        float(p50),
        float(p90),
        float(p95),
        float(p99),
        float(np.max(values)),
    )


def plot_hist(
    ax: plt.Axes,
    non_empty: list[tuple[str, np.ndarray]],
    bins: int,
    density: bool,
    log_x: bool,
    percent: bool,
    hist_style: str,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    if log_x:
        bin_edges = log_bins([arr for _, arr in non_empty], bins, x_min, x_max)
        for idx, (label, arr) in enumerate(non_empty):
            weights = np.full(arr.shape, 1.0 / arr.size) if percent else None
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            histtype = "stepfilled" if hist_style == "stepfilled" else hist_style
            ax.hist(
                arr,
                bins=bin_edges,
                alpha=0.18 if hist_style == "stepfilled" else 0.85 if hist_style == "step" else 0.38,
                density=False if percent else density,
                weights=weights,
                label=label,
                histtype=histtype,
                color=color,
                edgecolor=color if hist_style in {"step", "stepfilled"} else "white",
                linewidth=1.8 if hist_style == "step" else 0.4,
            )
            if hist_style == "stepfilled":
                ax.hist(
                    arr,
                    bins=bin_edges,
                    alpha=1.0,
                    density=False if percent else density,
                    weights=weights,
                    histtype="step",
                    color=color,
                    linewidth=1.8,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Token count (log scale)")
    else:
        bin_edges = linear_bins([arr for _, arr in non_empty], bins, x_min, x_max)
        for idx, (label, arr) in enumerate(non_empty):
            weights = np.full(arr.shape, 1.0 / arr.size) if percent else None
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            histtype = "stepfilled" if hist_style == "stepfilled" else hist_style
            ax.hist(
                arr,
                bins=bin_edges,
                alpha=0.18 if hist_style == "stepfilled" else 0.85 if hist_style == "step" else 0.38,
                density=False if percent else density,
                weights=weights,
                label=label,
                histtype=histtype,
                color=color,
                edgecolor=color if hist_style in {"step", "stepfilled"} else "white",
                linewidth=1.8 if hist_style == "step" else 0.4,
            )
            if hist_style == "stepfilled":
                ax.hist(
                    arr,
                    bins=bin_edges,
                    alpha=1.0,
                    density=False if percent else density,
                    weights=weights,
                    histtype="step",
                    color=color,
                    linewidth=1.8,
                )
        ax.set_xlabel("Token count")

    if percent:
        ax.set_ylabel("Sample fraction per bin")
        ax.set_ylim(bottom=0 if y_min is None else y_min, top=1 if y_max is None else y_max)
    else:
        ax.set_ylabel("Density" if density else "Number of samples")
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_title("Histogram")
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)


def plot_ecdf(
    ax: plt.Axes,
    non_empty: list[tuple[str, np.ndarray]],
    log_x: bool,
    percent: bool,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    for label, arr in non_empty:
        sorted_values = np.sort(arr)
        y = np.arange(1, sorted_values.size + 1) / sorted_values.size
        ax.step(sorted_values, y, where="post", label=label, linewidth=2)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlabel("Token count (log scale)")
    else:
        ax.set_xlabel("Token count")
    ax.set_ylabel("Cumulative sample fraction")
    ax.set_ylim(0 if y_min is None else y_min, 1.02 if y_max is None else y_max)
    ax.set_title("ECDF")
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)


def plot_survival(
    ax: plt.Axes,
    non_empty: list[tuple[str, np.ndarray]],
    log_x: bool,
    percent: bool,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    for label, arr in non_empty:
        sorted_values = np.sort(arr)
        y = 1 - (np.arange(1, sorted_values.size + 1) / sorted_values.size)
        y = np.maximum(y, 1 / sorted_values.size)
        ax.step(sorted_values, y, where="post", label=label, linewidth=2)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlabel("Token count (log scale)")
    else:
        ax.set_xlabel("Token count")
    ax.set_yscale("log")
    ax.set_ylabel("Sample fraction >= length (log scale)")
    ax.set_title("Tail / Survival Curve")
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)


def plot_percentile(
    ax: plt.Axes,
    non_empty: list[tuple[str, np.ndarray]],
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    percentiles = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99])
    for label, arr in non_empty:
        values = np.percentile(arr, percentiles)
        ax.plot(percentiles, values, marker="o", linewidth=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Token count (log scale)")
    ax.set_xticks(percentiles)
    ax.set_title("Percentile Profile")
    if x_min is not None or x_max is not None:
        ax.set_ylim(bottom=x_min, top=x_max)


def plot_box(ax: plt.Axes, non_empty: list[tuple[str, np.ndarray]], x_min: float | None, x_max: float | None) -> None:
    labels = [label for label, _ in non_empty]
    data = [arr for _, arr in non_empty]
    boxplot_kwargs = {
        "vert": False,
        "showfliers": False,
        "patch_artist": True,
        "boxprops": {"facecolor": "#D9E8F5", "edgecolor": "#4C78A8"},
        "medianprops": {"color": "#D62728", "linewidth": 1.8},
    }
    try:
        ax.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(data, labels=labels, **boxplot_kwargs)
    ax.set_xscale("log")
    ax.set_xlabel("Token count (log scale)")
    ax.set_title("Box Plot (outliers hidden)")
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)


def plot_violin(ax: plt.Axes, non_empty: list[tuple[str, np.ndarray]]) -> None:
    labels = [label for label, _ in non_empty]
    log_data = [np.log10(arr) for _, arr in non_empty]
    parts = ax.violinplot(log_data, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#2F4B68")
        body.set_alpha(0.45)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#D62728")
        parts["cmedians"].set_linewidth(1.8)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("log10(token count)")
    ax.set_title("Violin Plot")


def plot_ridgeline(
    ax: plt.Axes,
    non_empty: list[tuple[str, np.ndarray]],
    bins: int,
    x_min: float | None,
    x_max: float | None,
) -> None:
    arrays = [arr for _, arr in non_empty]
    bin_edges = log_bins(arrays, bins, x_min, x_max)
    if isinstance(bin_edges, int):
        combined = np.concatenate(arrays)
        bin_edges = np.linspace(combined.min(), combined.max() + 1, bin_edges)
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    for idx, (label, arr) in enumerate(non_empty):
        hist, _ = np.histogram(arr, bins=bin_edges, density=True)
        if hist.max() > 0:
            hist = hist / hist.max() * 0.75
        baseline = idx
        ax.fill_between(centers, baseline, baseline + hist, alpha=0.45)
        ax.plot(centers, baseline + hist, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(non_empty)))
    ax.set_yticklabels([label for label, _ in non_empty])
    ax.set_xlabel("Token count (log scale)")
    ax.set_title("Ridgeline Histogram")
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)


def finish_figure(fig: plt.Figure, axes: list[plt.Axes], output_path: Path, title: str) -> None:
    for ax in axes:
        ax.grid(True, which="major", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
        fig.subplots_adjust(top=0.86)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    LOGGER.info("Saved comparison plot: %s", output_path)


def plot_single(
    plot_type: str,
    non_empty: list[tuple[str, np.ndarray]],
    output_path: Path,
    bins: int,
    density: bool,
    log_x: bool,
    percent: bool,
    hist_style: str,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if plot_type == "hist":
        plot_hist(ax, non_empty, bins, density, log_x, percent, hist_style, x_min, x_max, y_min, y_max)
    elif plot_type == "ecdf":
        plot_ecdf(ax, non_empty, log_x, percent, x_min, x_max, y_min, y_max)
    elif plot_type == "survival":
        plot_survival(ax, non_empty, log_x, percent, x_min, x_max, y_min, y_max)
    elif plot_type == "box":
        plot_box(ax, non_empty, x_min, x_max)
    elif plot_type == "violin":
        plot_violin(ax, non_empty)
    elif plot_type == "percentile":
        plot_percentile(ax, non_empty, x_min, x_max, y_min, y_max)
    elif plot_type == "ridgeline":
        plot_ridgeline(ax, non_empty, bins, x_min, x_max)
    else:
        raise ValueError(f"Unsupported single plot type: {plot_type}")
    finish_figure(fig, [ax], output_path, "Token Length Distribution Comparison")


def plot_dashboard(
    non_empty: list[tuple[str, np.ndarray]],
    output_path: Path,
    bins: int,
    density: bool,
    log_x: bool,
    percent: bool,
    hist_style: str,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10))
    axes = list(axes_grid.ravel())
    plot_ecdf(axes[0], non_empty, log_x, percent, x_min, x_max, y_min, y_max)
    plot_survival(axes[1], non_empty, log_x, percent, x_min, x_max, y_min, y_max)
    plot_percentile(axes[2], non_empty, x_min, x_max, y_min, y_max)
    if len(non_empty) <= 6:
        plot_box(axes[3], non_empty, x_min, x_max)
    else:
        plot_ridgeline(axes[3], non_empty, bins, x_min, x_max)
    finish_figure(fig, axes, output_path, "Token Length Distribution Comparison")


def plot_comparison(
    arrays: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    plot_type: str,
    bins: int,
    density: bool,
    log_x: bool,
    percent: bool,
    hist_style: str,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    non_empty = non_empty_positive(arrays, labels)
    if not non_empty:
        raise ValueError("No positive token counts found in any input CSV.")

    if plot_type == "dashboard":
        plot_dashboard(non_empty, output_path, bins, density, log_x, percent, hist_style, x_min, x_max, y_min, y_max)
    else:
        plot_single(plot_type, non_empty, output_path, bins, density, log_x, percent, hist_style, x_min, x_max, y_min, y_max)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        validate_args(args)
        paths = [Path(item) for item in args.inputs]
        labels = args.labels if args.labels is not None else [path.stem for path in paths]
        arrays = [
            read_token_counts(path, args.token_column, args.max_rows, args.chunksize)
            for path in paths
        ]
        for label, values in zip(labels, arrays, strict=True):
            summarize(label, values)

        log_x = False if args.no_log_x else should_use_log_x(arrays)
        plot_comparison(
            arrays=arrays,
            labels=labels,
            output_path=Path(args.output),
            plot_type=args.plot_type,
            bins=args.bins,
            density=args.density,
            log_x=log_x,
            percent=args.percent,
            hist_style=args.hist_style,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

"""
Create presentation-ready visualizations from benchmark/error_analysis.py output.

Example:
    python benchmark/visualize_error_analysis.py \
      --input outputs/benchmark_qwen3_4b_error_analysis.xlsx \
      --output-dir outputs/benchmark_qwen3_4b_visuals
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COLORS = {
    "base": "#8A96A8",
    "ours": "#2166AC",
    "delta": "#1B9E77",
    "bad": "#D95F02",
    "neutral": "#6B7280",
    "grid": "#D6DAE0",
    "light": "#EEF2F6",
    "dark": "#20242A",
}


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def slug(text: str, limit: int = 42) -> str:
    text = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(text)).strip("_")
    return text[:limit] or "item"


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def style_axis(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=COLORS["dark"], pad=14)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=10, color=COLORS["neutral"], va="bottom")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="both", colors=COLORS["dark"], labelsize=9)


def plot_metric_bars(overall: pd.DataFrame, output_dir: Path) -> Path:
    metrics = overall[overall["metric"].isin(["parse_ok_rate", "schema_valid_rate", "exact_match_rate", "value_match_mean"])]
    labels = ["Parse OK", "Schema valid", "Exact match", "Value match"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    base = metrics["base"].to_numpy(dtype=float)
    ours = metrics["ours"].to_numpy(dtype=float)
    ax.bar(x - width / 2, base, width, label="Base", color=COLORS["base"])
    ax.bar(x + width / 2, ours, width, label="Ours", color=COLORS["ours"])
    for i, (b, o) in enumerate(zip(base, ours)):
        ax.text(i + width / 2, o + 0.02, f"+{(o-b)*100:.1f}p", ha="center", fontsize=9, color=COLORS["delta"], fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score / rate")
    style_axis(ax, "Core Metric Lift", "Ours improves both structural validity and exact/value accuracy.")
    ax.legend(frameon=False, loc="upper left")
    out = output_dir / "01_core_metric_lift.png"
    savefig(out)
    return out


def plot_radar(overall: pd.DataFrame, output_dir: Path) -> Path:
    metric_map = {
        "parse_ok_rate": "Parse",
        "schema_valid_rate": "Schema",
        "exact_match_rate": "Exact",
        "value_match_mean": "Value",
        "truncated_suspect_rate": "Not truncated",
    }
    rows = overall[overall["metric"].isin(metric_map)]
    labels = [metric_map[m] for m in rows["metric"]]
    base_values = []
    ours_values = []
    for _, row in rows.iterrows():
        if row["metric"] == "truncated_suspect_rate":
            base_values.append(1 - float(row["base"]))
            ours_values.append(1 - float(row["ours"]))
        else:
            base_values.append(float(row["base"]))
            ours_values.append(float(row["ours"]))

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    base_values = np.concatenate([base_values, [base_values[0]]])
    ours_values = np.concatenate([ours_values, [ours_values[0]]])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, base_values, color=COLORS["base"], linewidth=2, label="Base")
    ax.fill(angles, base_values, color=COLORS["base"], alpha=0.14)
    ax.plot(angles, ours_values, color=COLORS["ours"], linewidth=2.5, label="Ours")
    ax.fill(angles, ours_values, color=COLORS["ours"], alpha=0.18)
    ax.set_xticks(angles[:-1], labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([".25", ".50", ".75", "1.0"], fontsize=8, color=COLORS["neutral"])
    ax.grid(color=COLORS["grid"])
    ax.set_title("Capability Radar", fontsize=16, fontweight="bold", pad=22)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.2, 1.15))
    out = output_dir / "02_radar_capability.png"
    savefig(out)
    return out


def plot_error_distribution(base_dist: pd.DataFrame, best_dist: pd.DataFrame, output_dir: Path) -> Path:
    base = base_dist[["error_type", "count"]].rename(columns={"count": "base"})
    ours = best_dist[["error_type", "count"]].rename(columns={"count": "ours"})
    merged = base.merge(ours, on="error_type", how="outer").fillna(0)
    merged["total"] = merged["base"] + merged["ours"]
    merged = merged.sort_values("total", ascending=True).tail(12)

    y = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - 0.18, merged["base"], height=0.35, color=COLORS["base"], label="Base")
    ax.barh(y + 0.18, merged["ours"], height=0.35, color=COLORS["ours"], label="Ours")
    ax.set_yticks(y, merged["error_type"])
    ax.set_xlabel("Rows")
    style_axis(ax, "Error Type Distribution", "Training moves failures from parse/truncation toward residual value-level mistakes.")
    ax.legend(frameon=False, loc="lower right")
    out = output_dir / "03_error_distribution.png"
    savefig(out)
    return out


def plot_outcomes(outcome: pd.DataFrame, output_dir: Path) -> Path:
    data = outcome.sort_values("count", ascending=True)
    colors = [COLORS["delta"] if "improved" in x or "fixed" in x else COLORS["bad"] if "regressed" in x else COLORS["neutral"] for x in data["outcome"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["outcome"], data["count"], color=colors)
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row["count"] + 4, i, f"{row['ratio']*100:.1f}%", va="center", fontsize=9, color=COLORS["dark"])
    ax.set_xlabel("Rows")
    style_axis(ax, "Outcome Breakdown", "Most changed rows are improvements; regressions are small and inspectable.")
    out = output_dir / "04_outcome_breakdown.png"
    savefig(out)
    return out


def plot_transition_heatmap(transition: pd.DataFrame, output_dir: Path) -> Path:
    top_base = transition.groupby("base_error_type")["count"].sum().sort_values(ascending=False).head(8).index
    top_ours = transition.groupby("ours_error_type")["count"].sum().sort_values(ascending=False).head(8).index
    pivot = (
        transition[transition["base_error_type"].isin(top_base) & transition["ours_error_type"].isin(top_ours)]
        .pivot_table(index="base_error_type", columns="ours_error_type", values="count", aggfunc="sum", fill_value=0)
        .reindex(index=top_base, columns=top_ours, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = LinearSegmentedColormap.from_list("custom_blues", ["#F7FBFF", "#6BAED6", "#08306B"])
    im = ax.imshow(pivot.to_numpy(), cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iloc[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color="white" if val > pivot.to_numpy().max() * 0.45 else COLORS["dark"])
    ax.set_xlabel("Ours error type")
    ax.set_ylabel("Base error type")
    ax.set_title("Error Transition Heatmap", loc="left", fontsize=15, fontweight="bold", pad=14)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    out = output_dir / "05_transition_heatmap.png"
    savefig(out)
    return out


def plot_token_bucket(bucket: pd.DataFrame, output_dir: Path) -> Path | None:
    if bucket.empty or "unknown" in set(bucket["input_token_bucket"]) and len(bucket) == 1:
        return None
    x = np.arange(len(bucket))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, bucket["base_value"], marker="o", color=COLORS["base"], linewidth=2, label="Base value match")
    ax.plot(x, bucket["best_value"], marker="o", color=COLORS["ours"], linewidth=2.5, label="Ours value match")
    ax.bar(x, bucket["value_delta"], color=COLORS["delta"], alpha=0.18, label="Delta")
    ax.set_xticks(x, bucket["input_token_bucket"], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value match")
    style_axis(ax, "Length Robustness", "The largest gains appear on longer inputs.")
    ax.legend(frameon=False, loc="lower right")
    out = output_dir / "06_token_bucket_performance.png"
    savefig(out)
    return out


def plot_language(language: pd.DataFrame, output_dir: Path) -> Path | None:
    if language.empty:
        return None
    x = np.arange(len(language))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, language["base_value"], width, color=COLORS["base"], label="Base")
    ax.bar(x + width / 2, language["best_value"], width, color=COLORS["ours"], label="Ours")
    ax.set_xticks(x, language["language_group"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value match")
    style_axis(ax, "Language Slice", "Both Korean and non-Korean rows improve, with larger delta on non-Korean rows.")
    ax.legend(frameon=False, loc="lower right")
    out = output_dir / "07_language_comparison.png"
    savefig(out)
    return out


def plot_residual_pareto(residual: pd.DataFrame, output_dir: Path) -> Path:
    data = residual.sort_values("count", ascending=False).head(10).copy()
    data["cum_ratio"] = data["count"].cumsum() / data["count"].sum()
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(x, data["count"], color=COLORS["bad"], alpha=0.82)
    ax1.set_xticks(x, data["ours_error_type"], rotation=35, ha="right")
    ax1.set_ylabel("Residual rows")
    style_axis(ax1, "Ours Residual Error Pareto", "Remaining errors are dominated by value wrong and missing value cases.")
    ax2 = ax1.twinx()
    ax2.plot(x, data["cum_ratio"], color=COLORS["dark"], marker="o", linewidth=2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative share")
    ax2.spines[["top", "left"]].set_visible(False)
    out = output_dir / "08_residual_error_pareto.png"
    savefig(out)
    return out


def plot_path_top(paths: pd.DataFrame, output_dir: Path) -> Path | None:
    if paths.empty:
        return None
    data = paths.head(15).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [str(v)[-58:] for v in data["path"]]
    ax.barh(labels, data["count"], color=COLORS["ours"])
    ax.set_xlabel("Mentions across residual path fields")
    style_axis(ax, "Top Residual Paths", "Use this as a checklist for targeted data/prompt fixes.")
    out = output_dir / "09_top_residual_paths.png"
    savefig(out)
    return out


def plot_dashboard(sheets: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    overall = sheets["overall_summary"]
    outcome = sheets["outcome_summary"]
    residual = sheets["best_residual_errors"]
    bucket = sheets["by_token_bucket"]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    metrics = overall[overall["metric"].isin(["parse_ok_rate", "schema_valid_rate", "exact_match_rate", "value_match_mean"])]
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, metrics["base"], 0.36, color=COLORS["base"], label="Base")
    ax.bar(x + 0.18, metrics["ours"], 0.36, color=COLORS["ours"], label="Ours")
    ax.set_xticks(x, ["Parse", "Schema", "Exact", "Value"])
    ax.set_ylim(0, 1.05)
    style_axis(ax, "Core Metrics")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    outcome_data = outcome.sort_values("count", ascending=True)
    colors = [COLORS["delta"] if "improved" in x or "fixed" in x else COLORS["bad"] if "regressed" in x else COLORS["neutral"] for x in outcome_data["outcome"]]
    ax.barh(outcome_data["outcome"], outcome_data["count"], color=colors)
    style_axis(ax, "Outcome Mix")

    ax = fig.add_subplot(gs[1, 0])
    if not bucket.empty and not ("unknown" in set(bucket["input_token_bucket"]) and len(bucket) == 1):
        x = np.arange(len(bucket))
        ax.plot(x, bucket["base_value"], marker="o", color=COLORS["base"], label="Base")
        ax.plot(x, bucket["best_value"], marker="o", color=COLORS["ours"], label="Ours")
        ax.set_xticks(x, bucket["input_token_bucket"])
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
    style_axis(ax, "Value Match by Input Length")

    ax = fig.add_subplot(gs[1, 1])
    residual_data = residual.head(6).sort_values("count", ascending=True)
    ax.barh(residual_data["ours_error_type"], residual_data["count"], color=COLORS["bad"])
    style_axis(ax, "Ours Remaining Errors")

    fig.suptitle("Qwen3-4B Benchmark Error Analysis", x=0.02, ha="left", fontsize=20, fontweight="bold", color=COLORS["dark"])
    out = output_dir / "00_dashboard_summary.png"
    savefig(out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PNG charts from error analysis workbook.")
    parser.add_argument("--input", default="outputs/benchmark_qwen3_4b_error_analysis.xlsx")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir) if args.output_dir else input_path.with_suffix("").with_name(input_path.stem + "_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    sheets = {
        name: read_sheet(input_path, name)
        for name in [
            "overall_summary",
            "outcome_summary",
            "base_error_distribution",
            "best_error_distribution",
            "transition_matrix",
            "by_token_bucket",
            "by_language",
            "best_residual_errors",
            "best_path_summary",
        ]
    }

    outputs = [
        plot_dashboard(sheets, output_dir),
        plot_metric_bars(sheets["overall_summary"], output_dir),
        plot_radar(sheets["overall_summary"], output_dir),
        plot_error_distribution(sheets["base_error_distribution"], sheets["best_error_distribution"], output_dir),
        plot_outcomes(sheets["outcome_summary"], output_dir),
        plot_transition_heatmap(sheets["transition_matrix"], output_dir),
        plot_token_bucket(sheets["by_token_bucket"], output_dir),
        plot_language(sheets["by_language"], output_dir),
        plot_residual_pareto(sheets["best_residual_errors"], output_dir),
        plot_path_top(sheets["best_path_summary"], output_dir),
    ]
    for path in outputs:
        if path is not None:
            print(path)


if __name__ == "__main__":
    main()

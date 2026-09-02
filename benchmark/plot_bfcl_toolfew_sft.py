"""Summarize the low-resource continued-SFT tool-selection experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "bfcl"
RUNS = (
    ("Base", "qwen3_4b_base_stageprompt"),
    ("Base + 256-example SFT", "qwen3_4b_base_toolfew_sft"),
    ("STAGE SFT", "qwen3_4b_sft_stageprompt"),
    ("STAGE + 256-example SFT", "qwen3_4b_stage_toolfew_sft"),
)
CATEGORIES = ("simple_python", "multiple", "parallel")


def read_score(path: Path) -> tuple[float, int]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return float(rows[0]["accuracy"]) * 100, sum("wrong_count" in row.get("error_type", "") for row in rows[1:])


def collect() -> list[dict[str, object]]:
    rows = []
    for label, run in RUNS:
        for category in CATEGORIES:
            accuracy, wrong_count = read_score(OUTPUT_ROOT / run / "score" / "json_decoder" / f"BFCL_v4_{category}_score.json")
            rows.append({"condition": label, "category": category, "ast_accuracy": accuracy, "wrong_count": wrong_count})
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    labels = [label for label, _ in RUNS]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), constrained_layout=True)
    x = range(len(CATEGORIES)); width = 0.19
    for i, label in enumerate(labels):
        values = [next(float(row["ast_accuracy"]) for row in rows if row["condition"] == label and row["category"] == category) for category in CATEGORIES]
        axes[0].bar([v + (i - 1.5) * width for v in x], values, width, label=label, color=colors[i])
    axes[0].set_title("BFCL JSON-native AST accuracy")
    axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(0, 100)
    axes[0].set_xticks(list(x), ["Simple", "Multiple", "Parallel"]); axes[0].grid(axis="y", alpha=0.25)
    multiple = [next(int(row["wrong_count"]) for row in rows if row["condition"] == label and row["category"] == "multiple") for label in labels]
    axes[1].bar(labels, multiple, color=colors)
    axes[1].set_title("Multiple: wrong call-count errors")
    axes[1].set_ylabel("Errors out of 200"); axes[1].set_ylim(0, 200); axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=18)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("256-example continued SFT: base and STAGE", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    rows = collect()
    write_csv(rows, args.output_dir / "toolfew_sft_summary.csv")
    plot(rows, args.output_dir / "toolfew_sft.png")


if __name__ == "__main__":
    main()

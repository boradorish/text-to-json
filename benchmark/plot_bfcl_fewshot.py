"""Summarize and plot the BFCL few-shot-prompting comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "bfcl"
SHOTS = (0, 1, 3, 5)
MODELS = (("base", "qwen3_4b_base"), ("STAGE SFT", "qwen3_4b_sft"))


def read_score(path: Path) -> tuple[float, int]:
    lines = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    accuracy = float(lines[0]["accuracy"])
    wrong_count = sum("wrong_count" in row.get("error_type", "") for row in lines[1:])
    return accuracy, wrong_count


def collect() -> list[dict[str, object]]:
    rows = []
    for label, prefix in MODELS:
        for shots in SHOTS:
            score_dir = OUTPUT_ROOT / f"{prefix}_fewshot{shots}" / "score" / "json_decoder"
            simple, _ = read_score(score_dir / "BFCL_v4_simple_python_score.json")
            multiple, wrong_count = read_score(score_dir / "BFCL_v4_multiple_score.json")
            rows.append(
                {
                    "model": label,
                    "shots": shots,
                    "simple_ast_accuracy": simple * 100,
                    "multiple_ast_accuracy": multiple * 100,
                    "multiple_wrong_count": wrong_count,
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"base": "#4C78A8", "STAGE SFT": "#F58518"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, field, title in zip(
        axes[:2],
        ("simple_ast_accuracy", "multiple_ast_accuracy"),
        ("Simple: AST accuracy", "Multiple: AST accuracy"),
    ):
        for label, _ in MODELS:
            values = [next(float(row[field]) for row in rows if row["model"] == label and row["shots"] == shot) for shot in SHOTS]
            axis.plot(SHOTS, values, marker="o", linewidth=2.2, label=label, color=colors[label])
        axis.set_title(title)
        axis.set_xlabel("Few-shot examples in prompt")
        axis.set_xticks(SHOTS)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Accuracy (%)")
        axis.grid(axis="y", alpha=0.25)
    for label, _ in MODELS:
        values = [next(int(row["multiple_wrong_count"]) for row in rows if row["model"] == label and row["shots"] == shot) for shot in SHOTS]
        axes[2].plot(SHOTS, values, marker="o", linewidth=2.2, label=label, color=colors[label])
    axes[2].set_title("Multiple: extra/wrong call count")
    axes[2].set_xlabel("Few-shot examples in prompt")
    axes[2].set_xticks(SHOTS)
    axes[2].set_ylim(0, 200)
    axes[2].set_ylabel("Errors out of 200")
    axes[2].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("BFCL few-shot prompting: base vs. STAGE SFT", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    rows = collect()
    write_csv(rows, args.output_dir / "fewshot_prompting_summary.csv")
    plot(rows, args.output_dir / "fewshot_prompting.png")


if __name__ == "__main__":
    main()

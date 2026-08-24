"""Compile four-dataset IN/OUT summaries into a Markdown document."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NYCBike1", "NYCBike2", "NYCTaxi", "BJTaxi")
STAGES = (
    ("paper_rescue", "Architecture and objective ablations"),
    ("residual_schedule", "Frozen versus joint residual-schedule ablations"),
    ("phase_training", "Final phase-training ablations"),
)
MAIN = "D_two_stage_frozen_event_weighted_residual"

def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {row["seeds"] for row in rows} != {"3"}:
        raise ValueError(f"incomplete three-seed summary: {path}")
    return {(row["variant"], row["flow"]): row for row in rows}

def value(row, metric):
    return f'{float(row[metric + "_mean"]):.3f} ± {float(row[metric + "_std"]):.3f}'

def make_table(rows):
    text = ["| Variant | IN MAE | OUT MAE | IN EEE | OUT EEE |", "|---|---:|---:|---:|---:|"]
    for variant in sorted({key[0] for key in rows}):
        inn, out = rows[(variant, "inflow")], rows[(variant, "outflow")]
        text.append(f"| {variant} | {value(inn, 'mae')} | {value(out, 'mae')} | {value(inn, 'eee')} | {value(out, 'eee')} |")
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=ROOT / "stssl_protocol_full_paper_results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "stssl_protocol_full_four_dataset_inout_results.md")
    args = parser.parse_args()
    phase = {d: read(args.results_root / d / "phase_training" / "summary_metrics.csv") for d in DATASETS}
    text = [
        "# Four-dataset ST-SSL-protocol results (IN/OUT only)",
        "",
        "All values are mean ± population standard deviation over seeds 1, 2, and 3.",
        "All runs use ST-SSL train+validation scaling, corrected v2 masks, and verified labels.",
        "",
        "## Main model",
        "",
        "| Dataset | IN MAE | OUT MAE | IN EEE | OUT EEE |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        inn, out = phase[dataset][(MAIN, "inflow")], phase[dataset][(MAIN, "outflow")]
        text.append(f"| {dataset} | {value(inn, 'mae')} | {value(out, 'mae')} | {value(inn, 'eee')} | {value(out, 'eee')} |")
    for dataset in DATASETS:
        text.extend(["", f"## {dataset}"])
        for stage, title in STAGES:
            text.extend(["", f"### {title}", ""])
            text.extend(make_table(read(args.results_root / dataset / stage / "summary_metrics.csv")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(args.output)

if __name__ == "__main__":
    main()

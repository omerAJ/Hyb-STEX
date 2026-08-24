"""Generate manuscript tables from complete corrected protocol-v2 records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NYCBike1", "NYCBike2", "NYCTaxi", "BJTaxi")
VARIANTS = ("base_mae", "event_weighted_base", "frozen_residual_mae",
            "joint_event_weighted_residual", "frozen_event_weighted_residual")
SEEDS = (1, 2, 3)
FINAL = "frozen_event_weighted_residual"
NAMES = {
    "base_mae": "MAE base",
    "event_weighted_base": "Event-weighted base",
    "frozen_residual_mae": "Frozen residual, MAE",
    "joint_event_weighted_residual": "Two-stage joint event-weighted residual",
    "frozen_event_weighted_residual": "Hyb-STEX: two-stage frozen event-weighted residual",
}
DETAILS = {
    "base_mae": ("--", "--", "Base, trained from initialization"),
    "event_weighted_base": ("--", "0.25", "Base, trained from initialization"),
    "frozen_residual_mae": ("Always on", "--", "Residual only; base frozen"),
    "joint_event_weighted_residual": ("Always on", "0.25", "Base and residual jointly"),
    "frozen_event_weighted_residual": ("Always on", "0.25", "Residual only; base frozen"),
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "corrected_valid_target_p90_results")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to include. The selected datasets must each have all five variants and three seeds.",
    )
    parser.add_argument("--markdown-output", type=Path,
                        default=ROOT / "analysis" / "corrected_paper_results_and_ablation_tables.md")
    parser.add_argument("--latex-output", type=Path,
                        default=ROOT / "analysis" / "corrected_paper_results_and_ablation_tables.tex")
    return parser.parse_args()


def load_rows(results_dir: Path, datasets: tuple[str, ...]) -> dict:
    records = {}
    for path in results_dir.glob("*/units/*/seed_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete" and record["dataset"] in datasets:
            key = (record["dataset"], record["variant"], int(record["seed"]))
            if key in records:
                raise ValueError(f"Duplicate unit record: {key}")
            records[key] = record
    expected = {(d, v, s) for d in datasets for v in VARIANTS for s in SEEDS}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        raise ValueError("Refusing partial manuscript tables; missing: " + ", ".join("/".join(map(str, key)) for key in missing))
    fingerprints = {record.get("evaluation_protocol_fingerprint") for record in records.values()}
    if len(fingerprints) != 1:
        raise ValueError("Inconsistent corrected evaluation protocol fingerprints.")
    rows = {}
    for (dataset, variant, seed), record in records.items():
        for row in record["metrics"]:
            key = (dataset, variant, seed, row["flow"])
            if key in rows:
                raise ValueError(f"Duplicate metric row: {key}")
            rows[key] = row
    expected_rows = {(d, v, s, f) for d in datasets for v in VARIANTS for s in SEEDS for f in ("inflow", "outflow", "mean")}
    if set(rows) != expected_rows:
        raise ValueError("Unit records are missing required IN, OUT, or mean metric rows.")
    return rows


def stat(rows: dict, dataset: str, variant: str, flow: str, metric: str) -> tuple[float, float]:
    values = [float(rows[(dataset, variant, seed, flow)][metric]) for seed in SEEDS]
    return mean(values), pstdev(values)


def cell(rows: dict, dataset: str, variant: str, flow: str, metric: str) -> str:
    average, sd = stat(rows, dataset, variant, flow, metric)
    return f"{average:.3f} $\\pm$ {sd:.3f}"


def delta(rows: dict, dataset: str, variant: str, metric: str) -> float:
    baseline = stat(rows, dataset, "base_mae", "mean", metric)[0]
    candidate = stat(rows, dataset, variant, "mean", metric)[0]
    return 100 * (candidate - baseline) / baseline


def md_table(headers: tuple[str, ...], data: list[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in data)
    return "\n".join(lines)


def build_markdown(rows: dict, results_dir: Path, datasets: tuple[str, ...]) -> str:
    main = []
    for dataset in datasets:
        main.append((dataset, cell(rows, dataset, "base_mae", "mean", "mae"),
                     cell(rows, dataset, FINAL, "mean", "mae"), f"{delta(rows, dataset, FINAL, 'mae'):+.2f}%",
                     cell(rows, dataset, "base_mae", "mean", "eee"),
                     cell(rows, dataset, FINAL, "mean", "eee"), f"{delta(rows, dataset, FINAL, 'eee'):+.2f}%",
                     cell(rows, dataset, "base_mae", "mean", "normal_mae"),
                     cell(rows, dataset, FINAL, "mean", "normal_mae"), f"{delta(rows, dataset, FINAL, 'normal_mae'):+.2f}%"))
    ablation = []
    for variant in VARIANTS:
        residual, weight, updated = DETAILS[variant]
        values = [cell(rows, d, variant, "mean", "mae") + " / " + cell(rows, d, variant, "mean", "eee") for d in datasets]
        label = NAMES[variant]
        if variant == FINAL:
            label, values = f"**{label}**", [f"**{value}**" for value in values]
        ablation.append((label, residual, weight, updated, *values))
    flows = []
    for dataset in datasets:
        for flow in ("inflow", "outflow"):
            flows.append((dataset, flow.upper(), cell(rows, dataset, "base_mae", flow, "mae"),
                          cell(rows, dataset, FINAL, flow, "mae"), cell(rows, dataset, "base_mae", flow, "eee"),
                          cell(rows, dataset, FINAL, flow, "eee"), cell(rows, dataset, "base_mae", flow, "normal_mae"),
                          cell(rows, dataset, FINAL, flow, "normal_mae")))
    summary = []
    for dataset in datasets:
        for variant in VARIANTS:
            summary.append((dataset, NAMES[variant], cell(rows, dataset, variant, "mean", "mae"),
                            cell(rows, dataset, variant, "mean", "eee"), cell(rows, dataset, variant, "mean", "normal_mae"),
                            cell(rows, dataset, variant, "mean", "event_signed_error")))
    per_seed = []
    for dataset in datasets:
        for variant in VARIANTS:
            for seed in SEEDS:
                row = rows[(dataset, variant, seed, "mean")]
                per_seed.append((dataset, NAMES[variant], str(seed), f"{float(row['mae']):.6f}",
                                 f"{float(row['eee']):.6f}", f"{float(row['normal_mae']):.6f}",
                                 f"{float(row['event_signed_error']):.6f}"))
    return "\n\n".join([
        "# Corrected protocol-v2 paper results and ablation tables",
        f"**Status:** manuscript-ready corrected results from all {len(datasets) * len(VARIANTS) * len(SEEDS)} planned unit records for the selected datasets.\n\n"
        f"**Results root:** {results_dir}  \n"
        "**Protocol:** valid targets Y > 5; events are valid targets above node- and flow-specific training-target p90 thresholds; normal targets are valid non-events.  \n"
        "**Reporting:** arithmetic mean of inflow and outflow, reported as mean $\\pm$ population SD over seeds 1--3; lower is better.",
        "## Main Table 1. Matched base versus selected Hyb-STEX\n\n"
        "Hyb-STEX is a two-stage frozen-base, always-on residual trained with MAE plus event MAE weighted by 0.25.\n\n" +
        md_table(("Dataset", "Base MAE", "Hyb-STEX MAE", "Delta MAE", "Base EEE", "Hyb-STEX EEE", "Delta EEE",
                  "Base normal MAE", "Hyb-STEX normal MAE", "Delta normal MAE"), main),
        "## Main Table 2. Core corrected ablation\n\n"
        "Each dataset cell reports MAE / EEE.\n\n" +
        md_table(("Configuration", "Residual", "Event weight", "Parameters updated during adaptation", *datasets), ablation),
        "## Supplementary Table S1. Flow-specific matched results\n\n" +
        md_table(("Dataset", "Flow", "Base MAE", "Hyb-STEX MAE", "Base EEE", "Hyb-STEX EEE",
                  "Base normal MAE", "Hyb-STEX normal MAE"), flows),
        "## Supplementary Table S2. Complete corrected summary\n\n" +
        md_table(("Dataset", "Configuration", "MAE", "EEE", "Normal MAE", "Event signed error"), summary),
        "## Supplementary Table S3. Complete corrected per-seed mean-flow results\n\n" +
        md_table(("Dataset", "Configuration", "Seed", "MAE", "EEE", "Normal MAE", "Event signed error"), per_seed),
        "## Manuscript-use note\n\n"
        "- Replace every legacy event-aware numerical table with these corrected tables; legacy and corrected EEE values are not interchangeable.\n"
        "- Frame the result as an average-accuracy versus extreme-event-risk tradeoff. Do not claim uniform MAE or normal-region improvement.\n"
        "- The combined per-seed and summary CSV exports are the machine-readable evidence for these tables.",
    ]) + "\n"


def build_latex(rows: dict, datasets: tuple[str, ...]) -> str:
    main = []
    for dataset in datasets:
        main.append(dataset + " & " + " & ".join([
            cell(rows, dataset, "base_mae", "mean", "mae"), cell(rows, dataset, FINAL, "mean", "mae"),
            f"{delta(rows, dataset, FINAL, 'mae'):+.2f}\\%", cell(rows, dataset, "base_mae", "mean", "eee"),
            cell(rows, dataset, FINAL, "mean", "eee"), f"{delta(rows, dataset, FINAL, 'eee'):+.2f}\\%",
            cell(rows, dataset, "base_mae", "mean", "normal_mae"), cell(rows, dataset, FINAL, "mean", "normal_mae"),
            f"{delta(rows, dataset, FINAL, 'normal_mae'):+.2f}\\%"]) + " \\\\")
    ablation = []
    for variant in VARIANTS:
        residual, weight, updated = DETAILS[variant]
        label = NAMES[variant].replace("_", "\\_")
        values = [cell(rows, d, variant, "mean", "mae") + " / " + cell(rows, d, variant, "mean", "eee") for d in datasets]
        if variant == FINAL:
            label, values = "\\textbf{" + label + "}", ["\\textbf{" + value + "}" for value in values]
        ablation.append(label + " & " + residual + " & " + weight + " & " + updated + " & " + " & ".join(values) + " \\\\")
    return "\n".join([
        "% Generated by scripts/generate_corrected_paper_tables.py; do not edit values by hand.",
        "% Requires the booktabs and graphicx packages.",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Matched corrected-protocol performance of the selected Hyb-STEX configuration. Values are mean $\\pm$ population SD over three seeds; lower is better.}",
        "\\label{tab:corrected-main-results}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lccccccccc}", "\\toprule",
        "Dataset & Base MAE & Hyb-STEX MAE & $\\Delta$ MAE & Base EEE & Hyb-STEX EEE & $\\Delta$ EEE & Base normal MAE & Hyb-STEX normal MAE & $\\Delta$ normal MAE \\\\",
        "\\midrule", *main, "\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", "",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Corrected-protocol core ablation. Dataset cells report MAE / EEE (mean $\\pm$ population SD over three seeds); lower is better.}",
        "\\label{tab:corrected-core-ablation}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llll" + "c" * len(datasets) + "}", "\\toprule",
        "Configuration & Residual & Event weight & Parameters updated & " + " & ".join(datasets) + " \\\\",
        "\\midrule", *ablation, "\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", "",
    ])


def main() -> None:
    args = read_args()
    datasets = tuple(args.datasets)
    rows = load_rows(args.results_dir.resolve(), datasets)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(build_markdown(rows, args.results_dir.resolve(), datasets), encoding="utf-8")
    args.latex_output.write_text(build_latex(rows, datasets), encoding="utf-8")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.latex_output}")


if __name__ == "__main__":
    main()

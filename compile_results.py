import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev


BASE_DIR = Path(r"d:\omer\Hyb-STEX\experiments")
EXPERIMENT_NAME = "GPD-Experiment1"
DATASETS = ["BJTaxi", "NYCBike1", "NYCBike2", "NYCTaxi"]
OUTPUT_CSV = BASE_DIR / f"{EXPERIMENT_NAME}_results.csv"
PHASE4_MAE_CSV = BASE_DIR / f"{EXPERIMENT_NAME}_phase4_mae_summary.csv"
PHASE4_EEE_CSV = BASE_DIR / f"{EXPERIMENT_NAME}_phase4_eee_summary.csv"

INFLOW_RE = re.compile(r"INFLOW,\s*MAE:\s*([0-9.]+),\s*EEE:\s*([0-9.]+)")
OUTFLOW_RE = re.compile(r"OUTFLOW,\s*MAE:\s*([0-9.]+),\s*EEE:\s*([0-9.]+)")


def parse_log(log_path):
    lines = log_path.read_text(errors="ignore").splitlines()
    results = []
    in_test = False
    training_finished = False
    inflow = None
    outflow = None

    for line in lines:
        if "== Training finished." in line:
            training_finished = True
            continue

        if "== Test results." in line:
            if not training_finished:
                continue
            in_test = True
            inflow = None
            outflow = None
            continue

        if not in_test:
            continue

        m = INFLOW_RE.search(line)
        if m:
            inflow = (float(m.group(1)), float(m.group(2)))
            continue

        m = OUTFLOW_RE.search(line)
        if m:
            outflow = (float(m.group(1)), float(m.group(2)))

        if inflow and outflow:
            results.append(
                {
                    "inflow_mae": inflow[0],
                    "inflow_eee": inflow[1],
                    "outflow_mae": outflow[0],
                    "outflow_eee": outflow[1],
                }
            )
            in_test = False
            training_finished = False
            inflow = None
            outflow = None

    return results


def pick_latest_log(seed_dir):
    logs = list(seed_dir.rglob("run.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def seed_number(seed_dir_name):
    try:
        return int(seed_dir_name.split("=", 1)[1])
    except Exception:
        return seed_dir_name


def safe_stdev(values):
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def parse_args():
    parser = argparse.ArgumentParser(description="Compile GPD results across datasets and seeds.")
    parser.add_argument(
        "--experiment",
        default=EXPERIMENT_NAME,
        help="Experiment folder name under each dataset (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_name = args.experiment
    output_csv = BASE_DIR / f"{experiment_name}_results.csv"
    phase4_mae_csv = BASE_DIR / f"{experiment_name}_phase4_mae_summary.csv"
    phase4_eee_csv = BASE_DIR / f"{experiment_name}_phase4_eee_summary.csv"

    rows = []
    summary_rows = []
    phase4_mae_rows = []
    phase4_eee_rows = []
    warnings = []

    for dataset in DATASETS:
        exp_dir = BASE_DIR / dataset / experiment_name
        if not exp_dir.exists():
            warnings.append(f"Missing experiment folder: {exp_dir}")
            continue

        seed_dirs = sorted(exp_dir.glob("seed=*"), key=lambda p: seed_number(p.name))
        dataset_phase_values = {}

        for seed_dir in seed_dirs:
            log_path = pick_latest_log(seed_dir)
            if not log_path:
                warnings.append(f"No run.log under {seed_dir}")
                continue

            phase_results = parse_log(log_path)
            if not phase_results:
                warnings.append(f"No test results found in {log_path}")
                continue

            seed_id = seed_number(seed_dir.name)
            for phase_idx, res in enumerate(phase_results, start=1):
                rows.append(
                    {
                        "section": "seed",
                        "dataset": dataset,
                        "seed": seed_id,
                        "phase": phase_idx,
                        **res,
                    }
                )
                dataset_phase_values.setdefault(phase_idx, []).append(res)

        for phase_idx, res_list in sorted(dataset_phase_values.items()):
            metrics = ["inflow_mae", "inflow_eee", "outflow_mae", "outflow_eee"]
            agg = {}
            for metric in metrics:
                values = [r[metric] for r in res_list]
                agg[f"{metric}_mean"] = mean(values)
                agg[f"{metric}_std"] = safe_stdev(values)
                agg[f"{metric}_n"] = len(values)

            summary_rows.append(
                {
                    "section": "summary",
                    "dataset": dataset,
                    "seed": "avg",
                    "phase": phase_idx,
                    "inflow_mae": agg["inflow_mae_mean"],
                    "inflow_eee": agg["inflow_eee_mean"],
                    "outflow_mae": agg["outflow_mae_mean"],
                    "outflow_eee": agg["outflow_eee_mean"],
                    "n": agg["inflow_mae_n"],
                    "std_inflow_mae": agg["inflow_mae_std"],
                    "std_inflow_eee": agg["inflow_eee_std"],
                    "std_outflow_mae": agg["outflow_mae_std"],
                    "std_outflow_eee": agg["outflow_eee_std"],
                }
            )

            if phase_idx == 4:
                phase4_mae_rows.append(
                    {
                        "dataset": dataset,
                        "n": agg["inflow_mae_n"],
                        "inflow_mae_mean": agg["inflow_mae_mean"],
                        "inflow_mae_std": agg["inflow_mae_std"],
                        "outflow_mae_mean": agg["outflow_mae_mean"],
                        "outflow_mae_std": agg["outflow_mae_std"],
                    }
                )
                phase4_eee_rows.append(
                    {
                        "dataset": dataset,
                        "n": agg["inflow_eee_n"],
                        "inflow_eee_mean": agg["inflow_eee_mean"],
                        "inflow_eee_std": agg["inflow_eee_std"],
                        "outflow_eee_mean": agg["outflow_eee_mean"],
                        "outflow_eee_std": agg["outflow_eee_std"],
                    }
                )

    fieldnames = [
        "section",
        "dataset",
        "seed",
        "phase",
        "inflow_mae",
        "inflow_eee",
        "outflow_mae",
        "outflow_eee",
        "n",
        "std_inflow_mae",
        "std_inflow_eee",
        "std_outflow_mae",
        "std_outflow_eee",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_csv
    try:
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = row.copy()
                row.setdefault("n", "")
                row.setdefault("std_inflow_mae", "")
                row.setdefault("std_inflow_eee", "")
                row.setdefault("std_outflow_mae", "")
                row.setdefault("std_outflow_eee", "")
                writer.writerow(row)

            for row in summary_rows:
                writer.writerow(row)
    except PermissionError:
        output_path = output_csv.with_name(f"{output_csv.stem}.new{output_csv.suffix}")
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = row.copy()
                row.setdefault("n", "")
                row.setdefault("std_inflow_mae", "")
                row.setdefault("std_inflow_eee", "")
                row.setdefault("std_outflow_mae", "")
                row.setdefault("std_outflow_eee", "")
                writer.writerow(row)

            for row in summary_rows:
                writer.writerow(row)

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"- {w}")

    print(f"Wrote: {output_path}")

    if phase4_mae_rows:
        phase4_fields = [
            "dataset",
            "n",
            "inflow_mae_mean",
            "inflow_mae_std",
            "outflow_mae_mean",
            "outflow_mae_std",
        ]
        phase4_mae_path = phase4_mae_csv
        try:
            with phase4_mae_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_fields)
                writer.writeheader()
                for row in phase4_mae_rows:
                    writer.writerow(row)
        except PermissionError:
            phase4_mae_path = phase4_mae_csv.with_name(
                f"{phase4_mae_csv.stem}.new{phase4_mae_csv.suffix}"
            )
            with phase4_mae_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_fields)
                writer.writeheader()
                for row in phase4_mae_rows:
                    writer.writerow(row)
        print(f"Wrote: {phase4_mae_path}")
        print("Phase 4 MAE (avg ± std):")
        for row in phase4_mae_rows:
            print(
                f"- {row['dataset']} (n={row['n']}): "
                f"inflow {row['inflow_mae_mean']:.4f} ± {row['inflow_mae_std']:.4f}, "
                f"outflow {row['outflow_mae_mean']:.4f} ± {row['outflow_mae_std']:.4f}"
            )

    if phase4_eee_rows:
        phase4_eee_fields = [
            "dataset",
            "n",
            "inflow_eee_mean",
            "inflow_eee_std",
            "outflow_eee_mean",
            "outflow_eee_std",
        ]
        phase4_eee_path = phase4_eee_csv
        try:
            with phase4_eee_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_eee_fields)
                writer.writeheader()
                for row in phase4_eee_rows:
                    writer.writerow(row)
        except PermissionError:
            phase4_eee_path = phase4_eee_csv.with_name(
                f"{phase4_eee_csv.stem}.new{phase4_eee_csv.suffix}"
            )
            with phase4_eee_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_eee_fields)
                writer.writeheader()
                for row in phase4_eee_rows:
                    writer.writerow(row)
        print(f"Wrote: {phase4_eee_path}")
        print("Phase 4 EEE (avg ± std):")
        for row in phase4_eee_rows:
            print(
                f"- {row['dataset']} (n={row['n']}): "
                f"inflow {row['inflow_eee_mean']:.4f} ± {row['inflow_eee_std']:.4f}, "
                f"outflow {row['outflow_eee_mean']:.4f} ± {row['outflow_eee_std']:.4f}"
            )

    if summary_rows:
        summary_by_dataset = {}
        for row in summary_rows:
            if row["seed"] != "avg":
                continue
            summary_by_dataset.setdefault(row["dataset"], {})[row["phase"]] = row

        print("Phase-wise averages with delta vs previous phase:")
        for dataset in sorted(summary_by_dataset.keys()):
            phases = summary_by_dataset[dataset]
            print(f"- {dataset}")
            for phase_idx in sorted(phases.keys()):
                row = phases[phase_idx]
                prev = phases.get(phase_idx - 1)
                if prev:
                    mae_in_delta = row["inflow_mae"] - prev["inflow_mae"]
                    mae_out_delta = row["outflow_mae"] - prev["outflow_mae"]
                    eee_in_delta = row["inflow_eee"] - prev["inflow_eee"]
                    eee_out_delta = row["outflow_eee"] - prev["outflow_eee"]
                    mae_in_text = f"{row['inflow_mae']:.4f} ({mae_in_delta:+.4f})"
                    mae_out_text = f"{row['outflow_mae']:.4f} ({mae_out_delta:+.4f})"
                    eee_in_text = f"{row['inflow_eee']:.4f} ({eee_in_delta:+.4f})"
                    eee_out_text = f"{row['outflow_eee']:.4f} ({eee_out_delta:+.4f})"
                else:
                    mae_in_text = f"{row['inflow_mae']:.4f} (n/a)"
                    mae_out_text = f"{row['outflow_mae']:.4f} (n/a)"
                    eee_in_text = f"{row['inflow_eee']:.4f} (n/a)"
                    eee_out_text = f"{row['outflow_eee']:.4f} (n/a)"

                print(
                    f"  Phase {phase_idx}: "
                    f"MAE inflow {mae_in_text}, outflow {mae_out_text}; "
                    f"EEE inflow {eee_in_text}, outflow {eee_out_text}"
                )


if __name__ == "__main__":
    main()

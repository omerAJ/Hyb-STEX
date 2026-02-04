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


def main():
    rows = []
    summary_rows = []
    phase4_mae_rows = []
    phase4_eee_rows = []
    warnings = []

    for dataset in DATASETS:
        exp_dir = BASE_DIR / dataset / EXPERIMENT_NAME
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

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_CSV
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
        output_path = OUTPUT_CSV.with_name(f"{OUTPUT_CSV.stem}.new{OUTPUT_CSV.suffix}")
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
        phase4_mae_path = PHASE4_MAE_CSV
        try:
            with phase4_mae_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_fields)
                writer.writeheader()
                for row in phase4_mae_rows:
                    writer.writerow(row)
        except PermissionError:
            phase4_mae_path = PHASE4_MAE_CSV.with_name(
                f"{PHASE4_MAE_CSV.stem}.new{PHASE4_MAE_CSV.suffix}"
            )
            with phase4_mae_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_fields)
                writer.writeheader()
                for row in phase4_mae_rows:
                    writer.writerow(row)
        print(f"Wrote: {phase4_mae_path}")

    if phase4_eee_rows:
        phase4_eee_fields = [
            "dataset",
            "n",
            "inflow_eee_mean",
            "inflow_eee_std",
            "outflow_eee_mean",
            "outflow_eee_std",
        ]
        phase4_eee_path = PHASE4_EEE_CSV
        try:
            with phase4_eee_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_eee_fields)
                writer.writeheader()
                for row in phase4_eee_rows:
                    writer.writerow(row)
        except PermissionError:
            phase4_eee_path = PHASE4_EEE_CSV.with_name(
                f"{PHASE4_EEE_CSV.stem}.new{PHASE4_EEE_CSV.suffix}"
            )
            with phase4_eee_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=phase4_eee_fields)
                writer.writeheader()
                for row in phase4_eee_rows:
                    writer.writerow(row)
        print(f"Wrote: {phase4_eee_path}")


if __name__ == "__main__":
    main()

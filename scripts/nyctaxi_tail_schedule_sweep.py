import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "NYCTaxi.yaml"
DEFAULT_SWEEP_NAME = "nyctaxi_tail_schedule_sweep_v1"
DEFAULT_BASE_CHECKPOINTS = [
    {
        "base_id": "seed1",
        "seed": 1,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=1\20260410-165331\nodewise soft-gpd tail correction\best_model_pred.pth"
        ),
    },
    {
        "base_id": "seed2",
        "seed": 2,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=2\20260410-170812\nodewise soft-gpd tail correction\best_model_pred.pth"
        ),
    },
    {
        "base_id": "seed3",
        "seed": 3,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=3\20260410-171900\nodewise soft-gpd tail correction\best_model_pred.pth"
        ),
    },
]
DEFAULT_SEARCH_SPACE = {
    "tail_threshold_q": [0.90, 0.95, 0.97],
    "lr_init": [3e-4, 1e-3, 3e-3],
    "tail_schedule": ["static", "soft_ramp_fast", "soft_ramp_balanced", "soft_ramp_long"],
}
DEFAULT_SWEEP_CONTROLS = {
    "epochs": 140,
    "early_stop": True,
    "early_stop_patience": 20,
    "tail_lambda_cls": 1.0,
    "tail_lambda_gpd": 1.0,
    "tail_xi_min": -0.5,
    "tail_xi_max": -0.02,
    "tail_eps": 1.0e-6,
}
TRIAL_RESULTS_FIELDS = [
    "timestamp",
    "trial_id",
    "base_id",
    "base_seed",
    "base_checkpoint_path",
    "config_path",
    "experiment_dir",
    "run_log_path",
    "tail_threshold_q",
    "lr_init",
    "tail_schedule",
    "epochs",
    "early_stop_patience",
    "tail_lambda_cls",
    "tail_lambda_gpd",
    "tail_xi_min",
    "tail_xi_max",
    "tail_eps",
    "best_val_metric",
    "best_val_epoch",
    "inflow_test_mae",
    "inflow_test_eee",
    "outflow_test_mae",
    "outflow_test_eee",
    "combined_test_mae",
    "combined_test_eee",
]
FAILURE_FIELDS = [
    "timestamp",
    "trial_id",
    "base_id",
    "base_seed",
    "base_checkpoint_path",
    "config_path",
    "comment",
    "returncode",
    "experiment_dir",
    "run_log_path",
    "message",
]


def sanitize_name(value):
    sanitized = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized)


def format_q_value(q_value):
    return f"q{int(round(q_value * 100)):03d}"


def format_lr_value(lr_value):
    if math.isclose(lr_value, 3e-4):
        return "lr3e-4"
    if math.isclose(lr_value, 1e-3):
        return "lr1e-3"
    if math.isclose(lr_value, 3e-3):
        return "lr3e-3"
    return "lr" + f"{lr_value:.0e}".replace("+0", "").replace("+", "").replace("e-0", "e-")


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(payload, config_file, sort_keys=False)


def read_csv_rows(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_trials(search_space):
    trials = []
    for tail_threshold_q, lr_init, tail_schedule in itertools.product(
        search_space["tail_threshold_q"],
        search_space["lr_init"],
        search_space["tail_schedule"],
    ):
        trial_id = "__".join(
            [
                format_q_value(tail_threshold_q),
                format_lr_value(lr_init),
                f"sched_{tail_schedule}",
            ]
        )
        trials.append(
            {
                "trial_id": trial_id,
                "overrides": {
                    "tail_threshold_q": tail_threshold_q,
                    "lr_init": lr_init,
                    "tail_schedule": tail_schedule,
                },
            }
        )
    return trials


def build_trial_config(base_config, overrides, sweep_controls):
    config = dict(base_config)
    config.update(sweep_controls)
    config.update(overrides)
    return config


def build_manifest(sweep_name, base_checkpoints, search_space, sweep_controls, config_path):
    return {
        "sweep_name": sweep_name,
        "created_at": datetime.now().isoformat(),
        "dataset": "NYCTaxi",
        "config_path": str(config_path),
        "ranking_metric": "mean_combined_test_mae",
        "ranking_tiebreakers": [
            "std_combined_test_mae",
            "mean_combined_val_mae",
            "mean_combined_test_eee",
        ],
        "base_checkpoints": [
            {
                "base_id": base_info["base_id"],
                "seed": base_info["seed"],
                "checkpoint_path": str(base_info["checkpoint_path"]),
            }
            for base_info in base_checkpoints
        ],
        "search_space": search_space,
        "sweep_controls": sweep_controls,
    }


def find_comment_directories(base_seed, comment):
    seed_root = REPO_ROOT / "experiments" / "NYCTaxi" / f"pred__seed={base_seed}"
    if not seed_root.exists():
        return []
    return sorted(
        [path for path in seed_root.glob(f"*/{comment}") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )


def locate_experiment_directory(base_seed, comment, preexisting_dirs):
    candidate_dirs = find_comment_directories(base_seed, comment)
    if not candidate_dirs:
        return None
    preexisting_dirs = {str(path.resolve()) for path in preexisting_dirs}
    new_dirs = [path for path in candidate_dirs if str(path.resolve()) not in preexisting_dirs]
    if new_dirs:
        return new_dirs[-1]
    return candidate_dirs[-1]


def load_results_json(results_path):
    with open(results_path, "r", encoding="utf-8") as results_file:
        return json.load(results_file)


def build_success_row(trial, base_info, config_path, experiment_dir, payload):
    tail_results = payload["tail"]
    test_metrics = tail_results["test_metrics"]
    inflow_test_mae = float(test_metrics["inflow"]["mae"])
    inflow_test_eee = float(test_metrics["inflow"]["eee"])
    outflow_test_mae = float(test_metrics["outflow"]["mae"])
    outflow_test_eee = float(test_metrics["outflow"]["eee"])
    return {
        "timestamp": datetime.now().isoformat(),
        "trial_id": trial["trial_id"],
        "base_id": base_info["base_id"],
        "base_seed": base_info["seed"],
        "base_checkpoint_path": str(base_info["checkpoint_path"]),
        "config_path": str(config_path),
        "experiment_dir": str(experiment_dir),
        "run_log_path": str(experiment_dir / "run.log"),
        "tail_threshold_q": trial["overrides"]["tail_threshold_q"],
        "lr_init": trial["overrides"]["lr_init"],
        "tail_schedule": trial["overrides"]["tail_schedule"],
        "epochs": payload["hyperparameters"]["epochs"],
        "early_stop_patience": payload["hyperparameters"]["early_stop_patience"],
        "tail_lambda_cls": payload["hyperparameters"]["tail_lambda_cls"],
        "tail_lambda_gpd": payload["hyperparameters"]["tail_lambda_gpd"],
        "tail_xi_min": payload["hyperparameters"]["tail_xi_min"],
        "tail_xi_max": payload["hyperparameters"]["tail_xi_max"],
        "tail_eps": payload["hyperparameters"]["tail_eps"],
        "best_val_metric": float(tail_results["best_val_metric"]),
        "best_val_epoch": int(tail_results["best_val_epoch"]),
        "inflow_test_mae": inflow_test_mae,
        "inflow_test_eee": inflow_test_eee,
        "outflow_test_mae": outflow_test_mae,
        "outflow_test_eee": outflow_test_eee,
        "combined_test_mae": (inflow_test_mae + outflow_test_mae) / 2.0,
        "combined_test_eee": (inflow_test_eee + outflow_test_eee) / 2.0,
    }


def build_failure_row(trial, base_info, config_path, comment, returncode, experiment_dir, message):
    experiment_dir_str = str(experiment_dir) if experiment_dir is not None else ""
    run_log_path = str(Path(experiment_dir_str) / "run.log") if experiment_dir is not None else ""
    return {
        "timestamp": datetime.now().isoformat(),
        "trial_id": trial["trial_id"],
        "base_id": base_info["base_id"],
        "base_seed": base_info["seed"],
        "base_checkpoint_path": str(base_info["checkpoint_path"]),
        "config_path": str(config_path),
        "comment": comment,
        "returncode": returncode,
        "experiment_dir": experiment_dir_str,
        "run_log_path": run_log_path,
        "message": message,
    }


def aggregate_success_rows(success_rows, expected_base_count):
    grouped_rows = defaultdict(list)
    for row in success_rows:
        grouped_rows[row["trial_id"]].append(row)

    aggregate_rows = []
    for trial_id, rows in grouped_rows.items():
        first_row = rows[0]
        combined_test_maes = [float(row["combined_test_mae"]) for row in rows]
        combined_test_eees = [float(row["combined_test_eee"]) for row in rows]
        combined_val_maes = [float(row["best_val_metric"]) for row in rows]
        inflow_test_maes = [float(row["inflow_test_mae"]) for row in rows]
        outflow_test_maes = [float(row["outflow_test_mae"]) for row in rows]
        inflow_test_eees = [float(row["inflow_test_eee"]) for row in rows]
        outflow_test_eees = [float(row["outflow_test_eee"]) for row in rows]
        experiment_dirs = [row["experiment_dir"] for row in rows]
        is_complete = len(rows) == expected_base_count
        aggregate_rows.append(
            {
                "trial_id": trial_id,
                "tail_threshold_q": float(first_row["tail_threshold_q"]),
                "lr_init": float(first_row["lr_init"]),
                "tail_schedule": first_row["tail_schedule"],
                "bases_completed": len(rows),
                "is_complete": is_complete,
                "mean_combined_test_mae": mean(combined_test_maes),
                "std_combined_test_mae": pstdev(combined_test_maes) if len(combined_test_maes) > 1 else 0.0,
                "mean_combined_test_eee": mean(combined_test_eees),
                "mean_combined_val_mae": mean(combined_val_maes),
                "mean_inflow_test_mae": mean(inflow_test_maes),
                "mean_outflow_test_mae": mean(outflow_test_maes),
                "mean_inflow_test_eee": mean(inflow_test_eees),
                "mean_outflow_test_eee": mean(outflow_test_eees),
                "experiment_dirs": " | ".join(experiment_dirs),
            }
        )

    return sorted(
        aggregate_rows,
        key=lambda row: (
            0 if row["is_complete"] else 1,
            row["mean_combined_test_mae"],
            row["std_combined_test_mae"],
            row["mean_combined_val_mae"],
            row["mean_combined_test_eee"],
        ),
    )


def write_aggregate_csv(path, aggregate_rows):
    fieldnames = [
        "trial_id",
        "tail_threshold_q",
        "lr_init",
        "tail_schedule",
        "bases_completed",
        "is_complete",
        "mean_combined_test_mae",
        "std_combined_test_mae",
        "mean_combined_test_eee",
        "mean_combined_val_mae",
        "mean_inflow_test_mae",
        "mean_outflow_test_mae",
        "mean_inflow_test_eee",
        "mean_outflow_test_eee",
        "experiment_dirs",
    ]
    write_csv_rows(path, fieldnames, aggregate_rows)


def format_metric(value):
    return f"{float(value):.4f}"


def build_markdown_table(rows):
    if not rows:
        return "_No rows yet._\n"

    lines = [
        "| Rank | Trial | Schedule | q | lr | Bases | Mean combined test MAE | Std | In MAE | Out MAE | In EEE | Out EEE | Mean val MAE |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {trial_id} | {tail_schedule} | {tail_threshold_q:.2f} | {lr_init:.0e} | {bases_completed} | {mean_combined_test_mae} | {std_combined_test_mae} | {mean_inflow_test_mae} | {mean_outflow_test_mae} | {mean_inflow_test_eee} | {mean_outflow_test_eee} | {mean_combined_val_mae} |".format(
                rank=rank,
                trial_id=row["trial_id"],
                tail_schedule=row["tail_schedule"],
                tail_threshold_q=float(row["tail_threshold_q"]),
                lr_init=float(row["lr_init"]),
                bases_completed=row["bases_completed"],
                mean_combined_test_mae=format_metric(row["mean_combined_test_mae"]),
                std_combined_test_mae=format_metric(row["std_combined_test_mae"]),
                mean_inflow_test_mae=format_metric(row["mean_inflow_test_mae"]),
                mean_outflow_test_mae=format_metric(row["mean_outflow_test_mae"]),
                mean_inflow_test_eee=format_metric(row["mean_inflow_test_eee"]),
                mean_outflow_test_eee=format_metric(row["mean_outflow_test_eee"]),
                mean_combined_val_mae=format_metric(row["mean_combined_val_mae"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_leaderboard(path, aggregate_rows):
    completed_rows = [row for row in aggregate_rows if row["is_complete"]]
    partial_rows = [row for row in aggregate_rows if not row["is_complete"]]
    lines = [
        "# NYCTaxi Tail Schedule Sweep Leaderboard",
        "",
        f"Updated: {datetime.now().isoformat()}",
        "",
        "## Completed Configs",
        "",
        build_markdown_table(completed_rows).rstrip(),
        "",
        "## Partial Configs",
        "",
        build_markdown_table(partial_rows).rstrip(),
        "",
        "## Experiment Paths",
        "",
    ]
    if not aggregate_rows:
        lines.append("_No experiment paths yet._")
    else:
        for row in aggregate_rows:
            lines.append(f"### {row['trial_id']}")
            experiment_dirs = row["experiment_dirs"].split(" | ") if row["experiment_dirs"] else []
            if experiment_dirs:
                for experiment_dir in experiment_dirs:
                    lines.append(f"- `{experiment_dir}`")
            else:
                lines.append("- _No experiment directories yet._")
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_best_config(path, aggregate_rows, sweep_controls):
    completed_rows = [row for row in aggregate_rows if row["is_complete"]]
    if not completed_rows:
        if path.exists():
            path.unlink()
        return

    best_row = completed_rows[0]
    payload = {
        "dataset": "NYCTaxi",
        "ranking_metric": "mean_combined_test_mae",
        "selected_trial_id": best_row["trial_id"],
        "overrides": {
            "tail_threshold_q": float(best_row["tail_threshold_q"]),
            "lr_init": float(best_row["lr_init"]),
            "tail_schedule": best_row["tail_schedule"],
            "epochs": sweep_controls["epochs"],
            "early_stop": sweep_controls["early_stop"],
            "early_stop_patience": sweep_controls["early_stop_patience"],
            "tail_lambda_cls": sweep_controls["tail_lambda_cls"],
            "tail_lambda_gpd": sweep_controls["tail_lambda_gpd"],
            "tail_xi_min": sweep_controls["tail_xi_min"],
            "tail_xi_max": sweep_controls["tail_xi_max"],
            "tail_eps": sweep_controls["tail_eps"],
        },
        "summary": {
            "mean_combined_test_mae": float(best_row["mean_combined_test_mae"]),
            "std_combined_test_mae": float(best_row["std_combined_test_mae"]),
            "mean_combined_val_mae": float(best_row["mean_combined_val_mae"]),
            "mean_inflow_test_mae": float(best_row["mean_inflow_test_mae"]),
            "mean_outflow_test_mae": float(best_row["mean_outflow_test_mae"]),
            "mean_inflow_test_eee": float(best_row["mean_inflow_test_eee"]),
            "mean_outflow_test_eee": float(best_row["mean_outflow_test_eee"]),
        },
    }
    with open(path, "w", encoding="utf-8") as best_config_file:
        yaml.safe_dump(payload, best_config_file, sort_keys=False)


def write_top_findings(path, aggregate_rows):
    completed_rows = [row for row in aggregate_rows if row["is_complete"]]
    lines = [
        "# Top Findings",
        "",
        f"Updated: {datetime.now().isoformat()}",
        "",
    ]
    if not completed_rows:
        lines.append("No completed configs yet.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    best_overall = completed_rows[0]
    static_rows = [row for row in completed_rows if row["tail_schedule"] == "static"]
    scheduled_rows = [row for row in completed_rows if row["tail_schedule"] != "static"]
    best_static = static_rows[0] if static_rows else None
    best_scheduled = scheduled_rows[0] if scheduled_rows else None

    lines.extend(
        [
            "## Best Overall",
            "",
            f"- Trial: `{best_overall['trial_id']}`",
            f"- Schedule: `{best_overall['tail_schedule']}`",
            f"- q: `{float(best_overall['tail_threshold_q']):.2f}`",
            f"- lr: `{float(best_overall['lr_init']):.0e}`",
            f"- Mean combined test MAE: `{format_metric(best_overall['mean_combined_test_mae'])}`",
            f"- Mean combined validation MAE: `{format_metric(best_overall['mean_combined_val_mae'])}`",
            "",
            "## Best Static",
            "",
        ]
    )
    if best_static is None:
        lines.append("- No completed static config yet.")
    else:
        lines.extend(
            [
                f"- Trial: `{best_static['trial_id']}`",
                f"- Mean combined test MAE: `{format_metric(best_static['mean_combined_test_mae'])}`",
            ]
        )

    lines.extend(["", "## Best Scheduled", ""])
    if best_scheduled is None:
        lines.append("- No completed scheduled config yet.")
    else:
        lines.extend(
            [
                f"- Trial: `{best_scheduled['trial_id']}`",
                f"- Schedule: `{best_scheduled['tail_schedule']}`",
                f"- Mean combined test MAE: `{format_metric(best_scheduled['mean_combined_test_mae'])}`",
            ]
        )

    lines.extend(["", "## Schedule vs Static", ""])
    if not static_rows or not scheduled_rows:
        lines.append("- Insufficient completed configs to compare scheduled and static runs.")
    else:
        static_average = mean(float(row["mean_combined_test_mae"]) for row in static_rows)
        scheduled_average = mean(float(row["mean_combined_test_mae"]) for row in scheduled_rows)
        beats_static = scheduled_average < static_average
        lines.extend(
            [
                f"- Mean combined test MAE across completed static configs: `{format_metric(static_average)}`",
                f"- Mean combined test MAE across completed scheduled configs: `{format_metric(scheduled_average)}`",
                f"- Outcome: schedules {'beat' if beats_static else 'do not beat'} the static baseline on average.",
            ]
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_sweep_outputs(sweep_root, success_rows, failure_rows, expected_base_count, sweep_controls):
    trial_results_path = sweep_root / "trial_results.csv"
    failures_path = sweep_root / "failures.csv"
    aggregate_path = sweep_root / "aggregate_results.csv"
    leaderboard_path = sweep_root / "leaderboard.md"
    best_config_path = sweep_root / "best_config.yaml"
    top_findings_path = sweep_root / "top_findings.md"

    success_rows = sorted(success_rows, key=lambda row: (row["trial_id"], row["base_id"]))
    failure_rows = sorted(failure_rows, key=lambda row: (row["trial_id"], row["base_id"], row["timestamp"]))
    aggregate_rows = aggregate_success_rows(success_rows, expected_base_count)

    write_csv_rows(trial_results_path, TRIAL_RESULTS_FIELDS, success_rows)
    write_csv_rows(failures_path, FAILURE_FIELDS, failure_rows)
    write_aggregate_csv(aggregate_path, aggregate_rows)
    write_leaderboard(leaderboard_path, aggregate_rows)
    write_best_config(best_config_path, aggregate_rows, sweep_controls)
    write_top_findings(top_findings_path, aggregate_rows)


def launch_training_run(python_executable, config_path, base_info, comment):
    command = [
        str(python_executable),
        "main.py",
        "-cf",
        str(config_path),
        "-lp",
        str(base_info["checkpoint_path"]),
        "-s",
        str(base_info["seed"]),
        "-c",
        comment,
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME, help="Name of the sweep output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for launching runs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Base NYCTaxi config path.")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit the number of searched configs.")
    parser.add_argument("--max-bases", type=int, default=None, help="Limit the number of base checkpoints.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_SWEEP_CONTROLS["epochs"], help="Override epochs for the sweep.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_SWEEP_CONTROLS["early_stop_patience"],
        help="Override early stop patience for the sweep.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a 1-config, 1-base smoke sweep with 2 epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    python_executable = Path(args.python)
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_checkpoints = list(DEFAULT_BASE_CHECKPOINTS)
    for base_info in base_checkpoints:
        if not base_info["checkpoint_path"].is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {base_info['checkpoint_path']}")

    search_space = dict(DEFAULT_SEARCH_SPACE)
    sweep_controls = dict(DEFAULT_SWEEP_CONTROLS)
    sweep_controls["epochs"] = args.epochs
    sweep_controls["early_stop_patience"] = args.early_stop_patience

    if args.smoke:
        args.max_configs = 1 if args.max_configs is None else min(args.max_configs, 1)
        args.max_bases = 1 if args.max_bases is None else min(args.max_bases, 1)
        sweep_controls["epochs"] = 2
        sweep_controls["early_stop_patience"] = 1

    safe_sweep_name = sanitize_name(args.sweep_name)
    sweep_root = REPO_ROOT / "experiments" / "NYCTaxi" / "_sweeps" / safe_sweep_name
    trial_config_dir = sweep_root / "trial_configs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    trial_config_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sweep_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(
            build_manifest(safe_sweep_name, base_checkpoints, search_space, sweep_controls, config_path),
            manifest_file,
            indent=2,
        )

    base_config = read_yaml(config_path)
    trials = build_trials(search_space)
    if args.max_configs is not None:
        trials = trials[: args.max_configs]
    if args.max_bases is not None:
        base_checkpoints = base_checkpoints[: args.max_bases]

    trial_results_path = sweep_root / "trial_results.csv"
    failures_path = sweep_root / "failures.csv"
    success_rows = read_csv_rows(trial_results_path)
    failure_rows = read_csv_rows(failures_path)
    successful_keys = {(row["trial_id"], row["base_id"]) for row in success_rows}
    failure_index = {(row["trial_id"], row["base_id"]): row for row in failure_rows}

    print(f"Sweep root: {sweep_root}")
    print(f"Configs to search: {len(trials)}")
    print(f"Base checkpoints per config: {len(base_checkpoints)}")
    print(f"Total planned runs: {len(trials) * len(base_checkpoints)}")

    for trial_index, trial in enumerate(trials, start=1):
        trial_config = build_trial_config(base_config, trial["overrides"], sweep_controls)
        config_output_path = trial_config_dir / f"{trial['trial_id']}.yaml"
        write_yaml(config_output_path, trial_config)

        for base_info in base_checkpoints:
            key = (trial["trial_id"], base_info["base_id"])
            if key in successful_keys:
                print(f"[SKIP] {trial['trial_id']} on {base_info['base_id']} already completed.")
                continue

            comment = f"tailsweep__{safe_sweep_name}__t{trial_index:02d}__{base_info['base_id']}"
            print(
                f"[RUN] trial={trial['trial_id']} base={base_info['base_id']} "
                f"q={trial['overrides']['tail_threshold_q']:.2f} "
                f"lr={trial['overrides']['lr_init']:.0e} "
                f"schedule={trial['overrides']['tail_schedule']}"
            )
            preexisting_dirs = find_comment_directories(base_info["seed"], comment)
            completed_process = launch_training_run(
                python_executable=python_executable,
                config_path=config_output_path,
                base_info=base_info,
                comment=comment,
            )
            experiment_dir = locate_experiment_directory(
                base_seed=base_info["seed"],
                comment=comment,
                preexisting_dirs=preexisting_dirs,
            )

            results_path = experiment_dir / "results.json" if experiment_dir is not None else None
            if results_path is not None and results_path.is_file():
                payload = load_results_json(results_path)
                success_row = build_success_row(
                    trial=trial,
                    base_info=base_info,
                    config_path=config_output_path,
                    experiment_dir=experiment_dir,
                    payload=payload,
                )
                success_rows = [
                    row
                    for row in success_rows
                    if not (row["trial_id"] == trial["trial_id"] and row["base_id"] == base_info["base_id"])
                ]
                success_rows.append(success_row)
                successful_keys.add(key)
                failure_rows = [
                    row
                    for row in failure_rows
                    if not (row["trial_id"] == trial["trial_id"] and row["base_id"] == base_info["base_id"])
                ]
                failure_index.pop(key, None)
                write_sweep_outputs(
                    sweep_root=sweep_root,
                    success_rows=success_rows,
                    failure_rows=failure_rows,
                    expected_base_count=len(base_checkpoints),
                    sweep_controls=sweep_controls,
                )
                print(
                    "[DONE] trial={} base={} combined_test_mae={:.4f}".format(
                        trial["trial_id"],
                        base_info["base_id"],
                        float(success_row["combined_test_mae"]),
                    )
                )
                continue

            message = "Run finished without results.json."
            if completed_process.returncode != 0:
                message = f"Training command returned code {completed_process.returncode}."
            failure_row = build_failure_row(
                trial=trial,
                base_info=base_info,
                config_path=config_output_path,
                comment=comment,
                returncode=completed_process.returncode,
                experiment_dir=experiment_dir,
                message=message,
            )
            failure_index[key] = failure_row
            failure_rows = list(failure_index.values())
            write_sweep_outputs(
                sweep_root=sweep_root,
                success_rows=success_rows,
                failure_rows=failure_rows,
                expected_base_count=len(base_checkpoints),
                sweep_controls=sweep_controls,
            )
            print(f"[FAIL] trial={trial['trial_id']} base={base_info['base_id']} message={message}")


if __name__ == "__main__":
    main()

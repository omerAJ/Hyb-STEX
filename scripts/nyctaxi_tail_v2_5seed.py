import argparse
import csv
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "NYCTaxi.yaml"
DEFAULT_SWEEP_NAME = "nyctaxi_tail_v2_5seed"
EXISTING_BASE_CHECKPOINTS = [
    {
        "base_id": "seed1",
        "seed": 1,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=1\20260410-165331\nodewise soft-gpd tail correction\best_model_pred.pth"
        ),
        "source": "existing",
    },
    {
        "base_id": "seed2",
        "seed": 2,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=2\20260410-170812\nodewise soft-gpd tail correction\best_model_pred.pth"
        ),
        "source": "existing",
    },
    {
        "base_id": "seed3",
        "seed": 3,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=3\20260410-171900\nodewise soft-gpd tail correction\best_model_pred.pth"
        ),
        "source": "existing",
    },
]
BOOTSTRAP_SEEDS = [4, 5]
DEFAULT_SWEEP_CONTROLS = {
    "training_recipe": "tail_only",
    "epochs": 1000,
    "early_stop": True,
    "early_stop_patience": 35,
    "tail_threshold_q": 0.95,
    "lr_init": 1.0e-3,
    "tail_lambda_cls": 1.0,
    "tail_lambda_gpd": 1.0,
    "tail_mae_weight": 0.0,
    "tail_schedule": "static",
    "tail_classifier_loss_type": "bce",
    "tail_pos_weight_multiplier": 1.0,
    "tail_focal_gamma": 2.0,
    "tail_focal_alpha_pos": 0.75,
    "tail_xi_min": -0.5,
    "tail_xi_max": -0.02,
    "tail_eps": 1.0e-6,
}
STAGE_A_SCHEDULES = ["static", "cls_then_gpd_freeze", "cls_then_gpd_light_cls"]
STAGE_A_CLASSIFIER_VARIANTS = [
    {
        "classifier_label": "bce",
        "overrides": {
            "tail_classifier_loss_type": "bce",
            "tail_pos_weight_multiplier": 1.0,
            "tail_focal_gamma": 2.0,
        },
    },
    {
        "classifier_label": "weighted_bce_pw1p0",
        "overrides": {
            "tail_classifier_loss_type": "weighted_bce",
            "tail_pos_weight_multiplier": 1.0,
            "tail_focal_gamma": 2.0,
        },
    },
    {
        "classifier_label": "weighted_bce_pw2p0",
        "overrides": {
            "tail_classifier_loss_type": "weighted_bce",
            "tail_pos_weight_multiplier": 2.0,
            "tail_focal_gamma": 2.0,
        },
    },
    {
        "classifier_label": "focal_g2p0",
        "overrides": {
            "tail_classifier_loss_type": "focal",
            "tail_pos_weight_multiplier": 1.0,
            "tail_focal_gamma": 2.0,
        },
    },
    {
        "classifier_label": "focal_g3p0",
        "overrides": {
            "tail_classifier_loss_type": "focal",
            "tail_pos_weight_multiplier": 1.0,
            "tail_focal_gamma": 3.0,
        },
    },
]
XI_PRESETS = [
    {"xi_label": "xi_n0p7_to_n0p05", "tail_xi_min": -0.7, "tail_xi_max": -0.05},
    {"xi_label": "xi_n0p5_to_n0p02", "tail_xi_min": -0.5, "tail_xi_max": -0.02},
    {"xi_label": "xi_n0p3_to_n0p01", "tail_xi_min": -0.3, "tail_xi_max": -0.01},
]
STAGE_B_TAIL_MAE_WEIGHTS = [0.0, 0.05, 0.10]
TRIAL_RESULTS_FIELDS = [
    "timestamp",
    "stage_name",
    "run_id",
    "trial_id",
    "parent_trial_id",
    "base_id",
    "base_seed",
    "base_checkpoint_path",
    "config_path",
    "experiment_dir",
    "run_log_path",
    "training_recipe",
    "tail_schedule",
    "tail_classifier_loss_type",
    "tail_pos_weight_multiplier",
    "tail_focal_gamma",
    "tail_mae_weight",
    "tail_xi_min",
    "tail_xi_max",
    "tail_threshold_q",
    "lr_init",
    "epochs",
    "early_stop_patience",
    "joint_refine_epochs",
    "joint_refine_early_stop_patience",
    "joint_refine_pred_lr_scale",
    "joint_refine_classifier_lr_scale",
    "joint_refine_gpd_lr_scale",
    "joint_refine_lambda_cls",
    "joint_refine_lambda_gpd",
    "joint_refine_lambda_mae",
    "joint_refine_recompute_tail_u",
    "base_pos_weight",
    "effective_pos_weight",
    "exceedance_rate",
    "stage1_selection_metric_name",
    "stage1_best_val_metric",
    "stage1_best_val_epoch",
    "stage2_selection_metric_name",
    "stage2_best_val_metric",
    "stage2_best_val_epoch",
    "stage3_selection_metric_name",
    "stage3_best_val_metric",
    "stage3_best_val_epoch",
    "final_selection_metric_name",
    "final_best_val_metric",
    "final_best_val_epoch",
    "inflow_test_mae",
    "inflow_test_eee",
    "outflow_test_mae",
    "outflow_test_eee",
    "combined_test_mae",
    "combined_test_eee",
]
FAILURE_FIELDS = [
    "timestamp",
    "stage_name",
    "run_id",
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


def tokenize_number(value):
    return str(value).replace("-", "n").replace(".", "p")


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


def build_stage_a_trials():
    trials = []
    for schedule, classifier_variant in itertools.product(STAGE_A_SCHEDULES, STAGE_A_CLASSIFIER_VARIANTS):
        run_id = f"a{len(trials) + 1:02d}"
        trial_id = "__".join(
            [
                "stageA",
                f"sched_{schedule}",
                f"clf_{classifier_variant['classifier_label']}",
            ]
        )
        overrides = {
            "training_recipe": "tail_only",
            "tail_schedule": schedule,
            "tail_mae_weight": 0.0,
            "tail_xi_min": -0.5,
            "tail_xi_max": -0.02,
        }
        overrides.update(classifier_variant["overrides"])
        trials.append(
            {
                "stage_name": "stage_a",
                "run_id": run_id,
                "trial_id": trial_id,
                "parent_trial_id": "",
                "overrides": overrides,
            }
        )
    return trials


def build_stage_b_trials(top_stage_a_rows):
    trials = []
    for parent_index, parent_row in enumerate(top_stage_a_rows, start=1):
        parent_overrides = {
            "training_recipe": "tail_only",
            "tail_schedule": parent_row["tail_schedule"],
            "tail_classifier_loss_type": parent_row["tail_classifier_loss_type"],
            "tail_pos_weight_multiplier": float(parent_row["tail_pos_weight_multiplier"]),
            "tail_focal_gamma": float(parent_row["tail_focal_gamma"]),
        }
        for xi_preset, tail_mae_weight in itertools.product(XI_PRESETS, STAGE_B_TAIL_MAE_WEIGHTS):
            run_id = f"b{len(trials) + 1:02d}"
            trial_id = "__".join(
                [
                    "stageB",
                    f"top{parent_index}",
                    sanitize_name(parent_row["trial_id"]),
                    xi_preset["xi_label"],
                    f"mae_{tokenize_number(tail_mae_weight)}",
                ]
            )
            overrides = dict(parent_overrides)
            overrides.update(
                {
                    "tail_xi_min": xi_preset["tail_xi_min"],
                    "tail_xi_max": xi_preset["tail_xi_max"],
                    "tail_mae_weight": tail_mae_weight,
                }
            )
            trials.append(
                {
                    "stage_name": "stage_b",
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "parent_trial_id": parent_row["trial_id"],
                    "overrides": overrides,
                }
            )
    return trials


def build_trial_config(base_config, sweep_controls, overrides):
    config = dict(base_config)
    config.update(sweep_controls)
    config.update(overrides)
    return config


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
    preexisting = {str(path.resolve()) for path in preexisting_dirs}
    new_dirs = [path for path in candidate_dirs if str(path.resolve()) not in preexisting]
    if new_dirs:
        return new_dirs[-1]
    return candidate_dirs[-1]


def load_results_json(results_path):
    with open(results_path, "r", encoding="utf-8") as results_file:
        return json.load(results_file)


def format_metric(value):
    return f"{float(value):.4f}"


def read_bootstrap_manifest(path):
    if not path.exists():
        return {"created_at": datetime.now().isoformat(), "resolved_base_checkpoints": []}
    with open(path, "r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def write_bootstrap_manifest(path, resolved_base_checkpoints):
    payload = {
        "updated_at": datetime.now().isoformat(),
        "resolved_base_checkpoints": resolved_base_checkpoints,
    }
    with open(path, "w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2)


def update_bootstrap_entry(entries, new_entry):
    filtered_entries = [entry for entry in entries if int(entry["seed"]) != int(new_entry["seed"])]
    filtered_entries.append(new_entry)
    return sorted(filtered_entries, key=lambda entry: int(entry["seed"]))


def launch_training_run(python_executable, config_path, seed, comment, load_path=None):
    command = [
        str(python_executable),
        "main.py",
        "-cf",
        str(config_path),
        "-s",
        str(seed),
        "-c",
        comment,
    ]
    if load_path is not None:
        command.extend(["-lp", str(load_path)])
    return subprocess.run(command, cwd=REPO_ROOT, check=False)


def ensure_bootstrap_checkpoint(
    python_executable,
    base_config,
    bootstrap_manifest_path,
    trial_config_dir,
    sweep_controls,
    safe_sweep_name,
    seed,
):
    bootstrap_manifest = read_bootstrap_manifest(bootstrap_manifest_path)
    for entry in bootstrap_manifest["resolved_base_checkpoints"]:
        if int(entry["seed"]) == int(seed) and Path(entry["checkpoint_path"]).is_file():
            return entry

    comment = f"bootstrap_predonly__{safe_sweep_name}__seed{seed}"
    config_payload = build_trial_config(
        base_config,
        sweep_controls,
        {"training_recipe": "pred_only"},
    )
    config_path = trial_config_dir / f"bootstrap_seed{seed}.yaml"
    write_yaml(config_path, config_payload)
    preexisting_dirs = find_comment_directories(seed, comment)
    completed_process = launch_training_run(
        python_executable=python_executable,
        config_path=config_path,
        seed=seed,
        comment=comment,
        load_path=None,
    )
    experiment_dir = locate_experiment_directory(seed, comment, preexisting_dirs)
    if experiment_dir is None:
        raise RuntimeError(f"Bootstrap run for seed {seed} did not create an experiment directory.")
    checkpoint_path = experiment_dir / "best_model_pred.pth"
    results_path = experiment_dir / "results.json"
    if completed_process.returncode != 0 or not checkpoint_path.is_file() or not results_path.is_file():
        raise RuntimeError(
            f"Bootstrap run for seed {seed} failed. returncode={completed_process.returncode} checkpoint={checkpoint_path.is_file()} results={results_path.is_file()}"
        )
    entry = {
        "base_id": f"seed{seed}",
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "experiment_dir": str(experiment_dir),
        "comment": comment,
        "source": "bootstrap",
    }
    bootstrap_manifest["resolved_base_checkpoints"] = update_bootstrap_entry(
        bootstrap_manifest["resolved_base_checkpoints"],
        entry,
    )
    write_bootstrap_manifest(bootstrap_manifest_path, bootstrap_manifest["resolved_base_checkpoints"])
    return entry


def resolve_base_checkpoints(
    python_executable,
    base_config,
    bootstrap_manifest_path,
    trial_config_dir,
    sweep_controls,
    safe_sweep_name,
    smoke,
):
    resolved = []
    if not smoke:
        for base_info in EXISTING_BASE_CHECKPOINTS:
            if not base_info["checkpoint_path"].is_file():
                raise FileNotFoundError(f"Base checkpoint not found: {base_info['checkpoint_path']}")
            resolved.append(
                {
                    "base_id": base_info["base_id"],
                    "seed": base_info["seed"],
                    "checkpoint_path": str(base_info["checkpoint_path"]),
                    "source": "existing",
                    "comment": "",
                    "experiment_dir": str(base_info["checkpoint_path"].parent),
                }
            )

    bootstrap_seeds = [4] if smoke else BOOTSTRAP_SEEDS
    for seed in bootstrap_seeds:
        resolved.append(
            ensure_bootstrap_checkpoint(
                python_executable=python_executable,
                base_config=base_config,
                bootstrap_manifest_path=bootstrap_manifest_path,
                trial_config_dir=trial_config_dir,
                sweep_controls=sweep_controls,
                safe_sweep_name=safe_sweep_name,
                seed=seed,
            )
        )

    resolved = sorted(resolved, key=lambda entry: int(entry["seed"]))
    write_bootstrap_manifest(bootstrap_manifest_path, resolved)
    return resolved


def build_manifest(safe_sweep_name, config_path, sweep_controls):
    return {
        "sweep_name": safe_sweep_name,
        "created_at": datetime.now().isoformat(),
        "dataset": "NYCTaxi",
        "config_path": str(config_path),
        "ranking_metric": "mean_combined_test_mae",
        "ranking_tiebreakers": [
            "std_combined_test_mae",
            "mean_combined_val_mae",
            "mean_combined_test_eee",
        ],
        "sweep_controls": sweep_controls,
        "stage_a": {
            "schedules": STAGE_A_SCHEDULES,
            "classifier_variants": STAGE_A_CLASSIFIER_VARIANTS,
            "fixed_tail_xi_min": -0.5,
            "fixed_tail_xi_max": -0.02,
            "fixed_tail_mae_weight": 0.0,
        },
        "stage_b": {
            "top_stage_a_configs": 2,
            "xi_presets": XI_PRESETS,
            "tail_mae_weights": STAGE_B_TAIL_MAE_WEIGHTS,
        },
    }


def build_success_row(trial, base_info, config_path, experiment_dir, payload):
    tail_results = payload["tail"]
    stage1_results = tail_results.get("stage1")
    stage2_results = tail_results.get("stage2")
    stage3_results = tail_results.get("stage3")
    final_results = stage3_results or stage2_results or tail_results
    test_metrics = final_results["test_metrics"]
    inflow_test_mae = float(test_metrics["inflow"]["mae"])
    inflow_test_eee = float(test_metrics["inflow"]["eee"])
    outflow_test_mae = float(test_metrics["outflow"]["mae"])
    outflow_test_eee = float(test_metrics["outflow"]["eee"])
    tail_target_stats = tail_results.get("tail_target_stats", {})
    hyperparameters = payload["hyperparameters"]
    return {
        "timestamp": datetime.now().isoformat(),
        "stage_name": trial["stage_name"],
        "run_id": trial["run_id"],
        "trial_id": trial["trial_id"],
        "parent_trial_id": trial.get("parent_trial_id", ""),
        "base_id": base_info["base_id"],
        "base_seed": base_info["seed"],
        "base_checkpoint_path": base_info["checkpoint_path"],
        "config_path": str(config_path),
        "experiment_dir": str(experiment_dir),
        "run_log_path": str(experiment_dir / "run.log"),
        "training_recipe": hyperparameters["training_recipe"],
        "tail_schedule": hyperparameters["tail_schedule"],
        "tail_classifier_loss_type": hyperparameters["tail_classifier_loss_type"],
        "tail_pos_weight_multiplier": hyperparameters["tail_pos_weight_multiplier"],
        "tail_focal_gamma": hyperparameters["tail_focal_gamma"],
        "tail_mae_weight": hyperparameters["tail_mae_weight"],
        "tail_xi_min": hyperparameters["tail_xi_min"],
        "tail_xi_max": hyperparameters["tail_xi_max"],
        "tail_threshold_q": hyperparameters["tail_threshold_q"],
        "lr_init": hyperparameters["lr_init"],
        "epochs": hyperparameters["epochs"],
        "early_stop_patience": hyperparameters["early_stop_patience"],
        "joint_refine_epochs": hyperparameters.get("joint_refine_epochs"),
        "joint_refine_early_stop_patience": hyperparameters.get("joint_refine_early_stop_patience"),
        "joint_refine_pred_lr_scale": hyperparameters.get("joint_refine_pred_lr_scale"),
        "joint_refine_classifier_lr_scale": hyperparameters.get("joint_refine_classifier_lr_scale"),
        "joint_refine_gpd_lr_scale": hyperparameters.get("joint_refine_gpd_lr_scale"),
        "joint_refine_lambda_cls": hyperparameters.get("joint_refine_lambda_cls"),
        "joint_refine_lambda_gpd": hyperparameters.get("joint_refine_lambda_gpd"),
        "joint_refine_lambda_mae": hyperparameters.get("joint_refine_lambda_mae"),
        "joint_refine_recompute_tail_u": hyperparameters.get("joint_refine_recompute_tail_u"),
        "base_pos_weight": float(tail_target_stats.get("base_pos_weight", 1.0)),
        "effective_pos_weight": float(tail_target_stats.get("effective_pos_weight", 1.0)),
        "exceedance_rate": float(tail_target_stats.get("exceedance_rate", 0.0)),
        "stage1_selection_metric_name": "" if stage1_results is None else stage1_results["selection_metric_name"],
        "stage1_best_val_metric": "" if stage1_results is None else float(stage1_results["best_val_metric"]),
        "stage1_best_val_epoch": "" if stage1_results is None else int(stage1_results["best_val_epoch"]),
        "stage2_selection_metric_name": "" if stage2_results is None else stage2_results["selection_metric_name"],
        "stage2_best_val_metric": "" if stage2_results is None else float(stage2_results["best_val_metric"]),
        "stage2_best_val_epoch": "" if stage2_results is None else int(stage2_results["best_val_epoch"]),
        "stage3_selection_metric_name": "" if stage3_results is None else stage3_results["selection_metric_name"],
        "stage3_best_val_metric": "" if stage3_results is None else float(stage3_results["best_val_metric"]),
        "stage3_best_val_epoch": "" if stage3_results is None else int(stage3_results["best_val_epoch"]),
        "final_selection_metric_name": final_results["selection_metric_name"],
        "final_best_val_metric": float(final_results["best_val_metric"]),
        "final_best_val_epoch": int(final_results["best_val_epoch"]),
        "inflow_test_mae": inflow_test_mae,
        "inflow_test_eee": inflow_test_eee,
        "outflow_test_mae": outflow_test_mae,
        "outflow_test_eee": outflow_test_eee,
        "combined_test_mae": (inflow_test_mae + outflow_test_mae) / 2.0,
        "combined_test_eee": (inflow_test_eee + outflow_test_eee) / 2.0,
    }


def build_failure_row(trial, base_info, config_path, comment, returncode, experiment_dir, message):
    experiment_dir_str = str(experiment_dir) if experiment_dir is not None else ""
    return {
        "timestamp": datetime.now().isoformat(),
        "stage_name": trial["stage_name"],
        "run_id": trial["run_id"],
        "trial_id": trial["trial_id"],
        "base_id": base_info["base_id"],
        "base_seed": base_info["seed"],
        "base_checkpoint_path": base_info["checkpoint_path"],
        "config_path": str(config_path),
        "comment": comment,
        "returncode": returncode,
        "experiment_dir": experiment_dir_str,
        "run_log_path": str(Path(experiment_dir_str) / "run.log") if experiment_dir else "",
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
        combined_val_maes = [float(row["final_best_val_metric"]) for row in rows]
        inflow_test_maes = [float(row["inflow_test_mae"]) for row in rows]
        outflow_test_maes = [float(row["outflow_test_mae"]) for row in rows]
        inflow_test_eees = [float(row["inflow_test_eee"]) for row in rows]
        outflow_test_eees = [float(row["outflow_test_eee"]) for row in rows]
        aggregate_rows.append(
            {
                "stage_name": first_row["stage_name"],
                "run_id": first_row["run_id"],
                "trial_id": trial_id,
                "parent_trial_id": first_row["parent_trial_id"],
                "training_recipe": first_row["training_recipe"],
                "tail_schedule": first_row["tail_schedule"],
                "tail_classifier_loss_type": first_row["tail_classifier_loss_type"],
                "tail_pos_weight_multiplier": float(first_row["tail_pos_weight_multiplier"]),
                "tail_focal_gamma": float(first_row["tail_focal_gamma"]),
                "tail_mae_weight": float(first_row["tail_mae_weight"]),
                "tail_xi_min": float(first_row["tail_xi_min"]),
                "tail_xi_max": float(first_row["tail_xi_max"]),
                "tail_threshold_q": float(first_row["tail_threshold_q"]),
                "lr_init": float(first_row["lr_init"]),
                "joint_refine_epochs": float(first_row["joint_refine_epochs"]) if first_row["joint_refine_epochs"] != "" else "",
                "joint_refine_early_stop_patience": float(first_row["joint_refine_early_stop_patience"]) if first_row["joint_refine_early_stop_patience"] != "" else "",
                "joint_refine_pred_lr_scale": float(first_row["joint_refine_pred_lr_scale"]) if first_row["joint_refine_pred_lr_scale"] != "" else "",
                "joint_refine_classifier_lr_scale": float(first_row["joint_refine_classifier_lr_scale"]) if first_row["joint_refine_classifier_lr_scale"] != "" else "",
                "joint_refine_gpd_lr_scale": float(first_row["joint_refine_gpd_lr_scale"]) if first_row["joint_refine_gpd_lr_scale"] != "" else "",
                "joint_refine_lambda_cls": float(first_row["joint_refine_lambda_cls"]) if first_row["joint_refine_lambda_cls"] != "" else "",
                "joint_refine_lambda_gpd": float(first_row["joint_refine_lambda_gpd"]) if first_row["joint_refine_lambda_gpd"] != "" else "",
                "joint_refine_lambda_mae": float(first_row["joint_refine_lambda_mae"]) if first_row["joint_refine_lambda_mae"] != "" else "",
                "joint_refine_recompute_tail_u": first_row["joint_refine_recompute_tail_u"],
                "bases_completed": len(rows),
                "is_complete": len(rows) == expected_base_count,
                "mean_combined_test_mae": mean(combined_test_maes),
                "std_combined_test_mae": pstdev(combined_test_maes) if len(combined_test_maes) > 1 else 0.0,
                "mean_combined_test_eee": mean(combined_test_eees),
                "mean_combined_val_mae": mean(combined_val_maes),
                "mean_inflow_test_mae": mean(inflow_test_maes),
                "mean_outflow_test_mae": mean(outflow_test_maes),
                "mean_inflow_test_eee": mean(inflow_test_eees),
                "mean_outflow_test_eee": mean(outflow_test_eees),
                "mean_effective_pos_weight": mean(float(row["effective_pos_weight"]) for row in rows),
                "mean_exceedance_rate": mean(float(row["exceedance_rate"]) for row in rows),
                "experiment_dirs": " | ".join(row["experiment_dir"] for row in rows),
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
        "stage_name",
        "run_id",
        "trial_id",
        "parent_trial_id",
        "training_recipe",
        "tail_schedule",
        "tail_classifier_loss_type",
        "tail_pos_weight_multiplier",
        "tail_focal_gamma",
        "tail_mae_weight",
        "tail_xi_min",
        "tail_xi_max",
        "tail_threshold_q",
        "lr_init",
        "joint_refine_epochs",
        "joint_refine_early_stop_patience",
        "joint_refine_pred_lr_scale",
        "joint_refine_classifier_lr_scale",
        "joint_refine_gpd_lr_scale",
        "joint_refine_lambda_cls",
        "joint_refine_lambda_gpd",
        "joint_refine_lambda_mae",
        "joint_refine_recompute_tail_u",
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
        "mean_effective_pos_weight",
        "mean_exceedance_rate",
        "experiment_dirs",
    ]
    write_csv_rows(path, fieldnames, aggregate_rows)


def build_markdown_table(rows):
    if not rows:
        return "_No rows yet._\n"
    lines = [
        "| Rank | Stage | Trial | Schedule | Clf Loss | Xi Bounds | Tail MAE | Bases | Mean combined test MAE | Std | In MAE | Out MAE | Mean val MAE |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        xi_bounds = f"[{row['tail_xi_min']:.2f}, {row['tail_xi_max']:.2f}]"
        classifier_label = row["tail_classifier_loss_type"]
        if row["tail_classifier_loss_type"] == "weighted_bce":
            classifier_label += f" x{row['tail_pos_weight_multiplier']:.1f}"
        elif row["tail_classifier_loss_type"] == "focal":
            classifier_label += f" g={row['tail_focal_gamma']:.1f}"
        lines.append(
            "| {rank} | {stage_name} | {trial_id} | {tail_schedule} | {classifier_label} | {xi_bounds} | {tail_mae_weight:.2f} | {bases_completed} | {mean_combined_test_mae} | {std_combined_test_mae} | {mean_inflow_test_mae} | {mean_outflow_test_mae} | {mean_combined_val_mae} |".format(
                rank=rank,
                stage_name=row["stage_name"],
                trial_id=row["trial_id"],
                tail_schedule=row["tail_schedule"],
                classifier_label=classifier_label,
                xi_bounds=xi_bounds,
                tail_mae_weight=float(row["tail_mae_weight"]),
                bases_completed=row["bases_completed"],
                mean_combined_test_mae=format_metric(row["mean_combined_test_mae"]),
                std_combined_test_mae=format_metric(row["std_combined_test_mae"]),
                mean_inflow_test_mae=format_metric(row["mean_inflow_test_mae"]),
                mean_outflow_test_mae=format_metric(row["mean_outflow_test_mae"]),
                mean_combined_val_mae=format_metric(row["mean_combined_val_mae"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_leaderboard(path, aggregate_rows):
    completed_rows = [row for row in aggregate_rows if row["is_complete"]]
    partial_rows = [row for row in aggregate_rows if not row["is_complete"]]
    lines = [
        "# NYCTaxi Tail V2 5-Seed Leaderboard",
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
            for experiment_dir in row["experiment_dirs"].split(" | "):
                lines.append(f"- `{experiment_dir}`")
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
            "training_recipe": best_row["training_recipe"],
            "tail_threshold_q": float(best_row["tail_threshold_q"]),
            "lr_init": float(best_row["lr_init"]),
            "epochs": sweep_controls["epochs"],
            "early_stop": sweep_controls["early_stop"],
            "early_stop_patience": sweep_controls["early_stop_patience"],
            "tail_schedule": best_row["tail_schedule"],
            "tail_classifier_loss_type": best_row["tail_classifier_loss_type"],
            "tail_pos_weight_multiplier": float(best_row["tail_pos_weight_multiplier"]),
            "tail_focal_gamma": float(best_row["tail_focal_gamma"]),
            "tail_focal_alpha_pos": sweep_controls["tail_focal_alpha_pos"],
            "tail_mae_weight": float(best_row["tail_mae_weight"]),
            "tail_lambda_cls": sweep_controls["tail_lambda_cls"],
            "tail_lambda_gpd": sweep_controls["tail_lambda_gpd"],
            "tail_xi_min": float(best_row["tail_xi_min"]),
            "tail_xi_max": float(best_row["tail_xi_max"]),
            "tail_eps": sweep_controls["tail_eps"],
            "joint_refine_epochs": int(float(best_row["joint_refine_epochs"])) if best_row["joint_refine_epochs"] != "" else sweep_controls.get("joint_refine_epochs"),
            "joint_refine_early_stop_patience": int(float(best_row["joint_refine_early_stop_patience"])) if best_row["joint_refine_early_stop_patience"] != "" else sweep_controls.get("joint_refine_early_stop_patience"),
            "joint_refine_pred_lr_scale": float(best_row["joint_refine_pred_lr_scale"]) if best_row["joint_refine_pred_lr_scale"] != "" else sweep_controls.get("joint_refine_pred_lr_scale"),
            "joint_refine_classifier_lr_scale": float(best_row["joint_refine_classifier_lr_scale"]) if best_row["joint_refine_classifier_lr_scale"] != "" else sweep_controls.get("joint_refine_classifier_lr_scale"),
            "joint_refine_gpd_lr_scale": float(best_row["joint_refine_gpd_lr_scale"]) if best_row["joint_refine_gpd_lr_scale"] != "" else sweep_controls.get("joint_refine_gpd_lr_scale"),
            "joint_refine_lambda_cls": float(best_row["joint_refine_lambda_cls"]) if best_row["joint_refine_lambda_cls"] != "" else sweep_controls.get("joint_refine_lambda_cls"),
            "joint_refine_lambda_gpd": float(best_row["joint_refine_lambda_gpd"]) if best_row["joint_refine_lambda_gpd"] != "" else sweep_controls.get("joint_refine_lambda_gpd"),
            "joint_refine_lambda_mae": float(best_row["joint_refine_lambda_mae"]) if best_row["joint_refine_lambda_mae"] != "" else sweep_controls.get("joint_refine_lambda_mae"),
            "joint_refine_recompute_tail_u": (
                str(best_row["joint_refine_recompute_tail_u"]).lower() == "true"
                if best_row["joint_refine_recompute_tail_u"] != ""
                else sweep_controls.get("joint_refine_recompute_tail_u")
            ),
        },
        "summary": {
            "stage_name": best_row["stage_name"],
            "mean_combined_test_mae": float(best_row["mean_combined_test_mae"]),
            "std_combined_test_mae": float(best_row["std_combined_test_mae"]),
            "mean_combined_val_mae": float(best_row["mean_combined_val_mae"]),
            "mean_inflow_test_mae": float(best_row["mean_inflow_test_mae"]),
            "mean_outflow_test_mae": float(best_row["mean_outflow_test_mae"]),
            "mean_inflow_test_eee": float(best_row["mean_inflow_test_eee"]),
            "mean_outflow_test_eee": float(best_row["mean_outflow_test_eee"]),
            "mean_effective_pos_weight": float(best_row["mean_effective_pos_weight"]),
            "mean_exceedance_rate": float(best_row["mean_exceedance_rate"]),
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
    stage_a_rows = [row for row in completed_rows if row["stage_name"] == "stage_a"]
    stage_b_rows = [row for row in completed_rows if row["stage_name"] == "stage_b"]
    best_stage_a = stage_a_rows[0] if stage_a_rows else None
    best_stage_b = stage_b_rows[0] if stage_b_rows else None
    lines.extend(
        [
            "## Best Overall",
            "",
            f"- Trial: `{best_overall['trial_id']}`",
            f"- Stage: `{best_overall['stage_name']}`",
            f"- Schedule: `{best_overall['tail_schedule']}`",
            f"- Classifier loss: `{best_overall['tail_classifier_loss_type']}`",
            f"- Xi bounds: `[{best_overall['tail_xi_min']:.2f}, {best_overall['tail_xi_max']:.2f}]`",
            f"- Tail MAE weight: `{float(best_overall['tail_mae_weight']):.2f}`",
            f"- Mean combined test MAE: `{format_metric(best_overall['mean_combined_test_mae'])}`",
            "",
            "## Best Stage A",
            "",
        ]
    )
    if best_stage_a is None:
        lines.append("- No completed Stage A config yet.")
    else:
        lines.extend(
            [
                f"- Trial: `{best_stage_a['trial_id']}`",
                f"- Schedule: `{best_stage_a['tail_schedule']}`",
                f"- Classifier loss: `{best_stage_a['tail_classifier_loss_type']}`",
                f"- Mean combined test MAE: `{format_metric(best_stage_a['mean_combined_test_mae'])}`",
            ]
        )
    lines.extend(["", "## Best Stage B", ""])
    if best_stage_b is None:
        lines.append("- No completed Stage B config yet.")
    else:
        lines.extend(
            [
                f"- Trial: `{best_stage_b['trial_id']}`",
                f"- Xi bounds: `[{best_stage_b['tail_xi_min']:.2f}, {best_stage_b['tail_xi_max']:.2f}]`",
                f"- Tail MAE weight: `{float(best_stage_b['tail_mae_weight']):.2f}`",
                f"- Mean combined test MAE: `{format_metric(best_stage_b['mean_combined_test_mae'])}`",
            ]
        )
    lines.extend(["", "## Interpretation", ""])
    if best_stage_a is not None and best_stage_b is not None:
        improvement = float(best_stage_a["mean_combined_test_mae"]) - float(best_stage_b["mean_combined_test_mae"])
        lines.append(
            f"- Stage B {'improves on' if improvement > 0 else 'does not improve on'} the best Stage A config by `{abs(improvement):.4f}` combined test MAE."
        )
    lines.extend(
        [
            f"- Winning schedule: `{best_overall['tail_schedule']}`",
            f"- Winning classifier strategy: `{best_overall['tail_classifier_loss_type']}`",
            f"- Winning effective class-weight mean: `{format_metric(best_overall['mean_effective_pos_weight'])}`",
            f"- Winning exceedance-rate mean: `{format_metric(best_overall['mean_exceedance_rate'])}`",
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

    success_rows = sorted(success_rows, key=lambda row: (row["stage_name"], row["trial_id"], row["base_id"]))
    failure_rows = sorted(failure_rows, key=lambda row: (row["stage_name"], row["trial_id"], row["base_id"], row["timestamp"]))
    aggregate_rows = aggregate_success_rows(success_rows, expected_base_count)
    write_csv_rows(trial_results_path, TRIAL_RESULTS_FIELDS, success_rows)
    write_csv_rows(failures_path, FAILURE_FIELDS, failure_rows)
    write_aggregate_csv(aggregate_path, aggregate_rows)
    write_leaderboard(leaderboard_path, aggregate_rows)
    write_best_config(best_config_path, aggregate_rows, sweep_controls)
    write_top_findings(top_findings_path, aggregate_rows)
    return aggregate_rows


def write_manifest(path, payload):
    with open(path, "w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2)


def run_trial_set(
    stage_trials,
    base_config,
    base_checkpoints,
    sweep_controls,
    python_executable,
    sweep_root,
    trial_config_dir,
    safe_sweep_name,
    success_rows,
    failure_rows,
):
    trial_results_path = sweep_root / "trial_results.csv"
    failures_path = sweep_root / "failures.csv"
    successful_keys = {(row["trial_id"], row["base_id"]) for row in success_rows}
    failure_index = {(row["trial_id"], row["base_id"]): row for row in failure_rows}

    for trial in stage_trials:
        stage_dir = trial_config_dir / trial["stage_name"]
        config_path = stage_dir / f"{trial['run_id']}__{sanitize_name(trial['trial_id'])}.yaml"
        trial_config = build_trial_config(base_config, sweep_controls, trial["overrides"])
        write_yaml(config_path, trial_config)

        for base_info in base_checkpoints:
            key = (trial["trial_id"], base_info["base_id"])
            if key in successful_keys:
                print(f"[SKIP] {trial['trial_id']} on {base_info['base_id']} already completed.")
                continue

            comment = f"tailsweepv2__{safe_sweep_name}__{trial['run_id']}__{base_info['base_id']}"
            print(
                f"[RUN] stage={trial['stage_name']} trial={trial['trial_id']} base={base_info['base_id']} "
                f"schedule={trial['overrides']['tail_schedule']} clf={trial['overrides']['tail_classifier_loss_type']}"
            )
            preexisting_dirs = find_comment_directories(base_info["seed"], comment)
            completed_process = launch_training_run(
                python_executable=python_executable,
                config_path=config_path,
                seed=base_info["seed"],
                comment=comment,
                load_path=base_info["checkpoint_path"],
            )
            experiment_dir = locate_experiment_directory(base_info["seed"], comment, preexisting_dirs)
            results_path = experiment_dir / "results.json" if experiment_dir is not None else None

            if results_path is not None and results_path.is_file():
                payload = load_results_json(results_path)
                success_row = build_success_row(
                    trial=trial,
                    base_info=base_info,
                    config_path=config_path,
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
                    "[DONE] stage={} trial={} base={} combined_test_mae={:.4f}".format(
                        trial["stage_name"],
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
                config_path=config_path,
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
            print(f"[FAIL] stage={trial['stage_name']} trial={trial['trial_id']} base={base_info['base_id']} message={message}")

    success_rows = read_csv_rows(trial_results_path)
    failure_rows = read_csv_rows(failures_path)
    return success_rows, failure_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME, help="Name of the sweep output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Base NYCTaxi config path.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_SWEEP_CONTROLS["epochs"], help="Max epochs to use.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_SWEEP_CONTROLS["early_stop_patience"],
        help="Early stop patience to use.",
    )
    parser.add_argument("--max-stage-a-configs", type=int, default=None, help="Limit Stage A configs.")
    parser.add_argument("--max-stage-b-configs", type=int, default=None, help="Limit Stage B configs.")
    parser.add_argument("--smoke", action="store_true", help="Run a bootstrap + 1-config smoke test.")
    return parser.parse_args()


def main():
    args = parse_args()
    python_executable = Path(args.python)
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    safe_sweep_name = sanitize_name(args.sweep_name)
    sweep_root = REPO_ROOT / "experiments" / "NYCTaxi" / "_sweeps" / safe_sweep_name
    trial_config_dir = sweep_root / "trial_configs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    trial_config_dir.mkdir(parents=True, exist_ok=True)

    sweep_controls = dict(DEFAULT_SWEEP_CONTROLS)
    sweep_controls["epochs"] = args.epochs
    sweep_controls["early_stop_patience"] = args.early_stop_patience
    if args.smoke:
        sweep_controls["epochs"] = 2
        sweep_controls["early_stop_patience"] = 1

    manifest_path = sweep_root / "manifest.json"
    bootstrap_manifest_path = sweep_root / "bootstrap_manifest.json"
    write_manifest(manifest_path, build_manifest(safe_sweep_name, config_path, sweep_controls))

    base_config = read_yaml(config_path)
    base_checkpoints = resolve_base_checkpoints(
        python_executable=python_executable,
        base_config=base_config,
        bootstrap_manifest_path=bootstrap_manifest_path,
        trial_config_dir=trial_config_dir,
        sweep_controls=sweep_controls,
        safe_sweep_name=safe_sweep_name,
        smoke=args.smoke,
    )

    stage_a_trials = build_stage_a_trials()
    if args.max_stage_a_configs is not None:
        stage_a_trials = stage_a_trials[: args.max_stage_a_configs]
    if args.smoke:
        stage_a_trials = stage_a_trials[:1]

    success_rows = read_csv_rows(sweep_root / "trial_results.csv")
    failure_rows = read_csv_rows(sweep_root / "failures.csv")

    print(f"Sweep root: {sweep_root}")
    print(f"Base checkpoints in use: {len(base_checkpoints)}")
    print(f"Stage A configs: {len(stage_a_trials)}")
    print(f"Stage A planned runs: {len(stage_a_trials) * len(base_checkpoints)}")

    success_rows, failure_rows = run_trial_set(
        stage_trials=stage_a_trials,
        base_config=base_config,
        base_checkpoints=base_checkpoints,
        sweep_controls=sweep_controls,
        python_executable=python_executable,
        sweep_root=sweep_root,
        trial_config_dir=trial_config_dir,
        safe_sweep_name=safe_sweep_name,
        success_rows=success_rows,
        failure_rows=failure_rows,
    )
    aggregate_rows = write_sweep_outputs(
        sweep_root=sweep_root,
        success_rows=success_rows,
        failure_rows=failure_rows,
        expected_base_count=len(base_checkpoints),
        sweep_controls=sweep_controls,
    )

    completed_stage_a = [row for row in aggregate_rows if row["stage_name"] == "stage_a" and row["is_complete"]]
    if args.smoke:
        return
    if len(completed_stage_a) != len(stage_a_trials):
        print("Stage A is not fully complete yet. Re-run the script to continue before Stage B starts.")
        return

    top_stage_a_rows = completed_stage_a[:2]
    stage_b_trials = build_stage_b_trials(top_stage_a_rows)
    if args.max_stage_b_configs is not None:
        stage_b_trials = stage_b_trials[: args.max_stage_b_configs]
    print(f"Stage B configs: {len(stage_b_trials)}")
    print(f"Stage B planned runs: {len(stage_b_trials) * len(base_checkpoints)}")

    success_rows, failure_rows = run_trial_set(
        stage_trials=stage_b_trials,
        base_config=base_config,
        base_checkpoints=base_checkpoints,
        sweep_controls=sweep_controls,
        python_executable=python_executable,
        sweep_root=sweep_root,
        trial_config_dir=trial_config_dir,
        safe_sweep_name=safe_sweep_name,
        success_rows=success_rows,
        failure_rows=failure_rows,
    )
    write_sweep_outputs(
        sweep_root=sweep_root,
        success_rows=success_rows,
        failure_rows=failure_rows,
        expected_base_count=len(base_checkpoints),
        sweep_controls=sweep_controls,
    )


if __name__ == "__main__":
    main()

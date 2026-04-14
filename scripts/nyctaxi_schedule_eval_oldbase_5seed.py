import argparse
from pathlib import Path

import nyctaxi_tail_oldbase_focus_5seed as oldbase
import nyctaxi_tail_v2_5seed as base


DEFAULT_SWEEP_NAME = "nyctaxi_schedule_eval_oldbase_2seed"
DEFAULT_SWEEP_CONTROLS = dict(oldbase.DEFAULT_SWEEP_CONTROLS)
DEFAULT_SWEEP_CONTROLS.update(
    {
        "training_recipe": "tail_only",
        "tail_threshold_q": 0.93,
        "tail_mae_weight": 0.15,
        "tail_schedule": "cls_then_gpd_light_cls",
        "tail_classifier_loss_type": "bce",
        "tail_xi_min": -0.5,
        "tail_xi_max": -0.02,
        "joint_refine_epochs": 200,
        "joint_refine_early_stop_patience": 15,
        "joint_refine_pred_lr_scale": 0.05,
        "joint_refine_classifier_lr_scale": 1.0,
        "joint_refine_gpd_lr_scale": 1.0,
        "joint_refine_lambda_cls": 0.05,
        "joint_refine_lambda_gpd": 1.0,
        "joint_refine_lambda_mae": 0.15,
        "joint_refine_recompute_tail_u": False,
    }
)
STAGE_A_TRIAL_SPECS = [
    {
        "schedule_label": "static",
        "overrides": {
            "tail_schedule": "static",
        },
    },
    {
        "schedule_label": "cls_then_gpd_light_cls",
        "overrides": {
            "tail_schedule": "cls_then_gpd_light_cls",
        },
    },
    {
        "schedule_label": "static_then_joint_predlr_0p03",
        "overrides": {
            "tail_schedule": "static_then_joint",
            "joint_refine_pred_lr_scale": 0.03,
            "joint_refine_lambda_cls": 0.05,
        },
    },
    {
        "schedule_label": "static_then_joint_predlr_0p10",
        "overrides": {
            "tail_schedule": "static_then_joint",
            "joint_refine_pred_lr_scale": 0.10,
            "joint_refine_lambda_cls": 0.05,
        },
    },
    {
        "schedule_label": "lightcls_then_joint_predlr_0p03",
        "overrides": {
            "tail_schedule": "cls_then_gpd_light_cls_then_joint",
            "joint_refine_pred_lr_scale": 0.03,
            "joint_refine_lambda_cls": 0.05,
        },
    },
    {
        "schedule_label": "lightcls_then_joint_predlr_0p10",
        "overrides": {
            "tail_schedule": "cls_then_gpd_light_cls_then_joint",
            "joint_refine_pred_lr_scale": 0.10,
            "joint_refine_lambda_cls": 0.05,
        },
    },
]
STAGE_B_PRED_LR_SCALES = [0.01, 0.03, 0.10]
STAGE_B_JOINT_LAMBDA_CLS = [0.02, 0.05, 0.10]


def build_stage_a_trials():
    trials = []
    for spec in STAGE_A_TRIAL_SPECS:
        run_id = f"a{len(trials) + 1:02d}"
        trial_id = "__".join(
            [
                "stageA",
                spec["schedule_label"],
            ]
        )
        overrides = {
            "training_recipe": "tail_only",
            "tail_threshold_q": 0.93,
            "tail_mae_weight": 0.15,
            "tail_classifier_loss_type": "bce",
            "tail_pos_weight_multiplier": 1.0,
            "tail_focal_gamma": 2.0,
            "tail_xi_min": -0.5,
            "tail_xi_max": -0.02,
            "joint_refine_epochs": 200,
            "joint_refine_early_stop_patience": 15,
            "joint_refine_classifier_lr_scale": 1.0,
            "joint_refine_gpd_lr_scale": 1.0,
            "joint_refine_lambda_gpd": 1.0,
            "joint_refine_lambda_mae": 0.15,
            "joint_refine_recompute_tail_u": False,
        }
        overrides.update(spec["overrides"])
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


def build_stage_b_trials(top_joint_rows):
    trials = []
    for parent_index, parent_row in enumerate(top_joint_rows, start=1):
        for pred_lr_scale in STAGE_B_PRED_LR_SCALES:
            for lambda_cls in STAGE_B_JOINT_LAMBDA_CLS:
                run_id = f"b{len(trials) + 1:02d}"
                trial_id = "__".join(
                    [
                        "stageB",
                        f"top{parent_index}",
                        base.sanitize_name(parent_row["trial_id"]),
                        f"predlr_{base.tokenize_number(pred_lr_scale)}",
                        f"jointcls_{base.tokenize_number(lambda_cls)}",
                    ]
                )
                overrides = {
                    "training_recipe": "tail_only",
                    "tail_schedule": parent_row["tail_schedule"],
                    "tail_threshold_q": float(parent_row["tail_threshold_q"]),
                    "tail_mae_weight": float(parent_row["tail_mae_weight"]),
                    "tail_classifier_loss_type": "bce",
                    "tail_pos_weight_multiplier": 1.0,
                    "tail_focal_gamma": 2.0,
                    "tail_xi_min": float(parent_row["tail_xi_min"]),
                    "tail_xi_max": float(parent_row["tail_xi_max"]),
                    "joint_refine_epochs": 200,
                    "joint_refine_early_stop_patience": 15,
                    "joint_refine_pred_lr_scale": pred_lr_scale,
                    "joint_refine_classifier_lr_scale": 1.0,
                    "joint_refine_gpd_lr_scale": 1.0,
                    "joint_refine_lambda_cls": lambda_cls,
                    "joint_refine_lambda_gpd": 1.0,
                    "joint_refine_lambda_mae": float(parent_row["tail_mae_weight"]),
                    "joint_refine_recompute_tail_u": False,
                }
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


def build_manifest(safe_sweep_name, config_path, sweep_controls, base_checkpoints, selected_base_seeds):
    return {
        "sweep_name": safe_sweep_name,
        "created_at": base.datetime.now().isoformat(),
        "dataset": "NYCTaxi",
        "base_checkpoint_source": "fix phase-wise training evs_90",
        "selected_base_seeds": selected_base_seeds,
        "config_path": str(config_path),
        "ranking_metric": "mean_combined_test_mae",
        "ranking_tiebreakers": [
            "std_combined_test_mae",
            "mean_combined_val_mae",
            "mean_combined_test_eee",
        ],
        "sweep_controls": sweep_controls,
        "base_checkpoints": base_checkpoints,
        "stage_a": {
            "description": "Compare schedule families, including a late joint-refine stage after the tail head is in place.",
            "fixed_tail_threshold_q": 0.93,
            "fixed_tail_mae_weight": 0.15,
            "fixed_tail_xi_min": -0.5,
            "fixed_tail_xi_max": -0.02,
            "trial_specs": STAGE_A_TRIAL_SPECS,
        },
        "stage_b": {
            "description": "Refine only the best joint schedules by sweeping joint pred LR scale and stage-3 classifier regularization.",
            "top_joint_stage_a_configs": 2,
            "pred_lr_scales": STAGE_B_PRED_LR_SCALES,
            "joint_refine_lambda_cls": STAGE_B_JOINT_LAMBDA_CLS,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME, help="Name of the sweep output directory.")
    parser.add_argument("--python", default=base.sys.executable, help="Python executable to use.")
    parser.add_argument("--config", default=str(base.DEFAULT_CONFIG_PATH), help="Base NYCTaxi config path.")
    parser.add_argument(
        "--base-seeds",
        default="1,2",
        help="Comma-separated old-base seeds to use for the sweep. Default: 1,2",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_SWEEP_CONTROLS["epochs"], help="Max epochs to use.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_SWEEP_CONTROLS["early_stop_patience"],
        help="Early stop patience to use.",
    )
    parser.add_argument("--max-stage-a-configs", type=int, default=None, help="Limit Stage A configs.")
    parser.add_argument("--max-stage-b-configs", type=int, default=None, help="Limit Stage B configs.")
    parser.add_argument("--smoke", action="store_true", help="Run a 1-base, 1-config smoke test.")
    return parser.parse_args()


def main():
    args = parse_args()
    python_executable = Path(args.python)
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    selected_base_seeds = [int(seed.strip()) for seed in args.base_seeds.split(",") if seed.strip()]
    if not selected_base_seeds:
        raise ValueError("At least one base seed must be specified.")

    safe_sweep_name = base.sanitize_name(args.sweep_name)
    sweep_root = base.REPO_ROOT / "experiments" / "NYCTaxi" / "_sweeps" / safe_sweep_name
    trial_config_dir = sweep_root / "trial_configs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    trial_config_dir.mkdir(parents=True, exist_ok=True)

    sweep_controls = dict(DEFAULT_SWEEP_CONTROLS)
    sweep_controls["epochs"] = args.epochs
    sweep_controls["early_stop_patience"] = args.early_stop_patience
    if args.smoke:
        sweep_controls["epochs"] = 2
        sweep_controls["early_stop_patience"] = 1
        sweep_controls["joint_refine_epochs"] = 2
        sweep_controls["joint_refine_early_stop_patience"] = 1

    base_config = base.read_yaml(config_path)
    resolved_base_checkpoints = oldbase.resolve_base_checkpoints(smoke=False)
    base_checkpoints = [
        checkpoint for checkpoint in resolved_base_checkpoints if int(checkpoint["seed"]) in selected_base_seeds
    ]
    if args.smoke:
        base_checkpoints = base_checkpoints[:1]
    if len(base_checkpoints) == 0:
        raise ValueError(f"No old-base checkpoints matched the requested seeds: {selected_base_seeds}")
    base.write_manifest(
        sweep_root / "manifest.json",
        build_manifest(
            safe_sweep_name=safe_sweep_name,
            config_path=config_path,
            sweep_controls=sweep_controls,
            base_checkpoints=base_checkpoints,
            selected_base_seeds=selected_base_seeds,
        ),
    )

    stage_a_trials = build_stage_a_trials()
    if args.max_stage_a_configs is not None:
        stage_a_trials = stage_a_trials[: args.max_stage_a_configs]
    if args.smoke:
        stage_a_trials = stage_a_trials[:1]

    success_rows = base.read_csv_rows(sweep_root / "trial_results.csv")
    failure_rows = base.read_csv_rows(sweep_root / "failures.csv")

    print(f"Sweep root: {sweep_root}")
    print(f"Base checkpoints in use: {len(base_checkpoints)}")
    print(f"Stage A configs: {len(stage_a_trials)}")
    print(f"Stage A planned runs: {len(stage_a_trials) * len(base_checkpoints)}")

    success_rows, failure_rows = base.run_trial_set(
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
    aggregate_rows = base.write_sweep_outputs(
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

    completed_joint_stage_a = [
        row for row in completed_stage_a if row["tail_schedule"] in {"static_then_joint", "cls_then_gpd_light_cls_then_joint"}
    ]
    if len(completed_joint_stage_a) < 2:
        print("Not enough completed joint Stage A configs to start Stage B.")
        return

    top_joint_stage_a = completed_joint_stage_a[:2]
    stage_b_trials = build_stage_b_trials(top_joint_stage_a)
    if args.max_stage_b_configs is not None:
        stage_b_trials = stage_b_trials[: args.max_stage_b_configs]
    print(f"Stage B configs: {len(stage_b_trials)}")
    print(f"Stage B planned runs: {len(stage_b_trials) * len(base_checkpoints)}")

    success_rows, failure_rows = base.run_trial_set(
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
    base.write_sweep_outputs(
        sweep_root=sweep_root,
        success_rows=success_rows,
        failure_rows=failure_rows,
        expected_base_count=len(base_checkpoints),
        sweep_controls=sweep_controls,
    )


if __name__ == "__main__":
    main()

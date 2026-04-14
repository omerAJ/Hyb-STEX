import argparse
from pathlib import Path

import nyctaxi_tail_v2_5seed as base


DEFAULT_SWEEP_NAME = "nyctaxi_tail_oldbase_focus_5seed"
OLD_BASE_CHECKPOINTS = [
    {
        "base_id": "seed1",
        "seed": 1,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=1\20260410-103756\fix phase-wise training evs_90\best_model_pred.pth"
        ),
        "source": "fix_phasewise_evs90",
    },
    {
        "base_id": "seed2",
        "seed": 2,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=2\20260410-110138\fix phase-wise training evs_90\best_model_pred.pth"
        ),
        "source": "fix_phasewise_evs90",
    },
    {
        "base_id": "seed3",
        "seed": 3,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=3\20260410-111950\fix phase-wise training evs_90\best_model_pred.pth"
        ),
        "source": "fix_phasewise_evs90",
    },
    {
        "base_id": "seed4",
        "seed": 4,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=4\20260410-113846\fix phase-wise training evs_90\best_model_pred.pth"
        ),
        "source": "fix_phasewise_evs90",
    },
    {
        "base_id": "seed5",
        "seed": 5,
        "checkpoint_path": Path(
            r"D:\omer\Hyb-STEX\experiments\NYCTaxi\pred__seed=5\20260410-120749\fix phase-wise training evs_90\best_model_pred.pth"
        ),
        "source": "fix_phasewise_evs90",
    },
]
DEFAULT_SWEEP_CONTROLS = dict(base.DEFAULT_SWEEP_CONTROLS)
DEFAULT_SWEEP_CONTROLS.update(
    {
        "training_recipe": "tail_only",
        "epochs": 1000,
        "early_stop": True,
        "early_stop_patience": 35,
        "tail_threshold_q": 0.95,
        "lr_init": 1.0e-3,
        "tail_schedule": "static",
        "tail_classifier_loss_type": "bce",
        "tail_pos_weight_multiplier": 1.0,
        "tail_focal_gamma": 2.0,
        "tail_mae_weight": 0.0,
        "tail_xi_min": -0.5,
        "tail_xi_max": -0.02,
    }
)
STAGE_A_SCHEDULES = ["static", "cls_then_gpd_light_cls"]
STAGE_A_Q_VALUES = [0.93, 0.95, 0.97]
STAGE_A_TAIL_MAE_WEIGHTS = [0.05, 0.10, 0.15]
STAGE_B_XI_PRESETS = [
    {"xi_label": "xi_n0p7_to_n0p05", "tail_xi_min": -0.7, "tail_xi_max": -0.05},
    {"xi_label": "xi_n0p5_to_n0p02", "tail_xi_min": -0.5, "tail_xi_max": -0.02},
    {"xi_label": "xi_n0p4_to_n0p015", "tail_xi_min": -0.4, "tail_xi_max": -0.015},
    {"xi_label": "xi_n0p3_to_n0p01", "tail_xi_min": -0.3, "tail_xi_max": -0.01},
]


def build_stage_a_trials():
    trials = []
    for schedule in STAGE_A_SCHEDULES:
        for tail_threshold_q in STAGE_A_Q_VALUES:
            for tail_mae_weight in STAGE_A_TAIL_MAE_WEIGHTS:
                run_id = f"a{len(trials) + 1:02d}"
                trial_id = "__".join(
                    [
                        "stageA",
                        f"sched_{schedule}",
                        f"q_{base.tokenize_number(tail_threshold_q)}",
                        f"mae_{base.tokenize_number(tail_mae_weight)}",
                    ]
                )
                overrides = {
                    "training_recipe": "tail_only",
                    "tail_schedule": schedule,
                    "tail_classifier_loss_type": "bce",
                    "tail_pos_weight_multiplier": 1.0,
                    "tail_focal_gamma": 2.0,
                    "tail_threshold_q": tail_threshold_q,
                    "tail_mae_weight": tail_mae_weight,
                    "tail_xi_min": -0.5,
                    "tail_xi_max": -0.02,
                }
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
            "tail_classifier_loss_type": "bce",
            "tail_pos_weight_multiplier": 1.0,
            "tail_focal_gamma": 2.0,
            "tail_threshold_q": float(parent_row["tail_threshold_q"]),
            "tail_mae_weight": float(parent_row["tail_mae_weight"]),
        }
        for xi_preset in STAGE_B_XI_PRESETS:
            run_id = f"b{len(trials) + 1:02d}"
            trial_id = "__".join(
                [
                    "stageB",
                    f"top{parent_index}",
                    base.sanitize_name(parent_row["trial_id"]),
                    xi_preset["xi_label"],
                ]
            )
            overrides = dict(parent_overrides)
            overrides.update(
                {
                    "tail_xi_min": xi_preset["tail_xi_min"],
                    "tail_xi_max": xi_preset["tail_xi_max"],
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


def resolve_base_checkpoints(smoke):
    selected_checkpoints = OLD_BASE_CHECKPOINTS[:1] if smoke else OLD_BASE_CHECKPOINTS
    resolved = []
    for base_info in selected_checkpoints:
        if not base_info["checkpoint_path"].is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {base_info['checkpoint_path']}")
        resolved.append(
            {
                "base_id": base_info["base_id"],
                "seed": base_info["seed"],
                "checkpoint_path": str(base_info["checkpoint_path"]),
                "source": base_info["source"],
                "comment": "",
                "experiment_dir": str(base_info["checkpoint_path"].parent),
            }
        )
    return resolved


def build_manifest(safe_sweep_name, config_path, sweep_controls, base_checkpoints):
    return {
        "sweep_name": safe_sweep_name,
        "created_at": base.datetime.now().isoformat(),
        "dataset": "NYCTaxi",
        "base_checkpoint_source": "fix phase-wise training evs_90",
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
            "description": "Search the knobs that have moved MAE the most so far on the stronger old-base predictors.",
            "schedules": STAGE_A_SCHEDULES,
            "tail_threshold_q_values": STAGE_A_Q_VALUES,
            "tail_mae_weights": STAGE_A_TAIL_MAE_WEIGHTS,
            "classifier_loss_type": "bce",
            "fixed_tail_xi_min": -0.5,
            "fixed_tail_xi_max": -0.02,
        },
        "stage_b": {
            "top_stage_a_configs": 2,
            "xi_presets": STAGE_B_XI_PRESETS,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME, help="Name of the sweep output directory.")
    parser.add_argument("--python", default=base.sys.executable, help="Python executable to use.")
    parser.add_argument("--config", default=str(base.DEFAULT_CONFIG_PATH), help="Base NYCTaxi config path.")
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

    base_config = base.read_yaml(config_path)
    base_checkpoints = resolve_base_checkpoints(smoke=args.smoke)
    manifest_path = sweep_root / "manifest.json"
    base.write_manifest(
        manifest_path,
        build_manifest(
            safe_sweep_name=safe_sweep_name,
            config_path=config_path,
            sweep_controls=sweep_controls,
            base_checkpoints=base_checkpoints,
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

    top_stage_a_rows = completed_stage_a[:2]
    stage_b_trials = build_stage_b_trials(top_stage_a_rows)
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

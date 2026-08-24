import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import model_supervisor
from scripts.run_nyctaxi_event_gate_ablation import (
    BASE_VARIANT,
    FLOW_NAMES,
    add_metric_rows,
    apply_common_overrides,
    build_phase1_args,
    load_base_config,
    print_table,
    repo_path,
    resolve_data_dir,
    resolve_graph_file,
    summarize,
    write_csv,
)


VARIANT_CONFIGS = {
    "ungated_head_only": {
        "ablation_mode": "ungated_bias",
        "bias_param_scope": "head_only",
        "start_phase": "bias",
        "uses_event_loss": False,
    },
    "boosted_gate": {
        "ablation_mode": "boosted_gate",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
        "uses_event_loss": False,
    },
    "event_weighted_boosted_gate": {
        "ablation_mode": "event_weighted_boosted_gate",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
        "uses_event_loss": True,
    },
    "event_weighted_dual_residual": {
        "ablation_mode": "event_weighted_dual_residual",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
        "uses_event_loss": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run NYCTaxi boosted classifier-gate residual ablations. The boosted gate "
            "uses prediction = base + residual * (1 + boost_scale * p_event), so "
            "the classifier can amplify but not suppress the ungated residual."
        )
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--output-dir", default="nyctaxi_boosted_gate_ablation_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted([BASE_VARIANT, *VARIANT_CONFIGS]),
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--gate-floor", type=float, default=0.5)
    parser.add_argument("--boost-gate-scale", type=float, default=1.0)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="nyctaxi_boosted_gate_ablation")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one seed with one epoch and one train/eval batch per phase.",
    )
    return parser.parse_args()


def build_variant_args(base_configs, args, seed, variant, phase1_checkpoint, smoke_test):
    run_configs = dict(base_configs)
    data_dir = resolve_data_dir(run_configs, args.data_dir)
    graph_file = resolve_graph_file(run_configs, data_dir, args.graph_file)
    variant_config = dict(VARIANT_CONFIGS[variant])
    event_loss_weight = (
        args.event_loss_weight if variant_config.pop("uses_event_loss") else 0.0
    )

    run_configs.update(
        {
            "mode": "train",
            "seed": seed,
            "data_dir": data_dir,
            "graph_file": graph_file,
            "phase3_mode": "original",
            "gate_floor": args.gate_floor,
            "boost_gate_scale": args.boost_gate_scale,
            "classification_loss_weight": args.classification_loss_weight,
            "event_loss_weight": event_loss_weight,
            "load_path": str(phase1_checkpoint),
            "stop_after_phase": "pred_2",
            "comment": args.comment,
            "experimentName": f"nyctaxi_boosted_gate_{variant}_seed={seed}",
        }
    )
    run_configs.update(variant_config)
    apply_common_overrides(run_configs, args, smoke_test)
    return SimpleNamespace(**run_configs)


def main():
    args = parse_args()
    base_configs = load_base_config(args.config_filename)
    seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    variants = args.variants or [
        BASE_VARIANT,
        "ungated_head_only",
        "boosted_gate",
        "event_weighted_boosted_gate",
        "event_weighted_dual_residual",
    ]

    per_seed_rows = []
    phase1_rows = []
    for seed in seeds:
        phase1_args = build_phase1_args(base_configs, args, seed, args.smoke_test)
        print(f"\n=== shared_phase1 seed={seed} ===")
        phase1_result = model_supervisor(phase1_args)
        if phase1_result is None or "test_results" not in phase1_result:
            raise RuntimeError(f"Shared phase1 failed for seed={seed}")

        phase1_log_dir = Path(phase1_args.log_dir)
        phase1_checkpoint = phase1_log_dir / "best_model_pred.pth"
        if not phase1_checkpoint.is_file():
            raise FileNotFoundError(f"Shared phase1 checkpoint not found: {phase1_checkpoint}")
        add_metric_rows(
            phase1_rows,
            "shared_phase1",
            seed,
            phase1_result["test_results"],
            str(phase1_log_dir),
            str(phase1_checkpoint),
        )
        if BASE_VARIANT in variants:
            add_metric_rows(
                per_seed_rows,
                BASE_VARIANT,
                seed,
                phase1_result["test_results"],
                str(phase1_log_dir),
                str(phase1_checkpoint),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for variant in variants:
            if variant == BASE_VARIANT:
                continue
            run_args = build_variant_args(
                base_configs,
                args,
                seed,
                variant,
                phase1_checkpoint,
                args.smoke_test,
            )
            print(f"\n=== variant={variant} seed={seed} ===")
            result = model_supervisor(run_args)
            if result is None or "test_results" not in result:
                raise RuntimeError(f"Run failed for variant={variant}, seed={seed}")
            add_metric_rows(
                per_seed_rows,
                variant,
                seed,
                result["test_results"],
                str(Path(run_args.log_dir)),
                str(Path(run_args.log_dir) / "best_model_pred_2.pth"),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_dir = repo_path(args.output_dir)
    phase1_summary_rows = summarize(phase1_rows, ["shared_phase1"])
    summary_rows = summarize(per_seed_rows, variants)
    write_csv(output_dir / "phase1_metrics.csv", phase1_rows)
    write_csv(output_dir / "phase1_summary_metrics.csv", phase1_summary_rows)
    write_csv(output_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)

    print_table("Shared Phase-1 Metrics", phase1_rows)
    print_table("Per-seed Metrics", per_seed_rows)
    print_table("Averages", summary_rows)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()

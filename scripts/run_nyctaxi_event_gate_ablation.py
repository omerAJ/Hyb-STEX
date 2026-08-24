import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import model_supervisor


DEFAULTS = {
    "S_Loss": 0,
    "T_Loss": 0,
    "cheb_order": 3,
    "graph_init": "8_neighbours",
    "self_attention_flag": True,
    "cross_attention_flag": False,
    "feedforward_flag": False,
    "layer_norm_flag": False,
    "additional_sa_flag": False,
    "learnable_flag": False,
    "rank": 0,
    "pos_emb_flag": False,
    "add_8": False,
    "add_eye": False,
    "add_x_encoder": False,
    "freeze_encoder": False,
    "threshold_adj_mx": False,
    "affinity_conv": False,
    "loss": "mae",
    "load_path": None,
    "variant": None,
    "phase3_mode": "original",
    "ablation_mode": "original",
    "bias_param_scope": "head_only",
    "gate_floor": 0.5,
    "classification_loss_weight": 1.0,
    "event_loss_weight": 0.0,
    "start_phase": "pred",
    "stop_after_phase": "pred_2",
    "max_train_batches": None,
    "max_eval_batches": None,
}

FLOW_NAMES = ("inflow", "outflow")
BASE_VARIANT = "base_model"

VARIANT_CONFIGS = {
    "ungated_head_only": {
        "ablation_mode": "ungated_bias",
        "bias_param_scope": "head_only",
        "start_phase": "bias",
    },
    "original_head_only": {
        "ablation_mode": "original",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
    },
    "floor_gated": {
        "ablation_mode": "floor_gated",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
    },
    "dual_event_residual": {
        "ablation_mode": "dual_event_residual",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
    },
    "event_weighted_original": {
        "ablation_mode": "event_weighted_original",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
    },
    "event_weighted_dual_residual": {
        "ablation_mode": "event_weighted_dual_residual",
        "bias_param_scope": "head_only",
        "start_phase": "cls",
    },
    "ungated_legacy_bias_group": {
        "ablation_mode": "ungated_bias",
        "bias_param_scope": "legacy",
        "start_phase": "bias",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a shared-phase1 NYCTaxi event-gate ablation. The improved variants "
            "test residual gates that preserve the useful always-on correction path."
        )
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--output-dir", default="nyctaxi_event_gate_ablation_results")
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
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="nyctaxi_event_gate_ablation")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one seed with one epoch and one train/eval batch per phase.",
    )
    return parser.parse_args()


def repo_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def load_base_config(config_filename):
    config_path = repo_path(config_filename)
    with config_path.open("r") as handle:
        configs = yaml.load(handle, Loader=yaml.FullLoader)
    merged = dict(DEFAULTS)
    merged.update(configs)
    return merged


def resolve_data_dir(configs, cli_data_dir):
    dataset = configs["dataset"]
    data_dir = cli_data_dir or configs["data_dir"]
    if repo_path(data_dir).joinpath(dataset).is_dir():
        return data_dir
    fallback = "preprocessed_data"
    if repo_path(fallback).joinpath(dataset).is_dir():
        return fallback
    return data_dir


def resolve_graph_file(configs, data_dir, cli_graph_file):
    graph_file = cli_graph_file or configs["graph_file"]
    if repo_path(graph_file).is_file():
        return graph_file
    candidate = repo_path(data_dir).joinpath(configs["dataset"], "adj_mx.npz")
    if candidate.is_file():
        return str(candidate.relative_to(ROOT))
    return graph_file


def apply_common_overrides(run_configs, args, smoke_test):
    if args.device is not None:
        run_configs["device"] = args.device
    if args.batch_size is not None:
        run_configs["batch_size"] = args.batch_size
    if args.test_batch_size is not None:
        run_configs["test_batch_size"] = args.test_batch_size

    if args.epochs is not None:
        run_configs["epochs"] = args.epochs
        run_configs["num_epochs"] = args.num_epochs or args.epochs
    elif smoke_test:
        run_configs["epochs"] = 1
        run_configs["num_epochs"] = args.num_epochs or 1
    elif args.num_epochs is not None:
        run_configs["num_epochs"] = args.num_epochs

    if args.max_train_batches is not None:
        run_configs["max_train_batches"] = args.max_train_batches
    elif smoke_test:
        run_configs["max_train_batches"] = 1

    if args.max_eval_batches is not None:
        run_configs["max_eval_batches"] = args.max_eval_batches
    elif smoke_test:
        run_configs["max_eval_batches"] = 1


def build_phase1_args(base_configs, args, seed, smoke_test):
    run_configs = dict(base_configs)
    data_dir = resolve_data_dir(run_configs, args.data_dir)
    graph_file = resolve_graph_file(run_configs, data_dir, args.graph_file)

    run_configs.update(
        {
            "mode": "train",
            "seed": seed,
            "data_dir": data_dir,
            "graph_file": graph_file,
            "ablation_mode": "original",
            "phase3_mode": "original",
            "bias_param_scope": "head_only",
            "gate_floor": args.gate_floor,
            "classification_loss_weight": args.classification_loss_weight,
            "event_loss_weight": args.event_loss_weight,
            "load_path": None,
            "start_phase": "pred",
            "stop_after_phase": "pred",
            "comment": args.comment,
            "experimentName": f"nyctaxi_event_gate_shared_phase1_seed={seed}",
        }
    )
    apply_common_overrides(run_configs, args, smoke_test)
    return SimpleNamespace(**run_configs)


def build_variant_args(base_configs, args, seed, variant, phase1_checkpoint, smoke_test):
    run_configs = dict(base_configs)
    data_dir = resolve_data_dir(run_configs, args.data_dir)
    graph_file = resolve_graph_file(run_configs, data_dir, args.graph_file)
    variant_config = dict(VARIANT_CONFIGS[variant])

    run_configs.update(
        {
            "mode": "train",
            "seed": seed,
            "data_dir": data_dir,
            "graph_file": graph_file,
            "phase3_mode": "original",
            "gate_floor": args.gate_floor,
            "classification_loss_weight": args.classification_loss_weight,
            "event_loss_weight": args.event_loss_weight,
            "load_path": str(phase1_checkpoint),
            "stop_after_phase": "pred_2",
            "comment": args.comment,
            "experimentName": f"nyctaxi_event_gate_{variant}_seed={seed}",
        }
    )
    run_configs.update(variant_config)
    apply_common_overrides(run_configs, args, smoke_test)
    return SimpleNamespace(**run_configs)


def add_metric_rows(rows, variant, seed, test_results, log_dir, checkpoint):
    metrics = np.asarray(test_results, dtype=float)
    if metrics.shape != (2, 2):
        raise ValueError(f"Expected test_results shape (2, 2), got {metrics.shape}")

    for flow_idx, flow in enumerate(FLOW_NAMES):
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "flow": flow,
                "mae": metrics[flow_idx, 0],
                "eee": metrics[flow_idx, 1],
                "log_dir": log_dir,
                "checkpoint": checkpoint,
            }
        )

    rows.append(
        {
            "variant": variant,
            "seed": seed,
            "flow": "mean",
            "mae": float(np.nanmean(metrics[:, 0])),
            "eee": float(np.nanmean(metrics[:, 1])),
            "log_dir": log_dir,
            "checkpoint": checkpoint,
        }
    )


def summarize(rows, variants):
    summary = []
    for variant in variants:
        for flow in (*FLOW_NAMES, "mean"):
            selected = [
                row for row in rows
                if row["variant"] == variant and row["flow"] == flow
            ]
            if not selected:
                continue
            summary.append(
                {
                    "variant": variant,
                    "seed": "average",
                    "flow": flow,
                    "mae": float(np.nanmean([row["mae"] for row in selected])),
                    "eee": float(np.nanmean([row["eee"] for row in selected])),
                    "log_dir": "",
                    "checkpoint": "",
                }
            )
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "seed", "flow", "mae", "eee", "log_dir", "checkpoint"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(title, rows):
    print(f"\n{title}")
    headers = ["variant", "seed", "flow", "mae", "eee"]
    widths = {
        header: max(len(header), *(len(format_value(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(format_value(row[header]).ljust(widths[header]) for header in headers))


def main():
    args = parse_args()
    base_configs = load_base_config(args.config_filename)
    seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    variants = args.variants or [
        BASE_VARIANT,
        "ungated_head_only",
        "event_weighted_original",
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

    summary_rows = summarize(per_seed_rows, variants)
    phase1_summary_rows = summarize(phase1_rows, ["shared_phase1"])
    output_dir = repo_path(args.output_dir)
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

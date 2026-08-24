import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import model_supervisor
from scripts.run_nyctaxi_event_gate_ablation import (
    DEFAULTS,
    FLOW_NAMES,
    add_metric_rows,
    apply_common_overrides,
    format_value,
    load_base_config,
    print_table,
    repo_path,
    resolve_data_dir,
    resolve_graph_file,
    summarize,
    write_csv,
)


TRAINED_VARIANTS = {
    "expanded_base_additive": {
        "ablation_mode": "expanded_base_additive",
        "load_phase1": False,
        "start_phase": "pred",
        "stop_after_phase": "pred",
        "checkpoint_name": "best_model_pred.pth",
    },
    "node_bias_frozen_base": {
        "ablation_mode": "node_bias_only",
        "load_phase1": True,
        "start_phase": "pred_2",
        "stop_after_phase": "pred_2",
        "checkpoint_name": "best_model_pred_2.pth",
    },
    "flow_bias_frozen_base": {
        "ablation_mode": "flow_bias_only",
        "load_phase1": True,
        "start_phase": "pred_2",
        "stop_after_phase": "pred_2",
        "checkpoint_name": "best_model_pred_2.pth",
    },
}

REFERENCE_VARIANTS = {
    "base_model": ("nyctaxi_base_and_event_weighted_results", "base_model"),
    "ungated_residual_frozen_base": (
        "nyctaxi_frozen_ungated_residual_results",
        "ungated_residual_frozen_base",
    ),
    "ungated_head_only": ("nyctaxi_event_gate_ablation_results", "ungated_head_only"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run NYCTaxi controls that test whether ungated residual gains are "
            "explained by extra parameters alone."
        )
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--phase1-results", default="nyctaxi_base_and_event_weighted_results")
    parser.add_argument("--output-dir", default="nyctaxi_capacity_controls_results")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted([*TRAINED_VARIANTS, *REFERENCE_VARIANTS]),
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="nyctaxi_capacity_controls")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run seed 1 with one epoch and one train/eval batch.",
    )
    return parser.parse_args()


def read_csv(path):
    path = repo_path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def load_reference_rows(reference_name, source_dir, source_variant, seeds):
    rows = []
    for row in read_csv(Path(source_dir) / "per_seed_metrics.csv"):
        if row["variant"] != source_variant:
            continue
        seed = int(float(row["seed"]))
        if seed not in seeds:
            continue
        rows.append(
            {
                "variant": reference_name,
                "seed": seed,
                "flow": row["flow"],
                "mae": float(row["mae"]),
                "eee": float(row["eee"]),
                "log_dir": row.get("log_dir", ""),
                "checkpoint": row.get("checkpoint", ""),
            }
        )
    return rows


def phase1_checkpoints(phase1_results):
    lookup = {}
    for row in read_csv(Path(phase1_results) / "per_seed_metrics.csv"):
        if row["variant"] == "base_model" and row["flow"] == "mean":
            lookup[int(float(row["seed"]))] = row["checkpoint"]
    return lookup


def build_run_args(base_configs, args, seed, variant, phase1_checkpoint):
    variant_config = TRAINED_VARIANTS[variant]
    run_configs = dict(DEFAULTS)
    run_configs.update(base_configs)
    data_dir = resolve_data_dir(run_configs, args.data_dir)
    graph_file = resolve_graph_file(run_configs, data_dir, args.graph_file)
    run_configs.update(
        {
            "mode": "train",
            "seed": seed,
            "data_dir": data_dir,
            "graph_file": graph_file,
            "ablation_mode": variant_config["ablation_mode"],
            "phase3_mode": "original",
            "bias_param_scope": "head_only",
            "classification_loss_weight": 1.0,
            "event_loss_weight": 0.0,
            "load_path": phase1_checkpoint if variant_config["load_phase1"] else None,
            "start_phase": variant_config["start_phase"],
            "stop_after_phase": variant_config["stop_after_phase"],
            "comment": args.comment,
            "experimentName": f"nyctaxi_capacity_{variant}_seed={seed}",
        }
    )
    apply_common_overrides(run_configs, args, args.smoke_test)
    return SimpleNamespace(**run_configs)


def run_variant(base_configs, args, variant, seeds, checkpoints):
    rows = []
    variant_config = TRAINED_VARIANTS[variant]
    if variant_config["load_phase1"]:
        missing = [seed for seed in seeds if seed not in checkpoints]
        if missing:
            raise FileNotFoundError(
                f"Missing base_model phase-1 checkpoints for seeds: {missing}"
            )
    for seed in seeds:
        checkpoint = checkpoints.get(seed)
        run_args = build_run_args(base_configs, args, seed, variant, checkpoint)
        print(f"\n=== variant={variant} seed={seed} ===")
        result = model_supervisor(run_args)
        if result is None or "test_results" not in result:
            raise RuntimeError(f"Run failed for variant={variant}, seed={seed}")
        log_dir = Path(run_args.log_dir)
        add_metric_rows(
            rows,
            variant,
            seed,
            result["test_results"],
            str(log_dir),
            str(log_dir / variant_config["checkpoint_name"]),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    args = parse_args()
    seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    variants = args.variants or [
        "base_model",
        "expanded_base_additive",
        "flow_bias_frozen_base",
        "node_bias_frozen_base",
        "ungated_residual_frozen_base",
        "ungated_head_only",
    ]
    base_configs = load_base_config(args.config_filename)
    checkpoints = phase1_checkpoints(args.phase1_results)

    per_seed_rows = []
    for variant in variants:
        if variant in REFERENCE_VARIANTS:
            source_dir, source_variant = REFERENCE_VARIANTS[variant]
            per_seed_rows.extend(
                load_reference_rows(variant, source_dir, source_variant, seeds)
            )
            continue
        per_seed_rows.extend(run_variant(base_configs, args, variant, seeds, checkpoints))

    summary_rows = summarize(per_seed_rows, variants)
    output_dir = repo_path(args.output_dir)
    write_csv(output_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    print_table("Per-seed Metrics", per_seed_rows)
    print_table("Averages", summary_rows)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()

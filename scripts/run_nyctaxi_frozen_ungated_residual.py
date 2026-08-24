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


VARIANT = "ungated_residual_frozen_base"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train an ungated residual correction with the phase-1 base frozen. "
            "This isolates whether the residual head alone explains the base-to-ungated gain."
        )
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--phase1-results", default="nyctaxi_base_and_event_weighted_results")
    parser.add_argument("--output-dir", default="nyctaxi_frozen_ungated_residual_results")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="nyctaxi_frozen_ungated_residual")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one seed with one epoch and one train/eval batch.",
    )
    return parser.parse_args()


def read_csv(path):
    with repo_path(path).open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def phase1_checkpoints(phase1_results):
    rows = read_csv(Path(phase1_results) / "per_seed_metrics.csv")
    lookup = {}
    for row in rows:
        if row["variant"] == "base_model" and row["flow"] == "mean":
            lookup[int(float(row["seed"]))] = row["checkpoint"]
    return lookup


def build_run_args(base_configs, args, seed, checkpoint):
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
            "ablation_mode": "ungated_bias",
            "phase3_mode": "original",
            "bias_param_scope": "head_only",
            "classification_loss_weight": 1.0,
            "event_loss_weight": 0.0,
            "load_path": checkpoint,
            "start_phase": "pred_2",
            "stop_after_phase": "pred_2",
            "comment": args.comment,
            "experimentName": f"nyctaxi_{VARIANT}_seed={seed}",
        }
    )
    apply_common_overrides(run_configs, args, args.smoke_test)
    return SimpleNamespace(**run_configs)


def main():
    args = parse_args()
    if args.smoke_test:
        args.seeds = args.seeds[:1]
    base_configs = load_base_config(args.config_filename)
    checkpoints = phase1_checkpoints(args.phase1_results)
    missing = [seed for seed in args.seeds if seed not in checkpoints]
    if missing:
        raise FileNotFoundError(
            f"Missing base_model phase-1 checkpoints for seeds: {missing}"
        )

    rows = []
    for seed in args.seeds:
        run_args = build_run_args(base_configs, args, seed, checkpoints[seed])
        print(f"\n=== {VARIANT} seed={seed} ===")
        result = model_supervisor(run_args)
        if result is None or "test_results" not in result:
            raise RuntimeError(f"Run failed for seed={seed}")
        add_metric_rows(
            rows,
            VARIANT,
            seed,
            result["test_results"],
            str(Path(run_args.log_dir)),
            str(Path(run_args.log_dir) / "best_model_pred_2.pth"),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = summarize(rows, [VARIANT])
    output_dir = repo_path(args.output_dir)
    write_csv(output_dir / "per_seed_metrics.csv", rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    print_table("Frozen-Base Ungated Residual Metrics", rows)
    print_table("Averages", summary_rows)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()

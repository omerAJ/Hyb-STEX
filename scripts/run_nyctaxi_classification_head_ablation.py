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
    "max_train_batches": None,
    "max_eval_batches": None,
}

FLOW_NAMES = ("inflow", "outflow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NYCTaxi original vs ungated-bias classification-head ablation."
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--output-dir", default="nyctaxi_classification_head_ablation_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["original", "ungated_bias"],
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="nyctaxi_classification_head_ablation")
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


def build_run_args(base_configs, args, seed, variant, smoke_test):
    run_configs = dict(base_configs)
    data_dir = resolve_data_dir(run_configs, args.data_dir)
    graph_file = resolve_graph_file(run_configs, data_dir, args.graph_file)

    run_configs["mode"] = "train"
    run_configs["seed"] = seed
    run_configs["data_dir"] = data_dir
    run_configs["graph_file"] = graph_file
    run_configs["ablation_mode"] = variant
    run_configs["phase3_mode"] = "original"
    run_configs["load_path"] = None
    run_configs["comment"] = args.comment
    run_configs["experimentName"] = f"nyctaxi_cls_head_ablation_{variant}_seed={seed}"

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

    return SimpleNamespace(**run_configs)


def add_metric_rows(rows, variant, seed, test_results):
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
            }
        )

    rows.append(
        {
            "variant": variant,
            "seed": seed,
            "flow": "mean",
            "mae": float(np.nanmean(metrics[:, 0])),
            "eee": float(np.nanmean(metrics[:, 1])),
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
                }
            )
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "seed", "flow", "mae", "eee"])
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
    variants = args.variants or ["original", "ungated_bias"]

    per_seed_rows = []
    for variant in variants:
        for seed in seeds:
            run_args = build_run_args(base_configs, args, seed, variant, args.smoke_test)
            print(f"\n=== variant={variant} seed={seed} ===")
            result = model_supervisor(run_args)
            if result is None or "test_results" not in result:
                raise RuntimeError(f"Run failed for variant={variant}, seed={seed}")
            add_metric_rows(per_seed_rows, variant, seed, result["test_results"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_rows = summarize(per_seed_rows, variants)
    output_dir = repo_path(args.output_dir)
    write_csv(output_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)

    print_table("Per-seed metrics", per_seed_rows)
    print_table("Averages", summary_rows)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()

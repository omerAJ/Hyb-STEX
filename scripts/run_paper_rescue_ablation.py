import argparse
import csv
import hashlib
import json
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
    "variant": None,
    "phase3_mode": "original",
    "bias_param_scope": "head_only",
    "classification_loss_weight": 1.0,
    "event_loss_weight": 0.0,
    "scaler_fit": "train",
    "max_train_batches": None,
    "max_eval_batches": None,
}

FLOW_NAMES = ("inflow", "outflow")
VARIANTS = {
    "A_base_mae": {
        "source": "shared_phase1",
        "description": "base predictor trained with ordinary MAE",
    },
    "B_base_event_weighted": {
        "source": "independent",
        "ablation_mode": "event_weighted_base",
        "start_phase": "pred",
        "stop_after_phase": "pred",
        "event_weighted": True,
        "description": "base predictor trained with MAE plus event MAE",
    },
    "C_ungated_residual_mae": {
        "source": "shared_phase1",
        "ablation_mode": "ungated_bias",
        "start_phase": "bias",
        "stop_after_phase": "pred_2",
        "event_weighted": False,
        "description": "base plus one always-on residual, trained with MAE",
    },
    "D_ungated_residual_event_weighted": {
        "source": "shared_phase1",
        "ablation_mode": "event_weighted_ungated_residual",
        "start_phase": "bias",
        "stop_after_phase": "pred_2",
        "event_weighted": True,
        "description": "base plus one always-on residual, trained with event-weighted MAE",
    },
    "E_classifier_gated_residual_mae": {
        "source": "shared_phase1",
        "ablation_mode": "original",
        "start_phase": "cls",
        "stop_after_phase": "pred_2",
        "event_weighted": False,
        "description": "original classifier-gated residual trained with MAE and BCE",
    },
    "F_dual_residual_event_weighted": {
        "source": "shared_phase1",
        "ablation_mode": "event_weighted_dual_residual",
        "start_phase": "cls",
        "stop_after_phase": "pred_2",
        "event_weighted": True,
        "description": "always-on general residual plus classifier-gated event residual",
    },
    "G_dual_ungated_event_weighted": {
        "source": "shared_phase1",
        "ablation_mode": "event_weighted_dual_ungated",
        "start_phase": "bias",
        "stop_after_phase": "pred_2",
        "event_weighted": True,
        "description": "same two residual heads as F, both always on and no classifier",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the A-G paper-rescue ablation with fair shared phase-1 checkpoints."
    )
    parser.add_argument(
        "--config-filenames",
        nargs="+",
        default=["configs/NYCTaxi.yaml"],
        help="One or more dataset YAML files.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None, help="Only valid with one config file.")
    parser.add_argument("--output-dir", default="paper_rescue_ablation_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--bias-param-scope",
        choices=("legacy", "head_only"),
        default="head_only",
        help="Residual parameter grouping; legacy reproduces the original Hyb-STEX checkpoint protocol.",
    )
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--scaler-fit",
        choices=("train", "train_val"),
        default="train",
        help="Dataset statistics source. Use train_val only for a legacy benchmark reproduction.",
    )
    parser.add_argument(
        "--event-mask-protocol",
        choices=("legacy_raw_p90_v1", "train_all_node_flow_p90_valid_v2"),
        default="legacy_raw_p90_v1",
    )
    parser.add_argument(
        "--event-label-source",
        choices=("legacy_file_unverified", "file_verified", "generated"),
        default="legacy_file_unverified",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="paper_rescue_ablation")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore saved progress in the output directory and start a new result set.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use seed 1, one epoch, and one train/evaluation batch per phase.",
    )
    return parser.parse_args()


def repo_path(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_config(filename):
    with repo_path(filename).open("r") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    merged = dict(DEFAULTS)
    merged.update(config)
    return merged


def resolve_paths(config, args):
    data_dir = args.data_dir or config["data_dir"]
    if not repo_path(data_dir).joinpath(config["dataset"]).is_dir():
        fallback = repo_path("preprocessed_data").joinpath(config["dataset"])
        if fallback.is_dir():
            data_dir = "preprocessed_data"

    graph_file = args.graph_file or config["graph_file"]
    if not repo_path(graph_file).is_file():
        candidate = repo_path(data_dir).joinpath(config["dataset"], "adj_mx.npz")
        if candidate.is_file():
            graph_file = str(candidate)
    return data_dir, graph_file


def apply_overrides(config, args):
    config["scaler_fit"] = args.scaler_fit
    config["event_mask_protocol"] = args.event_mask_protocol
    config["event_label_source"] = args.event_label_source
    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.test_batch_size is not None:
        config["test_batch_size"] = args.test_batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
        config["num_epochs"] = args.num_epochs or args.epochs
    elif args.smoke_test:
        config["epochs"] = 1
        config["num_epochs"] = args.num_epochs or 1
    elif args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs

    config["max_train_batches"] = (
        args.max_train_batches if args.max_train_batches is not None
        else (1 if args.smoke_test else None)
    )
    config["max_eval_batches"] = (
        args.max_eval_batches if args.max_eval_batches is not None
        else (1 if args.smoke_test else None)
    )


def common_run_config(base, args, seed, experiment_name):
    config = dict(base)
    data_dir, graph_file = resolve_paths(config, args)
    config.update({
        "mode": "train",
        "seed": seed,
        "data_dir": data_dir,
        "graph_file": graph_file,
        "phase3_mode": "original",
        "bias_param_scope": args.bias_param_scope,
        "classification_loss_weight": args.classification_loss_weight,
        "comment": args.comment,
        "experimentName": experiment_name,
    })
    apply_overrides(config, args)
    return config


def build_shared_phase1_args(base, args, seed):
    dataset = base["dataset"]
    config = common_run_config(
        base, args, seed, f"paper_rescue_{dataset}_shared_phase1_seed={seed}"
    )
    config.update({
        "ablation_mode": "original",
        "event_loss_weight": 0.0,
        "load_path": None,
        "start_phase": "pred",
        "stop_after_phase": "pred",
    })
    return SimpleNamespace(**config)


def build_variant_args(base, args, seed, variant, phase1_checkpoint):
    spec = VARIANTS[variant]
    dataset = base["dataset"]
    config = common_run_config(
        base, args, seed, f"paper_rescue_{dataset}_{variant}_seed={seed}"
    )
    config.update({
        "ablation_mode": spec["ablation_mode"],
        "event_loss_weight": args.event_loss_weight if spec["event_weighted"] else 0.0,
        "load_path": None if spec["source"] == "independent" else str(phase1_checkpoint),
        "start_phase": spec["start_phase"],
        "stop_after_phase": spec["stop_after_phase"],
    })
    return SimpleNamespace(**config)


def checkpoint_for(run_args):
    phase = "pred" if run_args.stop_after_phase == "pred" else "pred_2"
    return Path(run_args.log_dir) / f"best_model_{phase}.pth"


def add_rows(rows, dataset, variant, seed, result, log_dir, checkpoint):
    metrics = np.asarray(result["test_results"], dtype=float)
    details = result.get("test_details") or [{}, {}]
    parameter_counts = result.get("parameter_counts") or {}
    active_parameters = parameter_counts.get("active", float("nan"))
    flow_rows = []
    for idx, flow in enumerate(FLOW_NAMES):
        detail = details[idx] if idx < len(details) else {}
        row = {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "flow": flow,
            "mae": metrics[idx, 0],
            "eee": metrics[idx, 1],
            "normal_mae": detail.get("normal_mae", float("nan")),
            "event_signed_error": detail.get("event_signed_error", float("nan")),
            "event_count": detail.get("event_count", 0),
            "normal_count": detail.get("normal_count", 0),
            "active_parameters": active_parameters,
            "log_dir": str(log_dir),
            "checkpoint": str(checkpoint),
        }
        rows.append(row)
        flow_rows.append(row)

    rows.append({
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "flow": "mean",
        "mae": float(np.nanmean([row["mae"] for row in flow_rows])),
        "eee": float(np.nanmean([row["eee"] for row in flow_rows])),
        "normal_mae": float(np.nanmean([row["normal_mae"] for row in flow_rows])),
        "event_signed_error": float(np.nanmean([row["event_signed_error"] for row in flow_rows])),
        "event_count": sum(row["event_count"] for row in flow_rows),
        "normal_count": sum(row["normal_count"] for row in flow_rows),
        "active_parameters": active_parameters,
        "log_dir": str(log_dir),
        "checkpoint": str(checkpoint),
    })


def summarize(rows):
    summary = []
    metrics = ("mae", "eee", "normal_mae", "event_signed_error")
    keys = sorted({(row["dataset"], row["variant"], row["flow"]) for row in rows})
    for dataset, variant, flow in keys:
        selected = [
            row for row in rows
            if (row["dataset"], row["variant"], row["flow"]) == (dataset, variant, flow)
        ]
        output = {
            "dataset": dataset,
            "variant": variant,
            "flow": flow,
            "seeds": len(selected),
            "active_parameters": selected[0]["active_parameters"],
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            output[f"{metric}_mean"] = float(np.nanmean(values))
            output[f"{metric}_std"] = float(np.nanstd(values))
        summary.append(output)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_metric_rows(path):
    if not path.is_file():
        return []
    rows = []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            row["seed"] = int(row["seed"])
            for name in ("mae", "eee", "normal_mae", "event_signed_error", "active_parameters"):
                row[name] = float(row[name])
            for name in ("event_count", "normal_count"):
                row[name] = int(float(row[name]))
            rows.append(row)
    return rows


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(args, seeds, variants):
    config_files = [repo_path(name).resolve() for name in args.config_filenames]
    code_files = [
        Path(__file__).resolve(),
        ROOT / "main.py",
        ROOT / "model" / "models.py",
        ROOT / "model" / "trainer.py",
        ROOT / "lib" / "dataloader.py",
    ]
    return {
        "schema_version": 1,
        "configs": [{"path": str(path), "sha256": file_digest(path)} for path in config_files],
        "code": [{"path": str(path), "sha256": file_digest(path)} for path in code_files],
        "seeds": seeds,
        "variants": variants,
        "data_dir": args.data_dir,
        "graph_file": args.graph_file,
        "device": args.device,
        "epochs": args.epochs,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "classification_loss_weight": args.classification_loss_weight,
        "bias_param_scope": args.bias_param_scope,
        "event_loss_weight": args.event_loss_weight,
        "scaler_fit": args.scaler_fit,
        "event_mask_protocol": args.event_mask_protocol,
        "event_label_source": args.event_label_source,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "smoke_test": args.smoke_test,
    }


def prepare_resume(output_dir, manifest, restart):
    manifest_path = output_dir / "run_manifest.json"
    result_files = (
        output_dir / "per_seed_metrics.csv",
        output_dir / "summary_metrics.csv",
        output_dir / "shared_phase1_metrics.csv",
        output_dir / "shared_phase1_summary.csv",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if restart:
        for path in (*result_files, manifest_path):
            if path.is_file():
                path.unlink()

    if manifest_path.is_file():
        with manifest_path.open("r") as handle:
            saved = json.load(handle)
        if saved != manifest:
            raise ValueError(
                f"Saved progress in {output_dir} belongs to a different run. "
                "Use another --output-dir or pass --restart."
            )
    elif any(path.is_file() for path in result_files):
        raise ValueError(
            f"Existing results in {output_dir} have no resume manifest. "
            "Use another --output-dir or pass --restart."
        )
    else:
        temporary = manifest_path.with_suffix(".json.tmp")
        with temporary.open("w") as handle:
            json.dump(manifest, handle, indent=2)
        temporary.replace(manifest_path)

    return (
        load_metric_rows(output_dir / "per_seed_metrics.csv"),
        load_metric_rows(output_dir / "shared_phase1_metrics.csv"),
    )


def matching_rows(rows, dataset, variant, seed):
    return [
        row for row in rows
        if row["dataset"] == dataset
        and row["variant"] == variant
        and row["seed"] == seed
    ]


def completed_checkpoint(rows, dataset, variant, seed):
    selected = matching_rows(rows, dataset, variant, seed)
    if {row["flow"] for row in selected} != {"inflow", "outflow", "mean"}:
        return None
    checkpoint = Path(next(row["checkpoint"] for row in selected if row["flow"] == "mean"))
    return checkpoint if checkpoint.is_file() else None


def discard_partial(rows, dataset, variant, seed):
    rows[:] = [
        row for row in rows
        if not (
            row["dataset"] == dataset
            and row["variant"] == variant
            and row["seed"] == seed
        )
    ]


def persist_progress(output_dir, rows, phase1_rows):
    write_csv(output_dir / "per_seed_metrics.csv", rows)
    write_csv(output_dir / "summary_metrics.csv", summarize(rows))
    write_csv(output_dir / "shared_phase1_metrics.csv", phase1_rows)
    write_csv(output_dir / "shared_phase1_summary.csv", summarize(phase1_rows))


def show_models(variants):
    print("\nAblation models")
    for variant in variants:
        print(f"  {variant}: {VARIANTS[variant]['description']}")


def print_summary(rows):
    print("\nAverages (mean +/- population std across seeds)")
    headers = ("dataset", "variant", "flow", "mae", "eee", "normal_mae", "event_bias")
    printable = []
    for row in rows:
        printable.append({
            "dataset": row["dataset"],
            "variant": row["variant"],
            "flow": row["flow"],
            "mae": f"{row['mae_mean']:.4f} +/- {row['mae_std']:.4f}",
            "eee": f"{row['eee_mean']:.4f} +/- {row['eee_std']:.4f}",
            "normal_mae": f"{row['normal_mae_mean']:.4f} +/- {row['normal_mae_std']:.4f}",
            "event_bias": f"{row['event_signed_error_mean']:.4f} +/- {row['event_signed_error_std']:.4f}",
        })
    widths = {key: max(len(key), *(len(str(row[key])) for row in printable)) for key in headers}
    print("  ".join(key.ljust(widths[key]) for key in headers))
    print("  ".join("-" * widths[key] for key in headers))
    for row in printable:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in headers))


def main():
    args = parse_args()
    if args.graph_file and len(args.config_filenames) != 1:
        raise ValueError("--graph-file can only be used with one config file")
    seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    variants = args.variants or list(VARIANTS)
    show_models(variants)

    output_dir = repo_path(args.output_dir)
    manifest = build_manifest(args, seeds, variants)
    rows, phase1_rows = prepare_resume(output_dir, manifest, args.restart)
    if rows or phase1_rows:
        print(f"\nResuming saved progress from {output_dir}")

    for config_filename in args.config_filenames:
        base = load_config(config_filename)
        dataset = base["dataset"]
        for seed in seeds:
            phase1_checkpoint = completed_checkpoint(
                phase1_rows, dataset, "shared_phase1", seed
            )
            if phase1_checkpoint is not None:
                print(f"\n[resume] dataset={dataset} shared_phase1 seed={seed}")
            else:
                discard_partial(phase1_rows, dataset, "shared_phase1", seed)
                print(f"\n=== dataset={dataset} shared_phase1 seed={seed} ===")
                phase1_args = build_shared_phase1_args(base, args, seed)
                phase1_result = model_supervisor(phase1_args)
                if phase1_result is None or "test_results" not in phase1_result:
                    raise RuntimeError(
                        f"Shared phase-1 failed for dataset={dataset}, seed={seed}"
                    )
                phase1_checkpoint = checkpoint_for(phase1_args)
                if not phase1_checkpoint.is_file():
                    raise FileNotFoundError(f"Missing shared checkpoint: {phase1_checkpoint}")
                add_rows(
                    phase1_rows, dataset, "shared_phase1", seed, phase1_result,
                    phase1_args.log_dir, phase1_checkpoint,
                )
                persist_progress(output_dir, rows, phase1_rows)

            if "A_base_mae" in variants:
                if completed_checkpoint(rows, dataset, "A_base_mae", seed) is None:
                    discard_partial(rows, dataset, "A_base_mae", seed)
                    for source in matching_rows(
                        phase1_rows, dataset, "shared_phase1", seed
                    ):
                        copied = dict(source)
                        copied["variant"] = "A_base_mae"
                        rows.append(copied)
                    persist_progress(output_dir, rows, phase1_rows)
                else:
                    print(f"[resume] dataset={dataset} variant=A_base_mae seed={seed}")

            for variant in variants:
                if variant == "A_base_mae":
                    continue
                if completed_checkpoint(rows, dataset, variant, seed) is not None:
                    print(f"[resume] dataset={dataset} variant={variant} seed={seed}")
                    continue
                discard_partial(rows, dataset, variant, seed)
                print(f"\n=== dataset={dataset} variant={variant} seed={seed} ===")
                run_args = build_variant_args(base, args, seed, variant, phase1_checkpoint)
                result = model_supervisor(run_args)
                if result is None or "test_results" not in result:
                    raise RuntimeError(
                        f"Run failed for dataset={dataset}, variant={variant}, seed={seed}"
                    )
                checkpoint = checkpoint_for(run_args)
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing final checkpoint: {checkpoint}")
                add_rows(rows, dataset, variant, seed, result, run_args.log_dir, checkpoint)
                persist_progress(output_dir, rows, phase1_rows)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = summarize(rows)
    persist_progress(output_dir, rows, phase1_rows)
    print_summary(summary)
    print(f"\nNegative event_bias means systematic underprediction.")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()

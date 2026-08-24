import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import model_supervisor
from scripts.run_paper_rescue_ablation import (
    add_rows,
    apply_overrides,
    build_manifest,
    completed_checkpoint,
    discard_partial,
    file_digest,
    load_config,
    load_metric_rows,
    matching_rows,
    persist_progress,
    prepare_resume,
    print_summary,
    repo_path,
    resolve_paths,
    summarize,
)


SOURCE_VARIANTS = {
    "A_base_mae": "A_base_mae",
    "C_joint_residual_mae": "C_ungated_residual_mae",
    "E_joint_residual_event_weighted": "D_ungated_residual_event_weighted",
}

TRAIN_VARIANTS = {
    "B_frozen_residual_mae": {
        "ablation_mode": "ungated_bias",
        "event_weighted": False,
    },
    "D_frozen_residual_event_weighted": {
        "ablation_mode": "event_weighted_ungated_residual",
        "event_weighted": True,
    },
}

VARIANT_DESCRIPTIONS = {
    "A_base_mae": "base predictor trained with ordinary MAE",
    "B_frozen_residual_mae": "frozen base; residual head trained with ordinary MAE",
    "C_joint_residual_mae": "base and residual jointly fine-tuned with ordinary MAE",
    "D_frozen_residual_event_weighted": (
        "frozen base; residual head trained with event-weighted MAE"
    ),
    "E_joint_residual_event_weighted": (
        "base and residual jointly fine-tuned with event-weighted MAE"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen-base and joint residual training under ordinary and "
            "event-weighted MAE while reusing completed paper-rescue runs."
        )
    )
    parser.add_argument("--config-filenames", nargs="+", default=["configs/NYCTaxi.yaml"])
    parser.add_argument("--source-results", default="paper_rescue_ablation_results")
    parser.add_argument(
        "--output-dir",
        default="residual_training_schedule_ablation_results",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--bias-param-scope", choices=("legacy", "head_only"), default="head_only")
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--scaler-fit",
        choices=("train", "train_val"),
        default="train",
        help="Must match the source paper-rescue result protocol.",
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
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANT_DESCRIPTIONS), default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="residual_training_schedule_ablation")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use seed 1, one epoch, and one train/evaluation batch.",
    )
    return parser.parse_args()


def load_source(source_dir):
    manifest_path = source_dir / "run_manifest.json"
    metrics_path = source_dir / "per_seed_metrics.csv"
    phase1_path = source_dir / "shared_phase1_metrics.csv"
    for path in (manifest_path, metrics_path, phase1_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source result file is missing: {path}")
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    return manifest, load_metric_rows(metrics_path), load_metric_rows(phase1_path)


def validate_source(source_manifest, args, seeds):
    source_seeds = {int(seed) for seed in source_manifest["seeds"]}
    missing_seeds = sorted(set(seeds) - source_seeds)
    if missing_seeds:
        raise ValueError(f"Source results are missing seeds: {missing_seeds}")
    if float(source_manifest["event_loss_weight"]) != args.event_loss_weight:
        raise ValueError(
            "--event-loss-weight must match the source run: "
            f"{source_manifest['event_loss_weight']}"
        )
    if source_manifest.get("scaler_fit", "train") != args.scaler_fit:
        raise ValueError(
            "--scaler-fit must match the source run: "
            f"{source_manifest.get('scaler_fit', 'train')}"
        )
    if source_manifest.get("bias_param_scope", "head_only") != args.bias_param_scope:
        raise ValueError("--bias-param-scope must match the source run")
    if source_manifest.get("event_mask_protocol", "legacy_raw_p90_v1") != args.event_mask_protocol:
        raise ValueError("--event-mask-protocol must match the source run")
    if source_manifest.get("event_label_source", "legacy_file_unverified") != args.event_label_source:
        raise ValueError("--event-label-source must match the source run")
    requested_configs = {str(repo_path(name).resolve()) for name in args.config_filenames}
    source_configs = {entry["path"] for entry in source_manifest["configs"]}
    missing_configs = requested_configs - source_configs
    if missing_configs:
        raise ValueError(
            "Source results do not contain the requested configs: "
            + ", ".join(sorted(missing_configs))
        )


def copy_source_variant(
    destination_rows,
    source_rows,
    dataset,
    source_variant,
    destination_variant,
    seed,
):
    if completed_checkpoint(destination_rows, dataset, destination_variant, seed):
        return True
    selected = matching_rows(source_rows, dataset, source_variant, seed)
    if {row["flow"] for row in selected} != {"inflow", "outflow", "mean"}:
        raise ValueError(
            f"Source results are incomplete for {dataset}, {source_variant}, seed={seed}"
        )
    checkpoint = Path(next(row["checkpoint"] for row in selected if row["flow"] == "mean"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing: {checkpoint}")
    discard_partial(destination_rows, dataset, destination_variant, seed)
    for source in selected:
        copied = dict(source)
        copied["variant"] = destination_variant
        destination_rows.append(copied)
    return False


def copy_phase1(destination_rows, source_rows, dataset, seed):
    checkpoint = completed_checkpoint(destination_rows, dataset, "shared_phase1", seed)
    if checkpoint is not None:
        return checkpoint
    selected = matching_rows(source_rows, dataset, "shared_phase1", seed)
    if {row["flow"] for row in selected} != {"inflow", "outflow", "mean"}:
        raise ValueError(f"Source phase-1 results are incomplete for {dataset}, seed={seed}")
    checkpoint = Path(next(row["checkpoint"] for row in selected if row["flow"] == "mean"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Source phase-1 checkpoint is missing: {checkpoint}")
    discard_partial(destination_rows, dataset, "shared_phase1", seed)
    destination_rows.extend(dict(row) for row in selected)
    return checkpoint


def build_frozen_args(base, args, seed, variant, phase1_checkpoint):
    spec = TRAIN_VARIANTS[variant]
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
        "event_loss_weight": args.event_loss_weight if spec["event_weighted"] else 0.0,
        "ablation_mode": spec["ablation_mode"],
        "load_path": str(phase1_checkpoint),
        "start_phase": "pred_2",
        "stop_after_phase": "pred_2",
        "comment": args.comment,
        "experimentName": f"schedule_{base['dataset']}_{variant}_seed={seed}",
    })
    apply_overrides(config, args)
    return SimpleNamespace(**config)


def show_variants(variants):
    print("\nResidual training-schedule ablation")
    for variant in variants:
        description = VARIANT_DESCRIPTIONS[variant]
        print(f"  {variant}: {description}")


def main():
    args = parse_args()
    if args.graph_file and len(args.config_filenames) != 1:
        raise ValueError("--graph-file can only be used with one config file")
    seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    variants = args.variants or list(VARIANT_DESCRIPTIONS)
    source_dir = repo_path(args.source_results)
    source_manifest, source_rows, source_phase1_rows = load_source(source_dir)
    validate_source(source_manifest, args, seeds)
    show_variants(variants)

    manifest = build_manifest(args, seeds, variants)
    manifest.update({
        "experiment": "residual_training_schedule_ablation",
        "source_manifest_sha256": file_digest(source_dir / "run_manifest.json"),
        "runner_sha256": file_digest(Path(__file__).resolve()),
    })
    output_dir = repo_path(args.output_dir)
    rows, phase1_rows = prepare_resume(output_dir, manifest, args.restart)
    if rows or phase1_rows:
        print(f"\nResuming saved progress from {output_dir}")

    for config_filename in args.config_filenames:
        base = load_config(config_filename)
        dataset = base["dataset"]
        for seed in seeds:
            phase1_checkpoint = copy_phase1(
                phase1_rows, source_phase1_rows, dataset, seed
            )
            persist_progress(output_dir, rows, phase1_rows)

            for destination, source in SOURCE_VARIANTS.items():
                if destination not in variants:
                    continue
                resumed = copy_source_variant(
                    rows, source_rows, dataset, source, destination, seed
                )
                status = "resume" if resumed else "reuse"
                print(f"[{status}] dataset={dataset} variant={destination} seed={seed}")
            persist_progress(output_dir, rows, phase1_rows)

            for variant in variants:
                if variant not in TRAIN_VARIANTS:
                    continue
                if completed_checkpoint(rows, dataset, variant, seed):
                    print(f"[resume] dataset={dataset} variant={variant} seed={seed}")
                    continue
                discard_partial(rows, dataset, variant, seed)
                print(f"\n=== dataset={dataset} variant={variant} seed={seed} ===")
                run_args = build_frozen_args(
                    base, args, seed, variant, phase1_checkpoint
                )
                result = model_supervisor(run_args)
                if result is None or "test_results" not in result:
                    raise RuntimeError(
                        f"Run failed for dataset={dataset}, variant={variant}, seed={seed}"
                    )
                checkpoint = Path(run_args.log_dir) / "best_model_pred_2.pth"
                if not checkpoint.is_file():
                    checkpoint = phase1_checkpoint
                    print(
                        f"[baseline retained] {variant} did not improve validation loss; "
                        "recording the shared phase-1 checkpoint."
                    )
                add_rows(
                    rows, dataset, variant, seed, result, run_args.log_dir, checkpoint
                )
                persist_progress(output_dir, rows, phase1_rows)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = summarize(rows)
    persist_progress(output_dir, rows, phase1_rows)
    print_summary(summary)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()

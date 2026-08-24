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


VARIANTS = {
    "A_base_mae": "base trained with ordinary MAE",
    "B_event_weighted_base": "base trained from initialization with event-weighted MAE",
    "C_single_stage_event_weighted_residual": (
        "base and residual trained together from initialization with event-weighted MAE"
    ),
    "D_two_stage_frozen_event_weighted_residual": (
        "ordinary-MAE base, followed by event-weighted residual-only adaptation"
    ),
    "E_two_stage_joint_event_weighted_residual": (
        "ordinary-MAE base, followed by joint event-weighted base/residual adaptation"
    ),
    "F_three_stage_joint_then_residual_event_weighted": (
        "ordinary-MAE base, joint event-weighted adaptation, then residual-only fine-tuning"
    ),
}

PAPER_VARIANTS = {
    "A_base_mae": "A_base_mae",
    "B_event_weighted_base": "B_base_event_weighted",
}

SCHEDULE_VARIANTS = {
    "D_two_stage_frozen_event_weighted_residual": "D_frozen_residual_event_weighted",
    "F_three_stage_joint_then_residual_event_weighted": "E_joint_residual_event_weighted",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare single-stage and staged event-weighted residual training while "
            "reusing all compatible completed runs."
        )
    )
    parser.add_argument("--config-filenames", nargs="+", default=["configs/NYCTaxi.yaml"])
    parser.add_argument("--paper-results", default="paper_rescue_ablation_results")
    parser.add_argument(
        "--schedule-results",
        default="residual_training_schedule_ablation_results",
    )
    parser.add_argument(
        "--output-dir",
        default="event_weighted_phase_training_ablation_results",
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
        help="Must match the paper-rescue and schedule source protocols.",
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
    parser.add_argument("--comment", default="event_weighted_phase_training_ablation")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use seed 1, one epoch, and one train/evaluation batch.",
    )
    return parser.parse_args()


def load_results(directory):
    manifest_path = directory / "run_manifest.json"
    metrics_path = directory / "per_seed_metrics.csv"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete source results: {directory}")
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    phase_path = directory / "shared_phase1_metrics.csv"
    phase_rows = load_metric_rows(phase_path) if phase_path.is_file() else []
    return manifest, load_metric_rows(metrics_path), phase_rows


def validate_source(manifest, args, seeds, label):
    missing = sorted(set(seeds) - {int(seed) for seed in manifest["seeds"]})
    if missing:
        raise ValueError(f"{label} results are missing seeds: {missing}")
    if float(manifest["event_loss_weight"]) != args.event_loss_weight:
        raise ValueError(
            f"{label} event-loss weight is {manifest['event_loss_weight']}, "
            f"not {args.event_loss_weight}"
        )
    if manifest.get("scaler_fit", "train") != args.scaler_fit:
        raise ValueError(
            f"{label} scaler_fit is {manifest.get('scaler_fit', 'train')}, "
            f"not {args.scaler_fit}"
        )
    if manifest.get("bias_param_scope", "head_only") != args.bias_param_scope:
        raise ValueError(f"{label} bias_param_scope does not match")
    if manifest.get("event_mask_protocol", "legacy_raw_p90_v1") != args.event_mask_protocol:
        raise ValueError(f"{label} event_mask_protocol does not match")
    if manifest.get("event_label_source", "legacy_file_unverified") != args.event_label_source:
        raise ValueError(f"{label} event_label_source does not match")
    requested = {str(repo_path(name).resolve()) for name in args.config_filenames}
    available = {entry["path"] for entry in manifest["configs"]}
    if not requested.issubset(available):
        raise ValueError(f"{label} results do not contain all requested datasets")


def copy_rows(destination, source, dataset, source_variant, target_variant, seed):
    if completed_checkpoint(destination, dataset, target_variant, seed):
        return "resume"
    selected = matching_rows(source, dataset, source_variant, seed)
    if {row["flow"] for row in selected} != {"inflow", "outflow", "mean"}:
        raise ValueError(
            f"Incomplete source rows for {dataset}, {source_variant}, seed={seed}"
        )
    checkpoint = Path(next(row["checkpoint"] for row in selected if row["flow"] == "mean"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint}")
    discard_partial(destination, dataset, target_variant, seed)
    for row in selected:
        copied = dict(row)
        copied["variant"] = target_variant
        destination.append(copied)
    return "reuse"


def common_config(base, args, seed, experiment_name):
    config = dict(base)
    data_dir, graph_file = resolve_paths(config, args)
    config.update({
        "seed": seed,
        "data_dir": data_dir,
        "graph_file": graph_file,
        "phase3_mode": "original",
        "bias_param_scope": args.bias_param_scope,
        "event_mask_protocol": args.event_mask_protocol,
        "event_label_source": args.event_label_source,
        "classification_loss_weight": args.classification_loss_weight,
        "event_loss_weight": args.event_loss_weight,
        "comment": args.comment,
        "experimentName": experiment_name,
    })
    apply_overrides(config, args)
    return config


def build_single_stage_args(base, args, seed):
    config = common_config(
        base,
        args,
        seed,
        f"phase_training_{base['dataset']}_single_stage_seed={seed}",
    )
    config.update({
        "mode": "train",
        "ablation_mode": "event_weighted_ungated_end_to_end",
        "load_path": None,
        "start_phase": "pred",
        "stop_after_phase": "pred",
    })
    return SimpleNamespace(**config)


def build_intermediate_eval_args(base, args, seed, checkpoint):
    config = common_config(
        base,
        args,
        seed,
        f"phase_training_{base['dataset']}_two_stage_joint_eval_seed={seed}",
    )
    config.update({
        "mode": "test",
        "best_path": str(checkpoint),
        "ablation_mode": "event_weighted_ungated_residual",
        "load_path": None,
        "start_phase": "bias",
        "stop_after_phase": "bias",
    })
    return SimpleNamespace(**config)


def paper_joint_bias_checkpoint(paper_rows, dataset, seed):
    selected = matching_rows(
        paper_rows, dataset, "D_ungated_residual_event_weighted", seed
    )
    if not selected:
        raise ValueError(f"Missing joint event-weighted source run for {dataset}, seed={seed}")
    final = Path(next(row["checkpoint"] for row in selected if row["flow"] == "mean"))
    checkpoint = final.parent / "best_model_bias.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing intermediate joint checkpoint: {checkpoint}")
    return checkpoint


def show_variants():
    print("\nEvent-weighted phase-training ablation")
    for variant, description in VARIANTS.items():
        print(f"  {variant}: {description}")


def main():
    args = parse_args()
    if args.graph_file and len(args.config_filenames) != 1:
        raise ValueError("--graph-file can only be used with one config file")
    seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    paper_dir = repo_path(args.paper_results)
    schedule_dir = repo_path(args.schedule_results)
    paper_manifest, paper_rows, paper_phase_rows = load_results(paper_dir)
    schedule_manifest, schedule_rows, _ = load_results(schedule_dir)
    validate_source(paper_manifest, args, seeds, "Paper-rescue")
    validate_source(schedule_manifest, args, seeds, "Schedule")
    show_variants()

    manifest = build_manifest(args, seeds, list(VARIANTS))
    manifest.update({
        "experiment": "event_weighted_phase_training_ablation",
        "paper_manifest_sha256": file_digest(paper_dir / "run_manifest.json"),
        "schedule_manifest_sha256": file_digest(schedule_dir / "run_manifest.json"),
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
            if not completed_checkpoint(phase1_rows, dataset, "shared_phase1", seed):
                selected = matching_rows(paper_phase_rows, dataset, "shared_phase1", seed)
                if {row["flow"] for row in selected} != {"inflow", "outflow", "mean"}:
                    raise ValueError(f"Missing phase-1 source rows for {dataset}, seed={seed}")
                discard_partial(phase1_rows, dataset, "shared_phase1", seed)
                phase1_rows.extend(dict(row) for row in selected)

            for target, source in PAPER_VARIANTS.items():
                status = copy_rows(rows, paper_rows, dataset, source, target, seed)
                print(f"[{status}] dataset={dataset} variant={target} seed={seed}")
            for target, source in SCHEDULE_VARIANTS.items():
                status = copy_rows(rows, schedule_rows, dataset, source, target, seed)
                print(f"[{status}] dataset={dataset} variant={target} seed={seed}")
            persist_progress(output_dir, rows, phase1_rows)

            single_variant = "C_single_stage_event_weighted_residual"
            if completed_checkpoint(rows, dataset, single_variant, seed):
                print(f"[resume] dataset={dataset} variant={single_variant} seed={seed}")
            else:
                discard_partial(rows, dataset, single_variant, seed)
                print(f"\n=== dataset={dataset} variant={single_variant} seed={seed} ===")
                run_args = build_single_stage_args(base, args, seed)
                result = model_supervisor(run_args)
                if result is None or "test_results" not in result:
                    raise RuntimeError(f"Single-stage run failed for {dataset}, seed={seed}")
                checkpoint = Path(run_args.log_dir) / "best_model_pred.pth"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing single-stage checkpoint: {checkpoint}")
                add_rows(rows, dataset, single_variant, seed, result, run_args.log_dir, checkpoint)
                persist_progress(output_dir, rows, phase1_rows)

            joint_variant = "E_two_stage_joint_event_weighted_residual"
            if completed_checkpoint(rows, dataset, joint_variant, seed):
                print(f"[resume] dataset={dataset} variant={joint_variant} seed={seed}")
            else:
                discard_partial(rows, dataset, joint_variant, seed)
                checkpoint = paper_joint_bias_checkpoint(paper_rows, dataset, seed)
                print(f"\n=== dataset={dataset} variant={joint_variant} seed={seed} ===")
                eval_args = build_intermediate_eval_args(base, args, seed, checkpoint)
                result = model_supervisor(eval_args)
                if result is None or "test_results" not in result:
                    raise RuntimeError(f"Intermediate evaluation failed for {dataset}, seed={seed}")
                add_rows(rows, dataset, joint_variant, seed, result, eval_args.log_dir, checkpoint)
                persist_progress(output_dir, rows, phase1_rows)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(rows)
    persist_progress(output_dir, rows, phase1_rows)
    print_summary(summary)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()

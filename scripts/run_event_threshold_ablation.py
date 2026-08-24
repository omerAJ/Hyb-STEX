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

from lib.dataloader import get_dataloader
from lib.utils import load_graph
from main import model_supervisor
from scripts.plot_extreme_underprediction_evidence import predict_checkpoint
from scripts.run_event_weighted_phase_training_ablation import common_config
from scripts.run_paper_rescue_ablation import (
    add_rows,
    build_manifest,
    completed_checkpoint,
    discard_partial,
    file_digest,
    load_config,
    matching_rows,
    persist_progress,
    prepare_resume,
    print_summary,
    repo_path,
    resolve_paths,
    summarize,
    write_csv,
)
from scripts.run_residual_training_schedule_ablation import (
    copy_phase1,
    copy_source_variant,
    load_source,
    validate_source,
)


CONTROL_VARIANT = "A_residual_mae_no_event_weight"
SOURCE_CONTROL_VARIANT = "B_frozen_residual_mae"
SOURCE_P90_VARIANT = "D_frozen_residual_event_weighted"
FLOW_NAMES = ("inflow", "outflow")
COMMON_REGIONS = ("Overall", "EVS-90", "Normal", "Top 25%", "Top 10%", "Top 5%", "Top 1%")


def percentile_token(percentile):
    value = float(percentile)
    return f"{value:g}".replace(".", "p")


def threshold_variant(percentile):
    return f"event_weighted_p{percentile_token(percentile)}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare no event weighting with configurable training-target event "
            "percentiles using the finalized two-stage frozen-residual model."
        )
    )
    parser.add_argument("--config-filenames", nargs="+", default=["configs/NYCTaxi.yaml"])
    parser.add_argument(
        "--source-results",
        default="residual_training_schedule_ablation_results",
        help="Completed schedule ablation used for phase-1, control, and p90 reuse.",
    )
    parser.add_argument("--output-dir", default="event_threshold_ablation_results")
    parser.add_argument("--event-percentiles", nargs="+", type=float, default=[90.0, 95.0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--comment", default="event_threshold_ablation")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use seed 1, one epoch, and one train/evaluation batch per phase.",
    )
    return parser.parse_args()


def validate_percentiles(values):
    percentiles = sorted(set(float(value) for value in values))
    if not percentiles or any(value <= 0.0 or value >= 100.0 for value in percentiles):
        raise ValueError("--event-percentiles must contain values between 0 and 100")
    return percentiles


def build_threshold_args(base, args, seed, percentile, phase1_checkpoint):
    variant = threshold_variant(percentile)
    config = common_config(
        base,
        args,
        seed,
        f"event_threshold_{base['dataset']}_{variant}_seed={seed}",
    )
    config.update({
        "mode": "train",
        "ablation_mode": "event_weighted_ungated_residual",
        "event_percentile": float(percentile),
        "load_path": str(phase1_checkpoint),
        "start_phase": "pred_2",
        "stop_after_phase": "pred_2",
    })
    return SimpleNamespace(**config)


def copy_completed_variant(rows, source_rows, dataset, source, target, seed):
    resumed = copy_source_variant(rows, source_rows, dataset, source, target, seed)
    status = "resume" if resumed else "reuse"
    print(f"[{status}] dataset={dataset} variant={target} seed={seed}")


def train_threshold_variant(rows, base, args, seed, percentile, phase1_checkpoint):
    dataset = base["dataset"]
    variant = threshold_variant(percentile)
    if completed_checkpoint(rows, dataset, variant, seed):
        print(f"[resume] dataset={dataset} variant={variant} seed={seed}")
        return

    discard_partial(rows, dataset, variant, seed)
    print(
        f"\n=== dataset={dataset} variant={variant} "
        f"training_percentile={percentile:g} seed={seed} ==="
    )
    run_args = build_threshold_args(base, args, seed, percentile, phase1_checkpoint)
    result = model_supervisor(run_args)
    if result is None or "test_results" not in result:
        raise RuntimeError(f"Run failed for {dataset}, {variant}, seed={seed}")
    checkpoint = Path(run_args.log_dir) / "best_model_pred_2.pth"
    if not checkpoint.is_file():
        checkpoint = phase1_checkpoint
        print(
            f"[baseline retained] {variant} did not improve validation loss; "
            "recording the shared phase-1 checkpoint."
        )
    add_rows(rows, dataset, variant, seed, result, run_args.log_dir, checkpoint)


def region_masks(target, evs90, flow_index):
    y = target[..., flow_index]
    valid = y > 5.0
    values = y[valid]
    masks = {
        "Overall": valid,
        "EVS-90": evs90[..., flow_index] == 1,
        "Normal": (evs90[..., flow_index] != 1) & valid,
    }
    for percentile in (75, 90, 95, 99):
        label = f"Top {100 - percentile}%"
        masks[label] = valid & (y >= np.percentile(values, percentile))
    return masks


def common_metric_rows(dataset, variant, seed, prediction, target, evs90):
    rows = []
    for flow_index, flow in enumerate(FLOW_NAMES):
        difference = prediction[..., flow_index] - target[..., flow_index]
        for region, mask in region_masks(target, evs90, flow_index).items():
            selected = difference[mask]
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "flow": flow,
                "region": region,
                "count": int(mask.sum()),
                "mae": float(np.abs(selected).mean()),
                "signed_error": float(selected.mean()),
                "underprediction_rate": float((selected < 0).mean()),
            })

    for region in COMMON_REGIONS:
        selected = [row for row in rows if row["region"] == region]
        rows.append({
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "flow": "mean",
            "region": region,
            "count": sum(row["count"] for row in selected),
            "mae": float(np.mean([row["mae"] for row in selected])),
            "signed_error": float(np.mean([row["signed_error"] for row in selected])),
            "underprediction_rate": float(
                np.mean([row["underprediction_rate"] for row in selected])
            ),
        })
    return rows


def summarize_common(rows):
    summary = []
    keys = sorted({
        (row["dataset"], row["variant"], row["flow"], row["region"])
        for row in rows
    })
    for dataset, variant, flow, region in keys:
        selected = [
            row for row in rows
            if (row["dataset"], row["variant"], row["flow"], row["region"])
            == (dataset, variant, flow, region)
        ]
        output = {
            "dataset": dataset,
            "variant": variant,
            "flow": flow,
            "region": region,
            "seeds": len(selected),
            "count": selected[0]["count"],
        }
        for metric in ("mae", "signed_error", "underprediction_rate"):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_std"] = float(values.std())
        summary.append(output)
    return summary


def evaluate_common_regions(args, rows, percentiles, seeds, output_dir):
    common_rows = []
    variants = [CONTROL_VARIANT] + [threshold_variant(value) for value in percentiles]
    for config_filename in args.config_filenames:
        base = load_config(config_filename)
        dataset = base["dataset"]
        data_dir, graph_file = resolve_paths(base, args)
        train_batch_size = args.batch_size or base["batch_size"]
        test_batch_size = args.test_batch_size or base["test_batch_size"]
        dataloader = get_dataloader(
            data_dir,
            dataset,
            train_batch_size,
            test_batch_size,
            scalar_type="Standard",
            scaler_fit="train",
        )
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        graph = load_graph(graph_file, device=device)

        target_reference = None
        evs90_reference = None
        for variant in variants:
            mode = "ungated_bias" if variant == CONTROL_VARIANT else "event_weighted_ungated_residual"
            for seed in seeds:
                checkpoint = completed_checkpoint(rows, dataset, variant, seed)
                if checkpoint is None:
                    raise ValueError(f"Missing completed checkpoint for {dataset}, {variant}, seed={seed}")
                prediction, target, evs90 = predict_checkpoint(
                    base,
                    dataloader,
                    graph,
                    graph_file,
                    checkpoint,
                    mode,
                    "pred_2",
                )
                if target_reference is None:
                    target_reference = target
                    evs90_reference = evs90
                elif not np.allclose(target, target_reference) or not np.array_equal(evs90, evs90_reference):
                    raise ValueError("Common evaluation targets or EVS-90 labels changed")
                common_rows.extend(
                    common_metric_rows(dataset, variant, seed, prediction, target, evs90)
                )

    summary = summarize_common(common_rows)
    write_csv(output_dir / "common_test_metrics.csv", common_rows)
    write_csv(output_dir / "common_test_summary.csv", summary)
    return summary


def print_common_summary(summary):
    print("\nCommon test-set comparison (mean flow, mean +/- population std)")
    for row in summary:
        if row["flow"] != "mean" or row["region"] not in {"Overall", "EVS-90", "Top 10%", "Top 5%", "Top 1%"}:
            continue
        print(
            f"{row['variant']:32s} {row['region']:8s} "
            f"MAE={row['mae_mean']:.4f} +/- {row['mae_std']:.4f}  "
            f"signed={row['signed_error_mean']:.4f} +/- {row['signed_error_std']:.4f}  "
            f"under={100 * row['underprediction_rate_mean']:.1f}%"
        )


def main():
    args = parse_args()
    if args.graph_file and len(args.config_filenames) != 1:
        raise ValueError("--graph-file can only be used with one config file")
    percentiles = validate_percentiles(args.event_percentiles)
    seeds = args.seeds or ([1] if args.smoke_test else [1, 2, 3])
    variants = [CONTROL_VARIANT] + [threshold_variant(value) for value in percentiles]

    source_dir = repo_path(args.source_results)
    source_manifest, source_rows, source_phase1_rows = load_source(source_dir)
    validate_source(source_manifest, args, seeds)

    print("\nEvent-threshold ablation")
    print(f"  {CONTROL_VARIANT}: two-stage frozen residual with ordinary MAE")
    for percentile in percentiles:
        print(
            f"  {threshold_variant(percentile)}: two-stage frozen residual "
            f"with events above the training-target {percentile:g}th percentile"
        )

    manifest = build_manifest(args, seeds, variants)
    manifest.update({
        "experiment": "event_threshold_ablation",
        "event_percentiles": percentiles,
        "source_manifest_sha256": file_digest(source_dir / "run_manifest.json"),
        "runner_sha256": file_digest(Path(__file__).resolve()),
        "common_evaluation": "EVS-90 plus test-target upper-tail percentiles",
    })
    output_dir = repo_path(args.output_dir)
    if args.restart:
        for name in ("common_test_metrics.csv", "common_test_summary.csv"):
            path = output_dir / name
            if path.is_file():
                path.unlink()
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
            copy_completed_variant(
                rows,
                source_rows,
                dataset,
                SOURCE_CONTROL_VARIANT,
                CONTROL_VARIANT,
                seed,
            )
            if 90.0 in percentiles:
                copy_completed_variant(
                    rows,
                    source_rows,
                    dataset,
                    SOURCE_P90_VARIANT,
                    threshold_variant(90.0),
                    seed,
                )
            persist_progress(output_dir, rows, phase1_rows)

            for percentile in percentiles:
                if percentile == 90.0:
                    continue
                train_threshold_variant(
                    rows, base, args, seed, percentile, phase1_checkpoint
                )
                persist_progress(output_dir, rows, phase1_rows)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = summarize(rows)
    persist_progress(output_dir, rows, phase1_rows)
    print_summary(summary)
    common_summary = evaluate_common_regions(
        args, rows, percentiles, seeds, output_dir
    )
    print_common_summary(common_summary)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()

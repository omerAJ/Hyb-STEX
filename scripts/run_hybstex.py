#!/usr/bin/env python3
"""Train and evaluate the final Hyb-STEX model on one or more datasets.

The public Hyb-STEX configuration is a two-stage model:
1. train the STE-Base predictor with ordinary MAE;
2. freeze the base and train one always-on residual head with p90
   event-weighted MAE (weight 0.25).

This wrapper reuses the tested, resumable phase runners and writes concise
Hyb-STEX-only per-seed and summary CSV files at the selected output root.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NYCBike1", "NYCBike2", "NYCTaxi", "BJTaxi")
FINAL_VARIANT = "D_frozen_residual_event_weighted"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the final two-stage Hyb-STEX model."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--data-dir", default=str(ROOT / "preprocessed_data"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "hybstex"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N; auto-detected when omitted")
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print both phase commands without running them")
    parser.add_argument("--restart", action="store_true", help="restart this output root instead of resuming")
    return parser.parse_args(argv)


def _extend_if(arguments: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        arguments.extend((flag, str(value)))


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str], Path, Path]:
    if not args.seeds or any(seed < 1 for seed in args.seeds):
        raise ValueError("--seeds must contain positive integers")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if args.event_loss_weight < 0:
        raise ValueError("--event-loss-weight must be nonnegative")
    if args.smoke_test and tuple(args.seeds) not in {(1,), (1, 2, 3)}:
        raise ValueError("--smoke-test uses seed 1; omit --seeds or pass --seeds 1")

    output_root = Path(args.output_dir).expanduser().resolve()
    base_output = output_root / "base"
    final_output = output_root / "final"
    config_paths = [str(ROOT / "configs" / f"{dataset}.yaml") for dataset in args.datasets]
    selected_seeds = (1,) if args.smoke_test else tuple(args.seeds)

    common = [
        "--config-filenames", *config_paths,
        "--data-dir", str(Path(args.data_dir).expanduser().resolve()),
        "--seeds", *(str(seed) for seed in selected_seeds),
        "--event-loss-weight", str(args.event_loss_weight),
        "--bias-param-scope", "head_only",
        "--scaler-fit", "train_val",
        "--event-mask-protocol", "train_all_node_flow_p90_valid_v2",
        "--event-label-source", "file_verified",
    ]
    _extend_if(common, "--device", args.device)
    _extend_if(common, "--epochs", args.epochs)
    _extend_if(common, "--num-epochs", args.num_epochs)
    _extend_if(common, "--batch-size", args.batch_size)
    _extend_if(common, "--test-batch-size", args.test_batch_size)
    _extend_if(common, "--max-train-batches", args.max_train_batches)
    _extend_if(common, "--max-eval-batches", args.max_eval_batches)
    if args.smoke_test:
        common.append("--smoke-test")
    if args.restart:
        common.append("--restart")

    base_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_paper_rescue_ablation.py"),
        *common,
        "--output-dir", str(base_output),
        "--variants", "A_base_mae",
        "--comment", "hybstex_base",
    ]
    final_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_residual_training_schedule_ablation.py"),
        *common,
        "--source-results", str(base_output),
        "--output-dir", str(final_output),
        "--variants", FINAL_VARIANT,
        "--comment", "hybstex_final",
    ]
    return base_command, final_command, output_root, final_output


def _write_filtered_csv(source: Path, destination: Path) -> int:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = [row for row in reader if row.get("variant") == FINAL_VARIANT]
    if not rows:
        raise RuntimeError(f"No {FINAL_VARIANT} rows found in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_command, final_command, output_root, final_output = build_commands(args)
    for label, command in (("base", base_command), ("final Hyb-STEX", final_command)):
        print(f"[{label}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.dry_run:
        return 0

    per_seed_count = _write_filtered_csv(
        final_output / "per_seed_metrics.csv", output_root / "per_seed_metrics.csv"
    )
    summary_count = _write_filtered_csv(
        final_output / "summary_metrics.csv", output_root / "summary_metrics.csv"
    )
    run_info = {
        "model": "Hyb-STEX",
        "variant": FINAL_VARIANT,
        "datasets": list(args.datasets),
        "seeds": [1] if args.smoke_test else list(args.seeds),
        "event_percentile": 90.0,
        "event_loss_weight": args.event_loss_weight,
        "scaler_fit": "train_val",
        "event_mask_protocol": "train_all_node_flow_p90_valid_v2",
        "event_label_source": "file_verified",
        "per_seed_rows": per_seed_count,
        "summary_rows": summary_count,
    }
    (output_root / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Hyb-STEX results: {output_root / 'summary_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

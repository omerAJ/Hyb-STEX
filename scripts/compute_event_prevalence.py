"""Report protocol-v2 high-flow prevalence with an explicit denominator.

This script is the source of the high-flow prevalence rows for the revised
paper.  It deliberately does *not* read the legacy ``evs_90`` arrays: those
arrays represent raw p90 exceedances and can include targets excluded from the
ordinary MAE population.  Instead it rebuilds the canonical v2 masks:

    V = 1[Y > 5]
    E = V & 1[Y > q_train,0.90]

where one q is fitted from all training targets for each horizon, region, and
flow channel.  Reported shares are therefore |E| / |V|, not |E| divided by
all grid-time observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.event_masks import build_event_masks, corrected_p90_spec, fit_event_thresholds


DEFAULT_DATASETS = ("NYCBike1", "NYCBike2", "NYCTaxi", "BJTaxi")
FLOW_NAMES = ("inflow", "outflow")
SPLITS = ("train", "val", "test")
# The published benchmark aggregation intervals.  They are needed only to
# convert a count over a split to a 24-hour-equivalent rate.
SLOTS_PER_DAY = {"NYCBike1": 24, "NYCBike2": 48, "NYCTaxi": 48, "BJTaxi": 48}


def repo_root() -> Path:
    return ROOT


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_y(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    path = data_dir / dataset / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as loaded:
        if "y" not in loaded.files:
            raise KeyError(f"{path} does not contain a 'y' array")
        return loaded["y"].astype(np.float32)


def count_masked_targets(valid: np.ndarray, event: np.ndarray, slots_per_day: int) -> dict[str, Any]:
    """Count v2 valid targets and events, including their exact denominators."""
    if valid.shape != event.shape:
        raise ValueError("valid and event masks must have the same shape")
    if valid.ndim < 2 or valid.shape[-1] != len(FLOW_NAMES):
        raise ValueError("expected a target array whose final axis is inflow/outflow")
    if np.any(event & ~valid):
        raise ValueError("event mask must be a subset of the valid-target mask")

    # Every target time point in the split contributes one region-channel
    # observation.  This remains well-defined when a chronological split
    # starts or ends mid-day, yielding a fractional number of day equivalents.
    # The dimensions before the region/flow axes are [sample, horizon].
    # Their product is the number of target temporal slots represented by the
    # split (one horizon in the present benchmark files).
    target_time_points = int(np.prod(valid.shape[:-2]))
    region_count = int(valid.shape[-2])
    temporal_slots = target_time_points
    day_equivalents = temporal_slots / slots_per_day
    if day_equivalents <= 0:
        raise ValueError("split must contain at least one temporal target slot")

    result: dict[str, Any] = {
        "all_target_observations": int(valid.size),
        "valid_observations": int(valid.sum()),
        "event_observations": int(event.sum()),
        "event_share_of_valid": float(event.sum() / valid.sum()) if valid.any() else float("nan"),
        "regions": region_count,
        "target_temporal_slots": temporal_slots,
        "slots_per_day": slots_per_day,
        "day_equivalents": day_equivalents,
    }
    for flow_index, flow_name in enumerate(FLOW_NAMES):
        flow_valid = valid[..., flow_index]
        flow_event = event[..., flow_index]
        valid_count = int(flow_valid.sum())
        event_count = int(flow_event.sum())
        result[f"{flow_name}_all_target_observations"] = int(flow_valid.size)
        result[f"{flow_name}_valid_observations"] = valid_count
        result[f"{flow_name}_event_observations"] = event_count
        result[f"{flow_name}_event_share_of_valid"] = (
            event_count / valid_count if valid_count else float("nan")
        )
        result[f"{flow_name}_valid_observations_per_day"] = valid_count / day_equivalents
        result[f"{flow_name}_events_per_day"] = event_count / day_equivalents
    return result


def compute_dataset(data_dir: Path, dataset: str) -> dict[str, Any]:
    if dataset not in SLOTS_PER_DAY:
        raise ValueError(f"No published sampling interval is configured for {dataset!r}")
    y_by_split = {split: load_y(data_dir, dataset, split) for split in SPLITS}
    spec = corrected_p90_spec()
    thresholds = fit_event_thresholds(y_by_split["train"], spec)
    split_counts = {}
    for split, targets in y_by_split.items():
        masks = build_event_masks(targets, thresholds, spec)
        split_counts[split] = count_masked_targets(
            masks.valid, masks.event, SLOTS_PER_DAY[dataset]
        )
    return {
        "dataset": dataset,
        "protocol": spec.to_dict(),
        "threshold_source_split": "train",
        "evaluation_split_for_paper_table": "test",
        "split_target_shapes": {split: list(values.shape) for split, values in y_by_split.items()},
        "counts": split_counts,
    }


def flatten_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for split_name, counts in result["counts"].items():
            row = {
                "dataset": result["dataset"],
                "evaluated_split": split_name,
                "threshold_source_split": result["threshold_source_split"],
                "event_mask_protocol": result["protocol"]["protocol_id"],
                "valid_rule": "Y > 5",
                "event_rule": "(Y > 5) and (Y > train-channel-p90)",
            }
            row.update(counts)
            rows.append(row)
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def paper_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["evaluated_split"] != "test":
            continue
        result.append(
            {
                "dataset": row["dataset"],
                "test target slots / 24-h equivalents": (
                    f"{row['target_temporal_slots']:.0f} / {row['day_equivalents']:.3f}"
                ),
                "valid targets/day, IN / OUT": (
                    f"{row['inflow_valid_observations_per_day']:.2f} / "
                    f"{row['outflow_valid_observations_per_day']:.2f}"
                ),
                "|V| / |E|, IN; OUT": (
                    f"{row['inflow_valid_observations']:,} / {row['inflow_event_observations']:,}; "
                    f"{row['outflow_valid_observations']:,} / {row['outflow_event_observations']:,}"
                ),
                "high-flow share of valid targets, IN / OUT": (
                    f"{100 * row['inflow_event_share_of_valid']:.2f}% / "
                    f"{100 * row['outflow_event_share_of_valid']:.2f}%"
                ),
                "high-flow events per 24-h equivalent, IN / OUT": (
                    f"{row['inflow_events_per_day']:.2f} / "
                    f"{row['outflow_events_per_day']:.2f}"
                ),
            }
        )
    return result


def write_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(results)
    with (output_dir / "event_prevalence.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output_dir / "event_prevalence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    columns = list(paper_rows(rows)[0].keys())
    with (output_dir / "event_prevalence.md").open("w", encoding="utf-8") as handle:
        handle.write("# Protocol-v2 high-flow prevalence\n\n")
        handle.write(
            "Paper table uses the chronological test split only. Thresholds are fitted once from "
            "all training targets per horizon, region, and flow channel. A valid target satisfies "
            "`Y > 5`; an event satisfies `Y > 5` and `Y > q_train,0.90`. Shares use valid targets "
            "as their denominator. Rates use 24-hour equivalents, so non-day-aligned split boundaries "
            "can yield fractional day counts.\n\n"
        )
        handle.write(markdown_table(paper_rows(rows), columns))
        handle.write("\n")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(root / "preprocessed_data"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", default=str(root / "event_prevalence_results"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    results = [compute_dataset(data_dir, dataset) for dataset in parse_csv(args.datasets)]
    rows = flatten_rows(results)
    table_rows = paper_rows(rows)
    columns = list(table_rows[0].keys())
    print("\nPaper table metric: protocol-v2 high-flow share and rate on the test split\n")
    print(markdown_table(table_rows, columns))

    if not args.no_write:
        output_dir = Path(args.output_dir).resolve()
        write_outputs(output_dir, results)
        print(f"\nSaved outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

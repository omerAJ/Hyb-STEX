"""
Compute high-flow event prevalence for Hyb-STEX datasets.

Default command from the repo root:
  C:\\Users\\PCF\\.conda\\envs\\sds-test\\python.exe scripts\\compute_event_prevalence.py

The reported table uses corrected evs_90 labels by default. These labels are
created from node/flow-specific thresholds learned from the training targets,
then applied unchanged to train/val/test targets. Use --label-source train to
recompute the same labels from y instead of reading evs_90 from disk.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASETS = ("NYCBike1", "NYCBike2", "NYCTaxi", "BJTaxi")
FLOW_NAMES = ("inflow", "outflow")
SPLITS = ("train", "val", "test")
LABEL_SOURCE_CHOICES = ("file", "train")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def load_evs(data_dir: Path, dataset: str, split: str, label_name: str) -> np.ndarray:
    path = data_dir / dataset / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as loaded:
        if label_name not in loaded.files:
            raise KeyError(f"{path} does not contain a '{label_name}' array")
        return loaded[label_name].astype(np.float32)


def train_thresholds(y_train: np.ndarray, percentile: float) -> np.ndarray:
    # Threshold per prediction horizon, node, and flow direction.
    return np.percentile(y_train, percentile, axis=0)


def count_events(y: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    events = y > thresholds
    total = int(events.size)
    positives = int(events.sum())
    result: dict[str, Any] = {
        "positives": positives,
        "total": total,
        "prevalence": positives / total if total else float("nan"),
    }
    for flow_index, flow_name in enumerate(FLOW_NAMES):
        flow_events = events[..., flow_index]
        flow_total = int(flow_events.size)
        flow_positives = int(flow_events.sum())
        result[f"{flow_name}_positives"] = flow_positives
        result[f"{flow_name}_total"] = flow_total
        result[f"{flow_name}_prevalence"] = flow_positives / flow_total if flow_total else float("nan")
    return result


def count_labels(events: np.ndarray) -> dict[str, Any]:
    total = int(events.size)
    positives = int(events.sum())
    result: dict[str, Any] = {
        "positives": positives,
        "total": total,
        "prevalence": positives / total if total else float("nan"),
    }
    for flow_index, flow_name in enumerate(FLOW_NAMES):
        flow_events = events[..., flow_index]
        flow_total = int(flow_events.size)
        flow_positives = int(flow_events.sum())
        result[f"{flow_name}_positives"] = flow_positives
        result[f"{flow_name}_total"] = flow_total
        result[f"{flow_name}_prevalence"] = flow_positives / flow_total if flow_total else float("nan")
    return result


def combine_counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    positives = sum(int(item["positives"]) for item in items)
    total = sum(int(item["total"]) for item in items)
    result["positives"] = positives
    result["total"] = total
    result["prevalence"] = positives / total if total else float("nan")
    for flow_name in FLOW_NAMES:
        flow_positives = sum(int(item[f"{flow_name}_positives"]) for item in items)
        flow_total = sum(int(item[f"{flow_name}_total"]) for item in items)
        result[f"{flow_name}_positives"] = flow_positives
        result[f"{flow_name}_total"] = flow_total
        result[f"{flow_name}_prevalence"] = flow_positives / flow_total if flow_total else float("nan")
    return result


def compute_dataset(
    data_dir: Path,
    dataset: str,
    percentile: float,
    label_source: str,
    label_name: str,
) -> dict[str, Any]:
    if label_source == "file":
        split_counts = {
            split: count_labels(load_evs(data_dir, dataset, split, label_name))
            for split in SPLITS
        }
        y_train = load_y(data_dir, dataset, "train")
    elif label_source == "train":
        y_by_split = {split: load_y(data_dir, dataset, split) for split in SPLITS}
        thresholds = train_thresholds(y_by_split["train"], percentile)
        split_counts = {
            split: count_events(y_by_split[split], thresholds)
            for split in SPLITS
        }
        y_train = y_by_split["train"]
    else:
        raise ValueError(f"Unsupported label_source={label_source!r}. Choices: {LABEL_SOURCE_CHOICES}")
    split_counts["val_test"] = combine_counts([split_counts["val"], split_counts["test"]])
    split_counts["all"] = combine_counts([split_counts["train"], split_counts["val"], split_counts["test"]])
    return {
        "dataset": dataset,
        "percentile": percentile,
        "label_source": label_source,
        "label_name": label_name if label_source == "file" else None,
        "threshold_source_split": "train",
        "target_shape_train": list(y_train.shape),
        "counts": split_counts,
    }


def flatten_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for split_name, counts in result["counts"].items():
            row = {
                "dataset": result["dataset"],
                "label_source": result["label_source"],
                "label_name": result["label_name"],
                "threshold_source_split": result["threshold_source_split"],
                "percentile": result["percentile"],
                "evaluated_split": split_name,
                "positives": counts["positives"],
                "total": counts["total"],
                "prevalence": counts["prevalence"],
                "prevalence_percent": 100.0 * counts["prevalence"],
            }
            for flow_name in FLOW_NAMES:
                row[f"{flow_name}_positives"] = counts[f"{flow_name}_positives"]
                row[f"{flow_name}_total"] = counts[f"{flow_name}_total"]
                row[f"{flow_name}_prevalence"] = counts[f"{flow_name}_prevalence"]
                row[f"{flow_name}_prevalence_percent"] = 100.0 * counts[f"{flow_name}_prevalence"]
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


def write_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(results)
    with (output_dir / "event_prevalence.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output_dir / "event_prevalence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    paper_rows = [row for row in rows if row["evaluated_split"] == "val_test"]
    columns = [
        "dataset",
        "prevalence_percent",
        "inflow_prevalence_percent",
        "outflow_prevalence_percent",
        "positives",
        "total",
    ]
    with (output_dir / "event_prevalence.md").open("w", encoding="utf-8") as handle:
        handle.write("# Train-Threshold High-Flow Event Prevalence\n\n")
        handle.write("Evaluation split is validation + test. Thresholds are the training-set 90th percentile per horizon, node, and flow direction.\n\n")
        handle.write(markdown_table(paper_rows, columns))
        handle.write("\n")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(root / "preprocessed_data"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--percentile", default=90.0, type=float)
    parser.add_argument("--label-source", default="file", choices=LABEL_SOURCE_CHOICES)
    parser.add_argument("--label-name", default="evs_90")
    parser.add_argument("--output-dir", default=str(root / "event_prevalence_results"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    datasets = parse_csv(args.datasets)
    results = [
        compute_dataset(data_dir, dataset, args.percentile, args.label_source, args.label_name)
        for dataset in datasets
    ]
    rows = flatten_rows(results)
    paper_rows = [row for row in rows if row["evaluated_split"] == "val_test"]
    paper_columns = [
        "dataset",
        "prevalence_percent",
        "inflow_prevalence_percent",
        "outflow_prevalence_percent",
        "positives",
        "total",
    ]
    diagnostic_columns = [
        "dataset",
        "evaluated_split",
        "prevalence_percent",
        "inflow_prevalence_percent",
        "outflow_prevalence_percent",
        "positives",
        "total",
    ]

    print(f"\nData dir: {data_dir}")
    print(f"Label source: {args.label_source}")
    print("\nPaper table metric: val+test prevalence using train-derived high-flow labels\n")
    print(markdown_table(paper_rows, paper_columns))
    print("\nDiagnostics by split\n")
    print(markdown_table(rows, diagnostic_columns))

    if not args.no_write:
        output_dir = Path(args.output_dir).resolve()
        write_outputs(output_dir, results)
        print(f"\nSaved outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

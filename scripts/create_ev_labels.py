"""
Create high-flow event labels for Hyb-STEX/ST-SSL datasets.

The label rule matches the paper methodology: for each dataset, prediction
horizon, node, and flow direction, the high-flow threshold is estimated from
the training targets only. The same threshold tensor is then applied to the
train, validation, and test targets.

Default command from the repo root:
  python scripts/create_ev_labels.py

By default this writes labelled copies under preprocessed_data/<DATASET>/ and
leaves the original ST-SSL files untouched.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASETS = ("NYCBike1", "NYCBike2", "NYCTaxi", "BJTaxi")
SPLITS = ("train", "val", "test")
FLOW_NAMES = ("inflow", "outflow")


@dataclass(frozen=True)
class LabelSummary:
    dataset: str
    split: str
    threshold_source: str
    percentile: float
    label_name: str
    y_shape: tuple[int, ...]
    positives: int
    total: int
    prevalence: float
    inflow_positives: int
    inflow_total: int
    inflow_prevalence: float
    outflow_positives: int
    outflow_total: int
    outflow_prevalence: float
    output_file: str

    def as_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "threshold_source": self.threshold_source,
            "percentile": self.percentile,
            "label_name": self.label_name,
            "y_shape": "x".join(str(value) for value in self.y_shape),
            "positives": self.positives,
            "total": self.total,
            "prevalence": self.prevalence,
            "prevalence_percent": 100.0 * self.prevalence,
            "inflow_positives": self.inflow_positives,
            "inflow_total": self.inflow_total,
            "inflow_prevalence": self.inflow_prevalence,
            "inflow_prevalence_percent": 100.0 * self.inflow_prevalence,
            "outflow_positives": self.outflow_positives,
            "outflow_total": self.outflow_total,
            "outflow_prevalence": self.outflow_prevalence,
            "outflow_prevalence_percent": 100.0 * self.outflow_prevalence,
            "output_file": self.output_file,
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def dataset_file(data_dir: Path, dataset: str, split: str) -> Path:
    return data_dir / dataset / f"{split}.npz"


def require_dataset_files(data_dir: Path, datasets: list[str]) -> None:
    missing: list[Path] = []
    for dataset in datasets:
        dataset_dir = data_dir / dataset
        for split in SPLITS:
            path = dataset_file(data_dir, dataset, split)
            if not path.is_file():
                missing.append(path)
        adj_path = dataset_dir / "adj_mx.npz"
        if not adj_path.is_file():
            missing.append(adj_path)
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required dataset files:\n{details}")


def load_y(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as loaded:
        if "y" not in loaded.files:
            raise KeyError(f"{npz_path} does not contain a 'y' array.")
        return loaded["y"]


def train_thresholds(y_train: np.ndarray, percentile: float) -> np.ndarray:
    # Shape is (horizon, nodes, flows), broadcastable to every split's y.
    return np.percentile(y_train, percentile, axis=0)


def make_event_labels(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if y.ndim != 4:
        raise ValueError(f"Expected y to be 4D (#samples, horizon, nodes, flows), got {y.shape}.")
    if tuple(y.shape[1:]) != tuple(thresholds.shape):
        raise ValueError(f"Threshold shape {thresholds.shape} is incompatible with y shape {y.shape}.")
    return (y > thresholds).astype(np.float32)


def summarize_labels(
    dataset: str,
    split: str,
    percentile: float,
    label_name: str,
    labels: np.ndarray,
    output_file: Path,
) -> LabelSummary:
    total = int(labels.size)
    positives = int(labels.sum())
    flow_counts: dict[str, tuple[int, int, float]] = {}
    for flow_index, flow_name in enumerate(FLOW_NAMES):
        flow_labels = labels[..., flow_index]
        flow_total = int(flow_labels.size)
        flow_positives = int(flow_labels.sum())
        flow_prevalence = flow_positives / flow_total if flow_total else float("nan")
        flow_counts[flow_name] = (flow_positives, flow_total, flow_prevalence)

    return LabelSummary(
        dataset=dataset,
        split=split,
        threshold_source="train",
        percentile=percentile,
        label_name=label_name,
        y_shape=tuple(int(value) for value in labels.shape),
        positives=positives,
        total=total,
        prevalence=positives / total if total else float("nan"),
        inflow_positives=flow_counts["inflow"][0],
        inflow_total=flow_counts["inflow"][1],
        inflow_prevalence=flow_counts["inflow"][2],
        outflow_positives=flow_counts["outflow"][0],
        outflow_total=flow_counts["outflow"][1],
        outflow_prevalence=flow_counts["outflow"][2],
        output_file=str(output_file),
    )


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def stream_copy_npz_with_label(
    source_path: Path,
    output_path: Path,
    label_name: str,
    labels: np.ndarray,
    compression: int = zipfile.ZIP_DEFLATED,
    chunk_size: int = 16 * 1024 * 1024,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    label_member = f"{label_name}.npy"

    try:
        with zipfile.ZipFile(source_path, "r") as source, zipfile.ZipFile(
            temp_path,
            "w",
            compression=compression,
        ) as target:
            for info in source.infolist():
                if info.filename == label_member:
                    continue
                target_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                target_info.compress_type = compression
                target_info.external_attr = info.external_attr
                with source.open(info, "r") as source_member, target.open(
                    target_info,
                    "w",
                    force_zip64=True,
                ) as target_member:
                    shutil.copyfileobj(source_member, target_member, length=chunk_size)
            target.writestr(label_member, npy_bytes(labels), compress_type=compression)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def copy_adj_matrix(source_dir: Path, output_dir: Path, dataset: str, dry_run: bool) -> None:
    source_path = source_dir / dataset / "adj_mx.npz"
    output_path = output_dir / dataset / "adj_mx.npz"
    if dry_run:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def output_npz_path(source_data_dir: Path, output_data_dir: Path | None, dataset: str, split: str) -> Path:
    if output_data_dir is None:
        return dataset_file(source_data_dir, dataset, split)
    return output_data_dir / dataset / f"{split}.npz"


def generate_dataset_labels(
    data_dir: Path,
    dataset: str,
    percentile: float,
    label_name: str,
    output_dir: Path | None,
    dry_run: bool,
) -> list[LabelSummary]:
    train_path = dataset_file(data_dir, dataset, "train")
    y_train = load_y(train_path)
    thresholds = train_thresholds(y_train, percentile)

    summaries: list[LabelSummary] = []
    if output_dir is not None:
        copy_adj_matrix(data_dir, output_dir, dataset, dry_run)

    for split in SPLITS:
        source_path = dataset_file(data_dir, dataset, split)
        y = load_y(source_path)
        labels = make_event_labels(y, thresholds)
        target_path = output_npz_path(data_dir, output_dir, dataset, split)
        if not dry_run:
            stream_copy_npz_with_label(source_path, target_path, label_name, labels)
        summaries.append(
            summarize_labels(
                dataset=dataset,
                split=split,
                percentile=percentile,
                label_name=label_name,
                labels=labels,
                output_file=target_path,
            )
        )
    return summaries


def generate_all(
    data_dir: Path,
    datasets: list[str],
    percentile: float,
    label_name: str,
    output_dir: Path | None,
    dry_run: bool = False,
) -> list[LabelSummary]:
    require_dataset_files(data_dir, datasets)
    all_summaries: list[LabelSummary] = []
    for dataset in datasets:
        all_summaries.extend(
            generate_dataset_labels(
                data_dir=data_dir,
                dataset=dataset,
                percentile=percentile,
                label_name=label_name,
                output_dir=output_dir,
                dry_run=dry_run,
            )
        )
    return all_summaries


def write_summary_csv(summary_path: Path, summaries: list[LabelSummary]) -> None:
    if not summaries:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary.as_row() for summary in summaries]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_summary_table(summaries: list[LabelSummary]) -> str:
    headers = ["dataset", "split", "prevalence_percent", "inflow_prevalence_percent", "outflow_prevalence_percent"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for summary in summaries:
        row = summary.as_row()
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["split"]),
                    f"{float(row['prevalence_percent']):.4f}",
                    f"{float(row['inflow_prevalence_percent']):.4f}",
                    f"{float(row['outflow_prevalence_percent']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Generate train-threshold high-flow event labels.")
    parser.add_argument("--data-dir", default=str(root / "external" / "ST-SSL_Dataset"))
    parser.add_argument("--output-dir", default=str(root / "preprocessed_data"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--percentile", default=90.0, type=float)
    parser.add_argument("--label-name", default="evs_90")
    parser.add_argument("--summary-file", default=str(root / "event_prevalence_results" / "ev_label_generation_summary.csv"))
    parser.add_argument("--in-place", action="store_true", help="Update files in --data-dir instead of writing copies.")
    parser.add_argument("--dry-run", action="store_true", help="Compute labels and summaries without writing npz files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = None if args.in_place else Path(args.output_dir).resolve()
    datasets = parse_csv_list(args.datasets)

    print(f"Source data: {data_dir}")
    print(f"Output data: {'in-place' if output_dir is None else output_dir}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Label rule: training-set {args.percentile:g}th percentile per horizon/node/flow")

    summaries = generate_all(
        data_dir=data_dir,
        datasets=datasets,
        percentile=float(args.percentile),
        label_name=args.label_name,
        output_dir=output_dir,
        dry_run=bool(args.dry_run),
    )

    summary_path = Path(args.summary_file).resolve()
    if not args.dry_run:
        write_summary_csv(summary_path, summaries)
        print(f"Summary CSV: {summary_path}")
    print(format_summary_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

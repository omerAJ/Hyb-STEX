#!/usr/bin/env python3
"""Create the publication test-set comparison for STE-Base and Hyb-STEX.

The figure is deliberately prediction-blind.  For each flow channel in one
pre-specified dataset, it centers a fixed window on the largest *observed*
high-flow event in the chronological test split.  High-flow events use the
strict, node- and flow-specific training-target p90 protocol.  Predictions do
not participate in example selection, and the validation split is never read.

The default inputs are the provenance-checked artifacts produced by
``scripts/run_corrected_paper_results.py``.  Curves are the mean of the three
matched optimization seeds and ribbons show their population standard
deviation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FLOW_NAMES = ("inflow", "outflow")
FLOW_LABELS = ("Inflow", "Outflow")
MODEL_VARIANTS = {
    "STE-Base": "base_mae",
    "Hyb-STEX": "frozen_event_weighted_residual",
}
MODEL_COLORS = {
    "STE-Base": "#0072B2",
    "Hyb-STEX": "#D55E00",
}
OBSERVED_COLOR = "#111827"
THRESHOLD_COLOR = "#6B7280"
EVENT_COLOR = "#E9A23B"
SLOTS_PER_DAY = {
    "NYCBike1": 24,
    "NYCBike2": 48,
    "NYCTaxi": 48,
    "BJTaxi": 48,
}
DATASET_LABELS = {
    "NYCBike1": "NYC Bike 1",
    "NYCBike2": "NYC Bike 2",
    "NYCTaxi": "NYC Taxi",
    "BJTaxi": "BJ Taxi",
}
SEED_PATTERN = re.compile(r"seed_(\d+)\.npz$")
EXPECTED_PROTOCOL = {
    "protocol_id": "train_all_node_flow_p90_valid_v2",
    "percentile": 90.0,
    "valid_min": 5.0,
    "comparison": "gt",
    "fit_population": "all_targets",
}


@dataclass(frozen=True)
class Selection:
    """One prediction-blind maximum-event selection."""

    flow_index: int
    test_index: int
    horizon_index: int
    node_index: int
    window_start: int
    window_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot test-only maximum observed high-flow events for STE-Base "
            "and Hyb-STEX using train-fitted node/flow p90 thresholds."
        )
    )
    parser.add_argument(
        "--dataset",
        default="NYCTaxi",
        choices=tuple(SLOTS_PER_DAY),
        help="Pre-specified qualitative dataset (default: NYCTaxi).",
    )
    parser.add_argument(
        "--data-dir",
        default="preprocessed_data",
        help="Directory containing <dataset>/train.npz and test.npz.",
    )
    parser.add_argument(
        "--results-dir",
        default="corrected_valid_target_p90_results",
        help="Canonical corrected result suite.",
    )
    parser.add_argument(
        "--output-dir",
        default="publication_test_high_flow_figure",
        help="Directory for PDF, PNG, and provenance JSON outputs.",
    )
    parser.add_argument(
        "--basename",
        default="hybstex_vs_ste_base_test_high_flow",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=49,
        help="Odd number of test samples in each centered window.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_protocol(protocol: dict[str, Any], source: Path) -> None:
    for key, expected in EXPECTED_PROTOCOL.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"Unexpected {key!r} in {source}: "
                f"expected {expected!r}, found {protocol.get(key)!r}"
            )


def recorded_data_hash(
    run_manifest: dict[str, Any], dataset: str, filename: str
) -> str:
    matches = []
    for record in run_manifest.get("data_hashes", []):
        path = Path(str(record.get("path", "")))
        if path.name == filename and path.parent.name == dataset:
            matches.append(str(record.get("sha256", "")))
    if len(matches) != 1 or not matches[0]:
        raise ValueError(
            f"Run manifest must contain exactly one hash for {dataset}/{filename}"
        )
    return matches[0]


def verify_recorded_file(path: Path, expected_sha256: str, role: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{role} hash mismatch for {path}: expected {expected_sha256}, "
            f"found {actual}"
        )
    return actual


def load_raw_targets(
    data_dir: Path,
    results_dir: Path,
    dataset: str,
    run_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only train/test targets and verify the saved evaluation context.

    No validation path is constructed or opened here.
    """

    dataset_data_dir = data_dir / dataset
    train_path = require_file(dataset_data_dir / "train.npz")
    test_path = require_file(dataset_data_dir / "test.npz")
    train_sha = verify_recorded_file(
        train_path,
        recorded_data_hash(run_manifest, dataset, "train.npz"),
        "Training split",
    )
    test_sha = verify_recorded_file(
        test_path,
        recorded_data_hash(run_manifest, dataset, "test.npz"),
        "Test split",
    )

    with np.load(train_path, allow_pickle=False) as archive:
        y_train = np.asarray(archive["y"])
    with np.load(test_path, allow_pickle=False) as archive:
        y_test = np.asarray(archive["y"])

    if y_train.ndim != 4 or y_test.ndim != 4:
        raise ValueError("Expected train/test targets with shape [time,horizon,node,flow]")
    if y_train.shape[1:] != y_test.shape[1:]:
        raise ValueError("Train and test target channel shapes do not match")
    if y_test.shape[1] != 1 or y_test.shape[-1] != len(FLOW_NAMES):
        raise ValueError("Expected one-step targets with inflow and outflow channels")
    if not np.all(np.isfinite(y_train)) or not np.all(np.isfinite(y_test)):
        raise ValueError("Train and test targets must be finite")

    thresholds = np.percentile(y_train, EXPECTED_PROTOCOL["percentile"], axis=0)
    valid = y_test > EXPECTED_PROTOCOL["valid_min"]
    events = valid & (y_test > thresholds)

    dataset_results_dir = results_dir / dataset
    threshold_path = require_file(dataset_results_dir / "thresholds.npz")
    context_path = require_file(dataset_results_dir / "evaluation_context.npz")

    expected_threshold_hash = str(dataset_manifest["thresholds"]["sha256"])
    threshold_sha = verify_recorded_file(
        threshold_path, expected_threshold_hash, "Threshold artifact"
    )
    expected_context_hash = str(dataset_manifest["evaluation_context"]["sha256"])
    context_sha = verify_recorded_file(
        context_path, expected_context_hash, "Evaluation context"
    )

    with np.load(threshold_path, allow_pickle=False) as archive:
        saved_thresholds = np.asarray(archive["values"])
        threshold_protocol = str(archive["protocol_fingerprint"].item())
        threshold_fingerprint = str(archive["threshold_fingerprint"].item())
    if not np.array_equal(saved_thresholds, thresholds):
        raise ValueError("Saved thresholds do not exactly match raw training-target p90")

    with np.load(context_path, allow_pickle=False) as archive:
        context_target = np.asarray(archive["target"])
        context_event = np.asarray(archive["event"], dtype=bool)
        context_valid = np.asarray(archive["valid"], dtype=bool)
        evaluation_fingerprint = str(archive["evaluation_fingerprint"].item())
        context_protocol = str(archive["protocol_fingerprint"].item())

    protocol_fingerprint = str(dataset_manifest["protocol_fingerprint"])
    if threshold_protocol != protocol_fingerprint or context_protocol != protocol_fingerprint:
        raise ValueError("Threshold/context protocol fingerprint mismatch")
    if context_target.shape != y_test.shape:
        raise ValueError("Evaluation context shape does not match raw test targets")
    # Inverse-transform capture is float32 and can differ by a few ULPs from
    # raw float64 targets.  Event construction above intentionally uses raw y.
    if not np.allclose(context_target, y_test, rtol=0.0, atol=2e-4):
        raise ValueError("Evaluation context targets do not match the raw test split")
    if not np.array_equal(context_valid, valid):
        raise ValueError("Evaluation context valid mask does not match raw test targets")
    if not np.array_equal(context_event, events):
        raise ValueError("Evaluation context event mask does not match raw train/test data")
    if list(y_test.shape) != dataset_manifest["split_shapes"]["test"]:
        raise ValueError("Raw test shape does not match the dataset manifest")

    provenance = {
        "train_path": str(train_path),
        "train_sha256": train_sha,
        "test_path": str(test_path),
        "test_sha256": test_sha,
        "threshold_path": str(threshold_path),
        "threshold_sha256": threshold_sha,
        "threshold_fingerprint": threshold_fingerprint,
        "evaluation_context_path": str(context_path),
        "evaluation_context_sha256": context_sha,
        "evaluation_fingerprint": evaluation_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
    }
    return y_test, thresholds, events, provenance


def prediction_paths(
    dataset_results_dir: Path, variant: str, expected_seeds: Iterable[int]
) -> dict[int, Path]:
    directory = dataset_results_dir / "predictions" / variant
    found: dict[int, Path] = {}
    if directory.is_dir():
        for path in directory.glob("seed_*.npz"):
            match = SEED_PATTERN.match(path.name)
            if match:
                found[int(match.group(1))] = path
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    if tuple(sorted(found)) != expected:
        raise ValueError(
            f"Expected exactly seeds {expected} for {variant}; found {tuple(sorted(found))}"
        )
    return found


def local_checkpoint_path(
    dataset_results_dir: Path, variant: str, seed: int, unit: dict[str, Any]
) -> Path:
    recorded = Path(str(unit["checkpoint"]["path"]))
    if recorded.is_file():
        return recorded
    return require_file(dataset_results_dir / "checkpoints" / variant / f"seed_{seed}.pth")


def load_predictions(
    results_dir: Path,
    dataset: str,
    expected_seeds: tuple[int, ...],
    expected_shape: tuple[int, ...],
    context_sha256: str,
    protocol_fingerprint: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset_results_dir = results_dir / dataset
    predictions: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}

    for label, variant in MODEL_VARIANTS.items():
        paths = prediction_paths(dataset_results_dir, variant, expected_seeds)
        seeded = []
        model_records = []
        for seed in expected_seeds:
            path = paths[seed]
            unit_path = require_file(
                dataset_results_dir / "units" / variant / f"seed_{seed}.json"
            )
            unit = read_json(unit_path)
            if unit.get("status") != "complete":
                raise ValueError(f"Incomplete result unit: {unit_path}")
            if (
                unit.get("dataset") != dataset
                or unit.get("variant") != variant
                or int(unit.get("seed", -1)) != seed
            ):
                raise ValueError(f"Result-unit identity mismatch: {unit_path}")
            if unit.get("evaluation_protocol_fingerprint") != protocol_fingerprint:
                raise ValueError(f"Result-unit protocol mismatch: {unit_path}")

            prediction_sha = sha256_file(path)
            if prediction_sha != unit["prediction"]["sha256"]:
                raise ValueError(f"Prediction hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                prediction = np.asarray(archive["prediction"])
                linked_context_sha = str(
                    archive["evaluation_context_sha256"].item()
                )
            if linked_context_sha != context_sha256:
                raise ValueError(f"Prediction is linked to a different context: {path}")
            if prediction.shape != expected_shape or not np.all(np.isfinite(prediction)):
                raise ValueError(f"Invalid prediction array: {path}")

            checkpoint_path = local_checkpoint_path(
                dataset_results_dir, variant, seed, unit
            )
            checkpoint_sha = sha256_file(checkpoint_path)
            if checkpoint_sha != unit["checkpoint"]["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")

            seeded.append(prediction)
            model_records.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "prediction_path": str(path.resolve()),
                    "prediction_sha256": prediction_sha,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                    "unit_manifest_path": str(unit_path.resolve()),
                }
            )
        predictions[label] = np.stack(seeded, axis=0)
        provenance[label] = model_records
    return predictions, provenance


def centered_window(length: int, center: int, window_length: int) -> tuple[int, int]:
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer of at least 3")
    if window_length > length:
        raise ValueError("window_length cannot exceed the test split length")
    if center < 0 or center >= length:
        raise ValueError("center index lies outside the sequence")
    start = max(0, min(center - window_length // 2, length - window_length))
    return start, start + window_length


def select_maximum_events(
    target: np.ndarray, events: np.ndarray, window_length: int
) -> tuple[Selection, ...]:
    """Select each flow's largest observed test event without using predictions."""

    if target.shape != events.shape:
        raise ValueError("Target and event arrays must share a shape")
    selections = []
    for flow_index in range(target.shape[-1]):
        candidates = np.where(events[..., flow_index], target[..., flow_index], -np.inf)
        if not np.any(np.isfinite(candidates)):
            raise ValueError(f"No high-flow test events for {FLOW_NAMES[flow_index]}")
        # np.argmax gives the first row-major index on ties, making tie-breaking
        # deterministic and independent of both prediction sets.
        test_index, horizon_index, node_index = np.unravel_index(
            int(np.argmax(candidates)), candidates.shape
        )
        start, end = centered_window(target.shape[0], test_index, window_length)
        selections.append(
            Selection(
                flow_index=flow_index,
                test_index=int(test_index),
                horizon_index=int(horizon_index),
                node_index=int(node_index),
                window_start=start,
                window_end=end,
            )
        )
    return tuple(selections)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive [start, end] runs from a one-dimensional mask."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional event mask")
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def event_window_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    events: np.ndarray,
    selection: Selection,
) -> np.ndarray:
    sl = slice(selection.window_start, selection.window_end)
    truth = target[
        sl, selection.horizon_index, selection.node_index, selection.flow_index
    ]
    event_mask = events[
        sl, selection.horizon_index, selection.node_index, selection.flow_index
    ]
    seeded = prediction[
        :, sl, selection.horizon_index, selection.node_index, selection.flow_index
    ]
    return np.mean(np.abs(seeded[:, event_mask] - truth[event_mask]), axis=1)


def selection_record(
    dataset: str,
    selection: Selection,
    target: np.ndarray,
    thresholds: np.ndarray,
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    idx = (
        selection.test_index,
        selection.horizon_index,
        selection.node_index,
        selection.flow_index,
    )
    observed = float(target[idx])
    threshold = float(
        thresholds[
            selection.horizon_index, selection.node_index, selection.flow_index
        ]
    )
    record: dict[str, Any] = {
        "dataset": dataset,
        "flow": FLOW_NAMES[selection.flow_index],
        "test_index_zero_based": selection.test_index,
        "horizon_index_zero_based": selection.horizon_index,
        "node_index_zero_based": selection.node_index,
        "region_number_one_based": selection.node_index + 1,
        "window_start_inclusive": selection.window_start,
        "window_end_exclusive": selection.window_end,
        "observed_peak": observed,
        "training_target_p90": threshold,
        "event_test_indices_in_window": (
            np.flatnonzero(
                events[
                    selection.window_start : selection.window_end,
                    selection.horizon_index,
                    selection.node_index,
                    selection.flow_index,
                ]
            )
            + selection.window_start
        ).astype(int).tolist(),
        "models": {},
    }
    errors: dict[str, np.ndarray] = {}
    for label, seeded in predictions.items():
        values = seeded[:, idx[0], idx[1], idx[2], idx[3]]
        absolute_errors = np.abs(values - observed)
        errors[label] = absolute_errors
        window_mae = event_window_mae(seeded, target, events, selection)
        record["models"][label] = {
            "variant": MODEL_VARIANTS[label],
            "peak_predictions_by_seed": values.astype(float).tolist(),
            "peak_prediction_mean": float(np.mean(values)),
            "peak_prediction_population_sd": float(np.std(values, ddof=0)),
            "peak_absolute_errors_by_seed": absolute_errors.astype(float).tolist(),
            "peak_absolute_error_of_seed_mean": float(abs(np.mean(values) - observed)),
            "event_window_mae_by_seed": window_mae.astype(float).tolist(),
            "event_window_mae_mean": float(np.mean(window_mae)),
        }

    base_error = float(record["models"]["STE-Base"]["peak_absolute_error_of_seed_mean"])
    hyb_error = float(record["models"]["Hyb-STEX"]["peak_absolute_error_of_seed_mean"])
    record["peak_absolute_error_reduction"] = base_error - hyb_error
    record["peak_absolute_error_reduction_percent"] = (
        100.0 * (base_error - hyb_error) / base_error
    )
    record["hybstex_improves_peak_in_every_seed"] = bool(
        np.all(errors["Hyb-STEX"] < errors["STE-Base"])
    )
    return record


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def make_figure(
    dataset: str,
    target: np.ndarray,
    thresholds: np.ndarray,
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
    selections: tuple[Selection, ...],
) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.25, 6.9), sharex=True)
    hours_per_slot = 24.0 / SLOTS_PER_DAY[dataset]

    for panel_index, (axis, selection) in enumerate(zip(axes, selections)):
        sl = slice(selection.window_start, selection.window_end)
        relative_slots = (
            np.arange(selection.window_start, selection.window_end)
            - selection.test_index
        )
        x = relative_slots * hours_per_slot
        half_slot = hours_per_slot / 2.0
        truth = target[
            sl, selection.horizon_index, selection.node_index, selection.flow_index
        ]
        event_mask = events[
            sl, selection.horizon_index, selection.node_index, selection.flow_index
        ]
        threshold = float(
            thresholds[
                selection.horizon_index,
                selection.node_index,
                selection.flow_index,
            ]
        )

        for run_start, run_end in contiguous_true_runs(event_mask):
            axis.axvspan(
                x[run_start] - half_slot,
                x[run_end] + half_slot,
                color=EVENT_COLOR,
                alpha=0.16,
                linewidth=0,
                zorder=0,
            )

        axis.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linewidth=1.25,
            linestyle=(0, (4, 2.5)),
            zorder=1,
        )
        axis.plot(x, truth, color=OBSERVED_COLOR, linewidth=2.25, zorder=4)

        for label in MODEL_VARIANTS:
            seeded = predictions[label][
                :, sl, selection.horizon_index, selection.node_index, selection.flow_index
            ]
            mean = np.mean(seeded, axis=0)
            spread = np.std(seeded, axis=0, ddof=0)
            color = MODEL_COLORS[label]
            axis.fill_between(
                x,
                mean - spread,
                mean + spread,
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            axis.plot(x, mean, color=color, linewidth=1.9, zorder=3)

        record = selection_record(
            dataset, selection, target, thresholds, events, predictions
        )
        base_error = record["models"]["STE-Base"][
            "peak_absolute_error_of_seed_mean"
        ]
        hyb_error = record["models"]["Hyb-STEX"][
            "peak_absolute_error_of_seed_mean"
        ]
        reduction = record["peak_absolute_error_reduction_percent"]
        axis.text(
            0.015,
            0.95,
            (
                f"Peak |error|: STE-Base {base_error:.1f}; "
                f"Hyb-STEX {hyb_error:.1f} "
                f"({reduction:.1f}% lower)"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
            color="#374151",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "linewidth": 0.7,
                "alpha": 0.92,
            },
            zorder=6,
        )

        panel_letter = chr(ord("a") + panel_index)
        axis.set_title(
            (
                f"({panel_letter}) {FLOW_LABELS[selection.flow_index]} | "
                f"region {selection.node_index + 1} "
                f"(node index {selection.node_index}) | "
                f"training p90 = {threshold:g}"
            ),
            loc="left",
            fontweight="semibold",
            pad=7,
        )
        axis.set_ylabel("Flow")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.65, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0)

    axes[-1].set_xlabel("Time relative to the selected test maximum (hours)")
    axes[-1].set_xticks(np.arange(-12, 12.1, 4))

    observed_handle = Line2D(
        [0], [0], color=OBSERVED_COLOR, linewidth=2.25, label="Observed"
    )
    base_handle = Line2D(
        [0], [0], color=MODEL_COLORS["STE-Base"], linewidth=1.9, label="STE-Base"
    )
    hyb_handle = Line2D(
        [0], [0], color=MODEL_COLORS["Hyb-STEX"], linewidth=1.9, label="Hyb-STEX"
    )
    threshold_handle = Line2D(
        [0],
        [0],
        color=THRESHOLD_COLOR,
        linewidth=1.25,
        linestyle=(0, (4, 2.5)),
        label="Training-target 90th percentile",
    )
    event_handle = Patch(
        facecolor=EVENT_COLOR,
        edgecolor="none",
        alpha=0.30,
        label="High-flow test observation",
    )
    # Matplotlib fills multi-row legends column-first.  This ordering keeps
    # Observed, STE-Base, and Hyb-STEX together on the first visual row.
    legend_handles = [
        observed_handle,
        threshold_handle,
        base_handle,
        event_handle,
        hyb_handle,
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.7,
    )
    fig.suptitle(
        "Test-set forecasts at the largest observed high-flow events",
        fontsize=12.5,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"{DATASET_LABELS[dataset]} | mean prediction and population SD "
            "across three matched seeds"
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.012,
        (
            "Selection uses observed test targets only: the maximum high-flow "
            "observation in each flow; model predictions are not used."
        ),
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#4B5563",
    )
    fig.subplots_adjust(top=0.79, bottom=0.10, left=0.105, right=0.985, hspace=0.32)
    return fig


def write_metadata(
    path: Path,
    dataset: str,
    run_manifest_path: Path,
    run_manifest: dict[str, Any],
    dataset_manifest_path: Path,
    dataset_manifest: dict[str, Any],
    source_provenance: dict[str, Any],
    model_provenance: dict[str, Any],
    records: list[dict[str, Any]],
    outputs: dict[str, Path],
) -> None:
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "data_policy": {
            "figure_samples": "test only",
            "threshold_fit_split": "train only",
            "validation_split_read": False,
            "selection_rule": (
                "For each flow independently, select the first row-major "
                "argmax of observed Y among strict test events "
                "E=(Y>5) AND (Y>node/flow training-target p90). Use a fixed "
                "centered window. Predictions are loaded only after selection."
            ),
            "seed_aggregation": "mean and population SD across all matched seeds",
        },
        "model_mapping": MODEL_VARIANTS,
        "protocol": run_manifest["protocol"],
        "run_manifest": {
            "path": str(run_manifest_path.resolve()),
            "sha256": sha256_file(run_manifest_path),
            "fingerprint": run_manifest["fingerprint"],
            "seeds": run_manifest["seeds"],
            "scaler_policy": run_manifest["scaler_policy"],
        },
        "dataset_manifest": {
            "path": str(dataset_manifest_path.resolve()),
            "sha256": sha256_file(dataset_manifest_path),
            "fingerprint": dataset_manifest["fingerprint"],
        },
        "source_data": source_provenance,
        "model_artifacts": model_provenance,
        "selections": records,
        "outputs": {
            name: {"path": str(output.resolve()), "sha256": sha256_file(output)}
            for name, output in outputs.items()
        },
    }
    def numpy_json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=numpy_json_default,
        )
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.dpi < 150:
        raise ValueError("dpi must be at least 150")

    data_dir = resolve_repo_path(args.data_dir)
    results_dir = resolve_repo_path(args.results_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest_path = require_file(results_dir / "run_manifest.json")
    run_manifest = read_json(run_manifest_path)
    validate_protocol(run_manifest.get("protocol", {}), run_manifest_path)
    if run_manifest.get("scaler_policy") != "train_only":
        raise ValueError("Expected the canonical train-only scaling result suite")
    expected_seeds = tuple(int(seed) for seed in run_manifest.get("seeds", []))
    if expected_seeds != (1, 2, 3):
        raise ValueError(f"Expected matched seeds (1, 2, 3); found {expected_seeds}")

    dataset_manifest_path = require_file(
        results_dir / args.dataset / "dataset_manifest.json"
    )
    dataset_manifest = read_json(dataset_manifest_path)
    validate_protocol(dataset_manifest.get("protocol", {}), dataset_manifest_path)
    if dataset_manifest.get("dataset") != args.dataset:
        raise ValueError("Dataset manifest identity mismatch")

    # Select maxima from raw train/test data before any prediction arrays are
    # loaded.  This code path never constructs or opens val.npz.
    target, thresholds, events, source_provenance = load_raw_targets(
        data_dir, results_dir, args.dataset, run_manifest, dataset_manifest
    )
    selections = select_maximum_events(target, events, args.window_length)

    predictions, model_provenance = load_predictions(
        results_dir=results_dir,
        dataset=args.dataset,
        expected_seeds=expected_seeds,
        expected_shape=target.shape,
        context_sha256=source_provenance["evaluation_context_sha256"],
        protocol_fingerprint=source_provenance["protocol_fingerprint"],
    )
    records = [
        selection_record(
            args.dataset, selection, target, thresholds, events, predictions
        )
        for selection in selections
    ]
    for record in records:
        if not record["hybstex_improves_peak_in_every_seed"]:
            raise ValueError(
                "The prediction-blind maximum event no longer improves in every "
                "matched seed. Refusing to search for a more favorable example."
            )

    figure = make_figure(
        args.dataset, target, thresholds, events, predictions, selections
    )
    png_path = output_dir / f"{args.basename}.png"
    pdf_path = output_dir / f"{args.basename}.pdf"
    figure.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
        metadata={
            "Title": "Hyb-STEX versus STE-Base at maximum observed test events",
            "Description": "Prediction-blind test-only qualitative comparison",
        },
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": "Hyb-STEX versus STE-Base at maximum observed test events",
            "Subject": "Prediction-blind test-only qualitative comparison",
            "Creator": "scripts/plot_test_high_flow_comparison.py",
        },
    )
    plt.close(figure)

    metadata_path = output_dir / f"{args.basename}_metadata.json"
    write_metadata(
        metadata_path,
        args.dataset,
        run_manifest_path,
        run_manifest,
        dataset_manifest_path,
        dataset_manifest,
        source_provenance,
        model_provenance,
        records,
        {"png": png_path, "pdf": pdf_path},
    )

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {metadata_path}")
    for record in records:
        print(
            f"{record['flow']}: test={record['test_index_zero_based']}, "
            f"node={record['node_index_zero_based']}, "
            f"observed={record['observed_peak']:.1f}, "
            f"p90={record['training_target_p90']:.1f}, "
            f"peak error reduction="
            f"{record['peak_absolute_error_reduction_percent']:.1f}%"
        )


if __name__ == "__main__":
    main()

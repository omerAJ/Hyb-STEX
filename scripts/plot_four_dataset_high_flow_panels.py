#!/usr/bin/env python3
"""Export four clean test-event panels and one separate common legend.

For each dataset, the selected example is the largest observed high-flow event
in the raw chronological test split, considering every node and both flow
channels.  Selection occurs for all datasets before any prediction is loaded.
High-flow events use strict ``Y > 5`` and ``Y > training-target p90`` masks.
No validation observation is used for selection or plotted.  Inference
reconstructs the train+validation input scaler used by the already-trained
checkpoint suite; this script never trains or updates a model.

Each panel contains only the observed series, the three-seed mean STE-Base and
Hyb-STEX predictions, the selected node/flow training-p90 line, axes, grid, and
subtle data-derived event shading.  Titles, legends, annotations, captions,
panel letters, markers, and uncertainty ribbons are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.dataloader import get_dataloader
from lib.utils import load_graph
from model.models import STSSL

from plot_test_high_flow_comparison import (
    DATASET_LABELS,
    EVENT_COLOR,
    EXPECTED_PROTOCOL,
    FLOW_NAMES,
    MODEL_COLORS,
    OBSERVED_COLOR,
    ROOT,
    SLOTS_PER_DAY,
    THRESHOLD_COLOR,
    centered_window,
    contiguous_true_runs,
    load_raw_targets,
    read_json,
    require_file,
    resolve_repo_path,
    sha256_file,
    validate_protocol,
)


DATASET_ORDER = ("NYCBike1", "NYCBike2", "NYCTaxi", "BJTaxi")
FILE_STEMS = {
    "NYCBike1": "nyc_bike_1_high_flow_panel",
    "NYCBike2": "nyc_bike_2_high_flow_panel",
    "NYCTaxi": "nyc_taxi_high_flow_panel",
    "BJTaxi": "bj_taxi_high_flow_panel",
}
EXPECTED_SEEDS = (1, 2, 3)
MODEL_CONFIG_DEFAULTS = {
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
    "scaler_fit": "train_val",
    "max_train_batches": None,
    "max_eval_batches": None,
}
EVALUATION_MODELS = {
    "STE-Base": {
        "suite_variant": "A_base_mae",
        "ablation_mode": "original",
        "prediction_phase": "pred",
        "event_loss_weight": 0.0,
        "checkpoint_basename": "best_model_pred.pth",
    },
    "Hyb-STEX": {
        "suite_variant": "D_frozen_residual_event_weighted",
        "ablation_mode": "event_weighted_ungated_residual",
        "prediction_phase": "pred_2",
        "event_loss_weight": 0.25,
        "checkpoint_basename": "best_model_pred_2.pth",
    },
}


@dataclass(frozen=True)
class PanelSelection:
    dataset: str
    test_index: int
    horizon_index: int
    node_index: int
    flow_index: int
    window_start: int
    window_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one prediction-blind high-flow test panel per dataset and "
            "a separate common legend."
        )
    )
    parser.add_argument("--data-dir", default="preprocessed_data")
    parser.add_argument(
        "--selection-context-dir",
        default="corrected_valid_target_p90_results",
        help=(
            "Provenance-checked raw train/test targets and train-p90 context; "
            "used only for prediction-blind selection and verification."
        ),
    )
    parser.add_argument(
        "--model-results-dir",
        default="stssl_protocol_full_paper_results",
        help="Complete four-dataset checkpoint suite used for inference only.",
    )
    parser.add_argument(
        "--output-dir", default="publication_four_dataset_high_flow_panels"
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: CUDA when available).",
    )
    return parser.parse_args()


def select_global_maximum_event(
    dataset: str, target: np.ndarray, events: np.ndarray
) -> PanelSelection:
    """Select the deterministic global observed-event maximum for a dataset."""

    if target.shape != events.shape:
        raise ValueError("Target and event arrays must share a shape")
    candidates = np.where(events, target, -np.inf)
    if not np.any(np.isfinite(candidates)):
        raise ValueError(f"No high-flow test event exists for {dataset}")
    test_index, horizon_index, node_index, flow_index = np.unravel_index(
        int(np.argmax(candidates)), candidates.shape
    )
    # One day centered on the maximum, including both endpoints.  This gives
    # 25 samples for hourly NYC Bike 1 and 49 for the half-hourly datasets.
    window_length = SLOTS_PER_DAY[dataset] + 1
    start, end = centered_window(target.shape[0], int(test_index), window_length)
    return PanelSelection(
        dataset=dataset,
        test_index=int(test_index),
        horizon_index=int(horizon_index),
        node_index=int(node_index),
        flow_index=int(flow_index),
        window_start=start,
        window_end=end,
    )


def configure_panel_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def make_panel(
    selection: PanelSelection,
    target: np.ndarray,
    thresholds: np.ndarray,
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> plt.Figure:
    """Create one axes-only panel with no internal descriptive furniture."""

    configure_panel_style()
    figure, axis = plt.subplots(figsize=(5.4, 3.3))
    sl = slice(selection.window_start, selection.window_end)
    hours_per_slot = 24.0 / SLOTS_PER_DAY[selection.dataset]
    relative_hours = (
        np.arange(selection.window_start, selection.window_end)
        - selection.test_index
    ) * hours_per_slot
    half_slot = hours_per_slot / 2.0
    truth = target[
        sl,
        selection.horizon_index,
        selection.node_index,
        selection.flow_index,
    ]
    event_mask = events[
        sl,
        selection.horizon_index,
        selection.node_index,
        selection.flow_index,
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
            relative_hours[run_start] - half_slot,
            relative_hours[run_end] + half_slot,
            color=EVENT_COLOR,
            alpha=0.14,
            linewidth=0,
            zorder=0,
        )

    axis.axhline(
        threshold,
        color=THRESHOLD_COLOR,
        linewidth=1.15,
        linestyle=(0, (4, 2.5)),
        zorder=1,
    )
    axis.plot(relative_hours, truth, color=OBSERVED_COLOR, linewidth=2.15, zorder=4)

    plotted_values = [truth, np.asarray([threshold])]
    for label in EVALUATION_MODELS:
        seeded = predictions[label][
            :,
            sl,
            selection.horizon_index,
            selection.node_index,
            selection.flow_index,
        ]
        mean_prediction = np.mean(seeded, axis=0)
        plotted_values.append(mean_prediction)
        axis.plot(
            relative_hours,
            mean_prediction,
            color=MODEL_COLORS[label],
            linewidth=1.8,
            zorder=3,
        )

    upper = max(float(np.max(values)) for values in plotted_values)
    axis.set_ylim(0, upper * 1.075 if upper > 0 else 1.0)
    axis.set_xlim(-12.0, 12.0)
    axis.set_xticks((-12, -6, 0, 6, 12))
    axis.set_xlabel("Time relative to test maximum (h)")
    axis.set_ylabel("Flow")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.65, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.97)
    return figure


def legend_handles() -> list[Any]:
    return [
        Line2D([0], [0], color=OBSERVED_COLOR, linewidth=2.15, label="Observed"),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["STE-Base"],
            linewidth=1.8,
            label="STE-Base",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["Hyb-STEX"],
            linewidth=1.8,
            label="Hyb-STEX",
        ),
        Line2D(
            [0],
            [0],
            color=THRESHOLD_COLOR,
            linewidth=1.15,
            linestyle=(0, (4, 2.5)),
            label="Training-target p90",
        ),
        Patch(
            facecolor=EVENT_COLOR,
            edgecolor="none",
            alpha=0.28,
            label="High-flow test observation",
        ),
    ]


def make_common_legend() -> plt.Figure:
    configure_panel_style()
    figure = plt.figure(figsize=(9.4, 0.62))
    figure.legend(
        handles=legend_handles(),
        loc="center",
        ncol=5,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.7,
        fontsize=9,
    )
    return figure


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        handle.write("\n")


def recorded_path(value: str | Path) -> Path:
    """Resolve a path recorded by the existing checkpoint suite."""

    path = Path(value)
    if path.is_file():
        return path.resolve()
    candidate = (ROOT / path).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(path)


def load_model_config(path: Path) -> dict[str, Any]:
    """Load model construction settings without importing a training runner."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    config = dict(MODEL_CONFIG_DEFAULTS)
    config.update(raw)
    return config


def verify_suite_manifest(
    model_results_dir: Path, dataset: str
) -> tuple[Path, dict[str, Any], Path, Path]:
    """Validate the complete, already-trained result suite for one dataset."""

    suite_dir = model_results_dir / dataset / "residual_schedule"
    manifest_path = require_file(suite_dir / "run_manifest.json")
    manifest = read_json(manifest_path)
    if tuple(int(seed) for seed in manifest.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS} in {manifest_path}")
    required_variants = {
        spec["suite_variant"] for spec in EVALUATION_MODELS.values()
    }
    missing = required_variants.difference(manifest.get("variants", []))
    if missing:
        raise ValueError(f"Missing variants {sorted(missing)} in {manifest_path}")
    expected_fields = {
        "scaler_fit": "train_val",
        "event_mask_protocol": "train_all_node_flow_p90_valid_v2",
        "event_label_source": "file_verified",
        "smoke_test": False,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Unexpected {key} in {manifest_path}: "
                f"expected {expected!r}, found {manifest.get(key)!r}"
            )

    config_records = manifest.get("configs", [])
    if len(config_records) != 1:
        raise ValueError(f"Expected exactly one config in {manifest_path}")
    config_path = recorded_path(str(config_records[0]["path"]))
    if config_path.stem != dataset:
        raise ValueError(f"Config identity mismatch for {dataset}: {config_path}")
    if sha256_file(config_path) != str(config_records[0]["sha256"]):
        raise ValueError(f"Config hash mismatch: {config_path}")

    # The model definition and evaluation loader must match the code recorded
    # by the checkpoint suite.  This is read-only provenance checking.
    for record in manifest.get("code", []):
        path = recorded_path(str(record["path"]))
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"Recorded inference code hash mismatch: {path}")

    graph_path = recorded_path(str(manifest["graph_file"]))
    metrics_path = require_file(suite_dir / "per_seed_metrics.csv")
    return manifest_path, manifest, config_path, metrics_path


def checkpoint_lookup(
    metrics_path: Path, suite_variant: str
) -> dict[int, Path]:
    """Read one checkpoint per matched seed from the existing result table."""

    checkpoints: dict[int, Path] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != suite_variant or row.get("flow") != "mean":
                continue
            seed = int(row["seed"])
            if seed in checkpoints:
                raise ValueError(
                    f"Duplicate {suite_variant} seed {seed} in {metrics_path}"
                )
            checkpoints[seed] = recorded_path(row["checkpoint"])
    if tuple(sorted(checkpoints)) != EXPECTED_SEEDS:
        raise ValueError(
            f"Expected matched seeds {EXPECTED_SEEDS} for {suite_variant}; "
            f"found {tuple(sorted(checkpoints))}"
        )
    return checkpoints


def inference_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def predict_checkpoint(
    base_config: dict[str, Any],
    dataloader: dict[str, Any],
    graph: torch.Tensor,
    graph_path: Path,
    checkpoint: Path,
    ablation_mode: str,
    prediction_phase: str,
    event_loss_weight: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a fixed checkpoint on the test loader without any training state."""

    config = dict(base_config)
    config.update(
        {
            "mode": "test",
            "device": device,
            "num_nodes": len(graph),
            "graph_file": str(graph_path),
            "ablation_mode": ablation_mode,
            "event_loss_weight": event_loss_weight,
            "bias_param_scope": "head_only",
            "phase3_mode": "original",
            "classification_loss_weight": 1.0,
            "boost_gate_scale": 1.0,
            "gate_floor": 0.0,
        }
    )
    model = STSSL(SimpleNamespace(**config)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Checkpoint has no model state: {checkpoint}")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for data, target, _event, _bias in dataloader["test"]:
            representation, representation_cls = model(data, graph)
            prediction = model.predict(
                representation, representation_cls, prediction_phase
            )
            predictions.append(prediction.detach().cpu())
            targets.append(target.detach().cpu())

    prediction_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    scaler = dataloader["scaler"]
    prediction = scaler.inverse_transform(prediction_tensor).numpy()
    target = scaler.inverse_transform(target_tensor).numpy()
    del model, state, prediction_tensor, target_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction, target


def load_suite_predictions(
    dataset: str,
    data_dir: Path,
    model_results_dir: Path,
    expected_target: np.ndarray,
    requested_device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate both models over test only using three fixed checkpoints each."""

    manifest_path, manifest, config_path, metrics_path = verify_suite_manifest(
        model_results_dir, dataset
    )
    base_config = load_model_config(config_path)
    if base_config.get("dataset") != dataset:
        raise ValueError(f"Loaded the wrong dataset config: {config_path}")
    graph_path = recorded_path(str(manifest["graph_file"]))
    device = inference_device(requested_device)
    batch_size = int(base_config["test_batch_size"])

    # This exactly reconstructs the input normalization used by the fixed
    # checkpoints.  Only dataloader['test'] is iterated for inference.
    dataloader = get_dataloader(
        str(data_dir),
        dataset,
        batch_size,
        batch_size,
        scalar_type="Standard",
        scaler_fit=str(manifest["scaler_fit"]),
        event_mask_protocol=str(manifest["event_mask_protocol"]),
        event_label_source=str(manifest["event_label_source"]),
        device="cpu",
    )
    graph = load_graph(graph_path, device=device)
    predictions: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {
        "suite_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "metrics_table": {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "graph": {
            "path": str(graph_path.resolve()),
            "sha256": sha256_file(graph_path),
        },
        "device": device,
        "operation": "checkpoint inference only",
        "scaler_fit": manifest["scaler_fit"],
        "event_mask_protocol": manifest["event_mask_protocol"],
        "figure_split": "test only",
        "models": {},
    }

    verified_target: np.ndarray | None = None
    for label, spec in EVALUATION_MODELS.items():
        seeded_predictions: list[np.ndarray] = []
        checkpoint_records: list[dict[str, Any]] = []
        checkpoints = checkpoint_lookup(metrics_path, spec["suite_variant"])
        for seed, checkpoint in sorted(checkpoints.items()):
            if checkpoint.name != spec["checkpoint_basename"]:
                raise ValueError(
                    f"Unexpected checkpoint phase for {label}: {checkpoint}"
                )
            print(
                f"[inference only] {dataset} {label} seed={seed} "
                f"checkpoint={checkpoint.name}",
                flush=True,
            )
            prediction, target = predict_checkpoint(
                base_config,
                dataloader,
                graph,
                graph_path,
                checkpoint,
                str(spec["ablation_mode"]),
                str(spec["prediction_phase"]),
                float(spec["event_loss_weight"]),
                device,
            )
            if prediction.shape != expected_target.shape:
                raise ValueError(
                    f"Unexpected prediction shape for {dataset}/{label}/seed {seed}: "
                    f"{prediction.shape} != {expected_target.shape}"
                )
            max_target_error = float(
                np.max(
                    np.abs(
                        target.astype(np.float64)
                        - expected_target.astype(np.float64)
                    )
                )
            )
            if max_target_error > 1e-3:
                raise ValueError(
                    f"Inverse-scaled test target mismatch for {dataset}: "
                    f"max absolute error {max_target_error:g}"
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError(f"Non-finite prediction from {checkpoint}")
            if verified_target is None:
                verified_target = target
            seeded_predictions.append(prediction)
            checkpoint_records.append(
                {
                    "seed": seed,
                    "path": str(checkpoint.resolve()),
                    "sha256": sha256_file(checkpoint),
                }
            )
        predictions[label] = np.stack(seeded_predictions, axis=0)
        provenance["models"][label] = {
            **spec,
            "aggregation": "arithmetic mean over matched seeds",
            "checkpoints": checkpoint_records,
        }

    del dataloader, graph, verified_target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, provenance


def selection_report_row(
    selection: PanelSelection,
    target: np.ndarray,
    thresholds: np.ndarray,
    panel_png: Path,
    panel_pdf: Path,
) -> dict[str, Any]:
    index = (
        selection.test_index,
        selection.horizon_index,
        selection.node_index,
        selection.flow_index,
    )
    hours_per_slot = 24.0 / SLOTS_PER_DAY[selection.dataset]
    return {
        "dataset": DATASET_LABELS[selection.dataset],
        "dataset_id": selection.dataset,
        "node_index_zero_based": selection.node_index,
        "region_number_one_based": selection.node_index + 1,
        "flow": FLOW_NAMES[selection.flow_index],
        "training_target_p90": float(
            thresholds[
                selection.horizon_index,
                selection.node_index,
                selection.flow_index,
            ]
        ),
        "observed_test_maximum": float(target[index]),
        "peak_test_index_zero_based": selection.test_index,
        "window_start_test_index_inclusive": selection.window_start,
        "window_end_test_index_inclusive": selection.window_end - 1,
        "window_start_hours_relative_to_peak": (
            selection.window_start - selection.test_index
        )
        * hours_per_slot,
        "window_end_hours_relative_to_peak": (
            selection.window_end - 1 - selection.test_index
        )
        * hours_per_slot,
        "samples_in_window": selection.window_end - selection.window_start,
        "panel_png": str(panel_png.resolve()),
        "panel_pdf": str(panel_pdf.resolve()),
    }


def main() -> None:
    args = parse_args()
    if args.dpi < 150:
        raise ValueError("dpi must be at least 150")
    data_dir = resolve_repo_path(args.data_dir)
    selection_context_dir = resolve_repo_path(args.selection_context_dir)
    model_results_dir = resolve_repo_path(args.model_results_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest_path = require_file(selection_context_dir / "run_manifest.json")
    run_manifest = read_json(run_manifest_path)
    validate_protocol(run_manifest.get("protocol", {}), run_manifest_path)
    if run_manifest.get("scaler_policy") != "train_only":
        raise ValueError("Expected the canonical train-only selection context")

    # Pass 1: load raw train/test arrays, validate contexts, and freeze every
    # selection before predictions are available to the process.
    dataset_inputs: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        dataset_manifest_path = require_file(
            selection_context_dir / dataset / "dataset_manifest.json"
        )
        dataset_manifest = read_json(dataset_manifest_path)
        validate_protocol(dataset_manifest.get("protocol", {}), dataset_manifest_path)
        if dataset_manifest.get("dataset") != dataset:
            raise ValueError(f"Dataset manifest identity mismatch for {dataset}")
        target, thresholds, events, source_provenance = load_raw_targets(
            data_dir,
            selection_context_dir,
            dataset,
            run_manifest,
            dataset_manifest,
        )
        dataset_inputs[dataset] = {
            "dataset_manifest_path": dataset_manifest_path,
            "dataset_manifest": dataset_manifest,
            "target": target,
            "thresholds": thresholds,
            "events": events,
            "source_provenance": source_provenance,
            "selection": select_global_maximum_event(dataset, target, events),
        }

    # Pass 2: only after all target-only selections are fixed, load predictions.
    for dataset, inputs in dataset_inputs.items():
        predictions, model_provenance = load_suite_predictions(
            dataset=dataset,
            data_dir=data_dir,
            model_results_dir=model_results_dir,
            expected_target=inputs["target"],
            requested_device=args.device,
        )
        inputs["predictions"] = predictions
        inputs["model_provenance"] = model_provenance

    report_rows: list[dict[str, Any]] = []
    output_records: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        inputs = dataset_inputs[dataset]
        selection = inputs["selection"]
        figure = make_panel(
            selection,
            inputs["target"],
            inputs["thresholds"],
            inputs["events"],
            inputs["predictions"],
        )
        stem = FILE_STEMS[dataset]
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        figure.savefig(
            png_path,
            dpi=args.dpi,
            metadata={
                "Title": f"{DATASET_LABELS[dataset]} maximum observed test event",
                "Description": "Prediction-blind test-only high-flow panel",
            },
        )
        figure.savefig(
            pdf_path,
            metadata={
                "Title": f"{DATASET_LABELS[dataset]} maximum observed test event",
                "Subject": "Prediction-blind test-only high-flow panel",
                "Creator": "scripts/plot_four_dataset_high_flow_panels.py",
            },
        )
        plt.close(figure)
        report_rows.append(
            selection_report_row(
                selection,
                inputs["target"],
                inputs["thresholds"],
                png_path,
                pdf_path,
            )
        )
        output_records[dataset] = {
            "selection": asdict(selection),
            "panel_png": {
                "path": str(png_path.resolve()),
                "sha256": sha256_file(png_path),
            },
            "panel_pdf": {
                "path": str(pdf_path.resolve()),
                "sha256": sha256_file(pdf_path),
            },
            "source_provenance": inputs["source_provenance"],
            "model_provenance": inputs["model_provenance"],
        }

    legend = make_common_legend()
    legend_png = output_dir / "common_legend.png"
    legend_pdf = output_dir / "common_legend.pdf"
    legend.savefig(
        legend_png,
        dpi=args.dpi,
        metadata={"Title": "Common legend for high-flow test panels"},
    )
    legend.savefig(
        legend_pdf,
        metadata={
            "Title": "Common legend for high-flow test panels",
            "Creator": "scripts/plot_four_dataset_high_flow_panels.py",
        },
    )
    plt.close(legend)

    report_csv = output_dir / "panel_selections.csv"
    write_csv(report_csv, report_rows)
    metadata_path = output_dir / "panel_provenance.json"
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "selection_policy": {
                "figure_samples": "test only",
                "threshold_fit_split": "train only",
                "selection_inputs": ["raw training targets", "raw test targets"],
                "selection_uses_predictions": False,
                "validation_samples_selected_or_plotted": False,
                "rule": (
                    "Within each dataset, select the first row-major argmax "
                    "of observed raw test Y over all strict events, nodes, and "
                    "flow channels before loading any prediction."
                ),
                "window": "24 hours centered on the selected event",
            },
            "model_mapping": EVALUATION_MODELS,
            "seeds": list(EXPECTED_SEEDS),
            "selection_context_manifest": {
                "path": str(run_manifest_path.resolve()),
                "sha256": sha256_file(run_manifest_path),
                "fingerprint": run_manifest["fingerprint"],
                "protocol": EXPECTED_PROTOCOL,
            },
            "evaluation_policy": {
                "operation": "inference only from existing checkpoints",
                "training_performed": False,
                "optimizer_created": False,
                "backward_pass_performed": False,
                "prediction_aggregation": "arithmetic mean over seeds 1, 2, and 3",
                "model_suite_root": str(model_results_dir.resolve()),
                "checkpoint_scaler_fit": "train_val",
                "note": (
                    "Validation inputs are read only to reconstruct the fixed "
                    "input scaler used by the existing checkpoints; no validation "
                    "observation is selected, inferred for the figure, or plotted."
                ),
            },
            "datasets": output_records,
            "common_legend": {
                "png": {
                    "path": str(legend_png.resolve()),
                    "sha256": sha256_file(legend_png),
                },
                "pdf": {
                    "path": str(legend_pdf.resolve()),
                    "sha256": sha256_file(legend_pdf),
                },
            },
            "selection_report": str(report_csv.resolve()),
        },
    )

    print(f"Wrote four panels and common legend to {output_dir}")
    print(f"Wrote selection report to {report_csv}")
    print(f"Wrote provenance to {metadata_path}")
    for row in report_rows:
        print(
            f"{row['dataset']}: node={row['node_index_zero_based']}, "
            f"flow={row['flow']}, p90={row['training_target_p90']:g}, "
            f"test window={row['window_start_test_index_inclusive']}.."
            f"{row['window_end_test_index_inclusive']}"
        )


if __name__ == "__main__":
    main()

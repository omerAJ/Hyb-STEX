#!/usr/bin/env python3
"""Aggregate all-node, test-only distribution and high-flow error KDEs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import gaussian_kde

from plot_figure1_style_four_dataset_composite import (
    DATA_DIR, DATASET_LABELS, DATASET_ORDER, HYB_STEX, MODEL_RESULTS_DIR,
    OBSERVED, P90, ROOT, SELECTION_CONTEXT_DIR, STE_BASE, configure_style,
)
from plot_four_dataset_high_flow_panels import load_suite_predictions
from plot_test_high_flow_comparison import (
    load_raw_targets, read_json, require_file, validate_protocol,
)


OUTPUT_DIR = ROOT / "publication_aggregate_test_kde"
OUTPUT_STEM = "hybstex_all_node_test_distribution_and_error_kdes"
MAX_KDE_SAMPLES = 30_000


def representative_sample(values: np.ndarray) -> np.ndarray:
    """Deterministic, evenly spaced subset used only to evaluate a KDE curve."""

    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size <= MAX_KDE_SAMPLES:
        return values
    return values[np.linspace(0, values.size - 1, MAX_KDE_SAMPLES, dtype=int)]


def curve(values: np.ndarray, quantile_limits: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = representative_sample(values)
    if quantile_limits is None:
        lo, hi = float(np.min(values)), float(np.max(values))
    else:
        lo, hi = (float(np.quantile(values, q)) for q in quantile_limits)
    pad = max((hi - lo) * 0.08, 0.02)
    x = np.linspace(lo - pad, hi + pad, 400)
    return x, gaussian_kde(values)(x)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = require_file(SELECTION_CONTEXT_DIR / "run_manifest.json")
    context_manifest = read_json(manifest_path)
    validate_protocol(context_manifest.get("protocol", {}), manifest_path)

    inputs: dict[str, dict] = {}
    for dataset in DATASET_ORDER:
        manifest = read_json(require_file(SELECTION_CONTEXT_DIR / dataset / "dataset_manifest.json"))
        target, thresholds, events, _ = load_raw_targets(
            DATA_DIR, SELECTION_CONTEXT_DIR, dataset, context_manifest, manifest
        )
        predictions, _ = load_suite_predictions(dataset, DATA_DIR, MODEL_RESULTS_DIR, target, "auto")
        inputs[dataset] = {
            "target": target,
            "thresholds": thresholds,
            "events": events,
            "predictions": {label: np.mean(predictions[label], axis=0) for label in ("STE-Base", "Hyb-STEX")},
        }

    configure_style()
    figure, axes = plt.subplots(4, 2, figsize=(8.2, 10.2))
    figure.text(0.29, 0.902, "All test values / node-flow training p90", ha="center", fontsize=11)
    figure.text(0.75, 0.902, "High-flow normalized absolute error", ha="center", fontsize=11)
    metadata: dict[str, dict[str, int | str]] = {}

    for row, dataset in enumerate(DATASET_ORDER):
        item = inputs[dataset]
        threshold = item["thresholds"]
        valid = threshold > 0
        denominator = np.where(valid, threshold, 1.0)[None, ...]
        truth = item["target"]
        event_mask = item["events"] & valid[None, ...]
        # Pool every valid node/horizon/channel combination within this dataset.
        # The error KDE then uses the corresponding paired high-flow points.
        all_valid = np.broadcast_to(valid, truth.shape)
        full_truth = (truth / denominator)[all_valid]
        full_predictions = {
            label: (prediction / denominator)[all_valid]
            for label, prediction in item["predictions"].items()
        }
        high_flow_errors = {
            label: np.abs((prediction - truth) / denominator)[event_mask]
            for label, prediction in item["predictions"].items()
        }

        distribution_axis, error_axis = axes[row]
        for label, values, color in (
            ("Observed", full_truth, OBSERVED),
            ("STE-Base", full_predictions["STE-Base"], STE_BASE),
            ("Hyb-STEX", full_predictions["Hyb-STEX"], HYB_STEX),
        ):
            x, density = curve(values, (0.002, 0.998))
            distribution_axis.plot(x, density, color=color, lw=1.35)
        distribution_axis.axvline(1.0, color=P90, lw=1.2, ls="--")
        distribution_axis.set_ylabel(f"{DATASET_LABELS[dataset]}\nDensity")

        for label, color in (("STE-Base", STE_BASE), ("Hyb-STEX", HYB_STEX)):
            x, density = curve(high_flow_errors[label], (0.002, 0.998))
            error_axis.plot(x, density, color=color, lw=1.35)
        error_axis.axvline(0.0, color="#777777", lw=0.7, zorder=1)

        for axis in (distribution_axis, error_axis):
            axis.grid(False)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
        distribution_axis.set_xlabel("Normalized flow")
        error_axis.set_xlabel("Absolute error / node-flow p90")
        metadata[dataset] = {
            "split": "test only",
            "aggregation": "all valid nodes, horizons, and flow channels",
            "high_flow_points": int(event_mask.sum()),
        }

    figure.legend(
        handles=[
            Line2D([0], [0], color=OBSERVED, lw=1.55, label="Observed"),
            Line2D([0], [0], color=STE_BASE, lw=1.35, label="STE-Base"),
            Line2D([0], [0], color=HYB_STEX, lw=1.35, label="Hyb-STEX"),
            Line2D([0], [0], color=P90, lw=1.2, ls="--", label="Training-target p90 (= 1)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=4, frameon=True,
        fancybox=True, edgecolor="#CFCFCF", fontsize=8.8, handlelength=2.6, columnspacing=1.1,
    )
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.07, top=0.87, hspace=0.29, wspace=0.26)
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path, metadata={"Title": "All-node test distribution and high-flow error KDEs", "Creator": Path(__file__).name})
    plt.close(figure)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    for dataset, item in metadata.items():
        print(f"{dataset}: high-flow pooled test points={item['high_flow_points']}")


if __name__ == "__main__":
    main()

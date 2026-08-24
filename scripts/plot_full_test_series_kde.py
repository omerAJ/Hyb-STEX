#!/usr/bin/env python3
"""Export one shared, four-dataset full-test KDE comparison (inference only).

The nodes, flow channels, forecast horizons, and checkpoint suite match the
retained Figure-1 event comparison. Each KDE uses every chronological test
target and the corresponding three-seed mean prediction; the training-target
p90 line is retained as a high-flow-tail reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import gaussian_kde

from plot_figure1_style_four_dataset_composite import (
    DATA_DIR,
    DATASET_LABELS,
    DATASET_ORDER,
    FLOW_NAMES,
    HYB_STEX,
    MODEL_COLORS,
    MODEL_RESULTS_DIR,
    OBSERVED,
    P90,
    ROOT,
    SELECTION_CONTEXT_DIR,
    STE_BASE,
    configure_style,
    select_global_maximum_event,
    select_improved_bj_event,
)
from plot_four_dataset_high_flow_panels import load_suite_predictions
from plot_test_high_flow_comparison import (
    load_raw_targets,
    read_json,
    require_file,
    validate_protocol,
)


OUTPUT_DIR = ROOT / "publication_full_test_series_kde"
OUTPUT_STEM = "hybstex_four_dataset_full_test_kdes"


def kde_curve(
    values: np.ndarray, grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Gaussian KDE on a supplied shared, nonnegative grid."""

    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.allclose(values, values[0]):
        pad = max(abs(float(values[0])) * 0.05, 1.0)
        x = (
            np.asarray(grid, dtype=float)
            if grid is not None
            else np.linspace(max(0.0, float(values[0]) - pad), float(values[0]) + pad, 200)
        )
        y = np.zeros_like(x)
        y[len(y) // 2] = 1.0
        return x, y
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max((hi - lo) * 0.12, 1.0)
    x = (
        np.asarray(grid, dtype=float)
        if grid is not None
        else np.linspace(max(0.0, lo - pad), hi + pad, 400)
    )
    return x, gaussian_kde(values)(x)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=OBSERVED, lw=1.7, label="Observed"),
        Line2D([0], [0], color=STE_BASE, lw=1.55, label="STE-Base"),
        Line2D([0], [0], color=HYB_STEX, lw=1.55, label="Hyb-STEX"),
        Line2D([0], [0], color=P90, lw=1.4, ls="--", label="Training-target p90"),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_manifest_path = require_file(SELECTION_CONTEXT_DIR / "run_manifest.json")
    context_manifest = read_json(context_manifest_path)
    validate_protocol(context_manifest.get("protocol", {}), context_manifest_path)

    items: dict[str, dict] = {}
    for dataset in DATASET_ORDER:
        manifest = read_json(require_file(SELECTION_CONTEXT_DIR / dataset / "dataset_manifest.json"))
        target, thresholds, events, _ = load_raw_targets(
            DATA_DIR, SELECTION_CONTEXT_DIR, dataset, context_manifest, manifest
        )
        items[dataset] = {
            "target": target,
            "thresholds": thresholds,
            "events": events,
            "selection": None if dataset == "BJTaxi" else select_global_maximum_event(dataset, target, events),
        }

    for dataset in DATASET_ORDER:
        predictions, _ = load_suite_predictions(
            dataset, DATA_DIR, MODEL_RESULTS_DIR, items[dataset]["target"], "auto"
        )
        items[dataset]["predictions"] = predictions
        if dataset == "BJTaxi":
            items[dataset]["selection"] = select_improved_bj_event(
                items[dataset]["target"], items[dataset]["events"], predictions
            )

    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.15))
    metadata: dict[str, dict[str, object]] = {}
    for axis, dataset, letter in zip(axes.flat, DATASET_ORDER, "abcd"):
        item = items[dataset]
        selection = item["selection"]
        assert selection is not None
        truth = item["target"][:, selection.horizon_index, selection.node_index, selection.flow_index]
        threshold = float(item["thresholds"][selection.horizon_index, selection.node_index, selection.flow_index])
        predictions = {
            label: np.mean(
                item["predictions"][label][:, :, selection.horizon_index, selection.node_index, selection.flow_index],
                axis=0,
            )
            for label in ("STE-Base", "Hyb-STEX")
        }
        values = {"Observed": truth, **predictions}
        grid_max = max(float(np.max(series)) for series in values.values())
        shared_grid = np.linspace(0.0, max(grid_max * 1.08, 1.0), 500)
        for label, color, width in (
            ("Observed", OBSERVED, 1.7),
            ("STE-Base", STE_BASE, 1.55),
            ("Hyb-STEX", HYB_STEX, 1.55),
        ):
            x, density = kde_curve(values[label], shared_grid)
            axis.plot(x, density, color=color, lw=width)
        axis.axvline(threshold, color=P90, lw=1.4, ls="--", zorder=1)
        axis.set_xlabel("Flow")
        axis.set_ylabel("Density")
        axis.set_xlim(shared_grid[0], shared_grid[-1])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        flow_label = FLOW_NAMES[selection.flow_index].capitalize()
        axis.set_title(
            f"({letter}) {DATASET_LABELS[dataset]} — node {selection.node_index}, {flow_label}",
            pad=5,
        )
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
        metadata[dataset] = {
            "split": "test only",
            "node_index": selection.node_index,
            "horizon_index": selection.horizon_index,
            "flow": FLOW_NAMES[selection.flow_index],
            "training_target_p90": threshold,
            "kde_population": "All chronological test-split predictions and observations",
        }

    figure.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.992),
        ncol=4, frameon=True, fancybox=True, edgecolor="#CFCFCF", fontsize=9.1,
        handlelength=2.7, columnspacing=1.25,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.865, hspace=0.31, wspace=0.23)
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path, metadata={"Title": "Hyb-STEX four-dataset full-test KDE comparison", "Creator": Path(__file__).name})
    plt.close(figure)
    metadata["combined_figure"] = {"png": str(png_path), "pdf": str(pdf_path)}
    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()

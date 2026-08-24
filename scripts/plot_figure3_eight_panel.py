#!/usr/bin/env python3
"""Export the full-page eight-panel Figure 3 replacement (inference only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_figure1_style_four_dataset_composite import (
    DATA_DIR,
    DATASET_LABELS,
    DATASET_ORDER,
    EVENT_MARKER,
    FLOW_NAMES,
    MODEL_COLORS,
    MODEL_RESULTS_DIR,
    OBSERVED,
    OUTPUT_DIR as _UNUSED_OUTPUT_DIR,
    P90,
    ROOT,
    SELECTION_CONTEXT_DIR,
    configure_style,
    legend_handles,
    select_global_maximum_event,
    select_improved_bj_event,
)
from plot_four_dataset_high_flow_panels import load_suite_predictions
from plot_full_test_series_kde import kde_curve
from plot_test_high_flow_comparison import (
    load_raw_targets,
    read_json,
    require_file,
    validate_protocol,
)


OUTPUT_DIR = ROOT / "publication_figure3_eight_panel"
OUTPUT_STEM = "hybstex_figure3_timeseries_kde"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_path = require_file(SELECTION_CONTEXT_DIR / "run_manifest.json")
    context = read_json(context_path)
    validate_protocol(context.get("protocol", {}), context_path)

    items: dict[str, dict] = {}
    for dataset in DATASET_ORDER:
        manifest = read_json(
            require_file(SELECTION_CONTEXT_DIR / dataset / "dataset_manifest.json")
        )
        target, thresholds, events, _ = load_raw_targets(
            DATA_DIR, SELECTION_CONTEXT_DIR, dataset, context, manifest
        )
        items[dataset] = {
            "target": target,
            "thresholds": thresholds,
            "events": events,
            "selection": None
            if dataset == "BJTaxi"
            else select_global_maximum_event(dataset, target, events),
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
    figure, axes = plt.subplots(4, 2, figsize=(8.2, 10.4))
    letters = iter("abcdefgh")
    for row, dataset in enumerate(DATASET_ORDER):
        item = items[dataset]
        selection = item["selection"]
        assert selection is not None
        flow_label = FLOW_NAMES[selection.flow_index].capitalize()
        descriptor = (
            f"{DATASET_LABELS[dataset]} — node {selection.node_index}, {flow_label}"
        )
        threshold = float(
            item["thresholds"][
                selection.horizon_index,
                selection.node_index,
                selection.flow_index,
            ]
        )
        predictions = {
            label: np.mean(
                item["predictions"][label][
                    :,
                    :,
                    selection.horizon_index,
                    selection.node_index,
                    selection.flow_index,
                ],
                axis=0,
            )
            for label in ("STE-Base", "Hyb-STEX")
        }

        # Left: the approved high-flow event window.
        time_axis = axes[row, 0]
        sl = slice(selection.window_start, selection.window_end)
        x = np.arange(selection.window_start, selection.window_end)
        truth = item["target"][
            :, selection.horizon_index, selection.node_index, selection.flow_index
        ]
        events = item["events"][
            :, selection.horizon_index, selection.node_index, selection.flow_index
        ]
        window_values = [truth[sl], np.asarray([threshold])]
        time_axis.plot(x, truth[sl], color=OBSERVED, lw=1.5, zorder=4)
        for label in ("STE-Base", "Hyb-STEX"):
            values = predictions[label][sl]
            window_values.append(values)
            time_axis.plot(x, values, color=MODEL_COLORS[label], lw=1.35, zorder=3)
        time_axis.axhline(threshold, color=P90, lw=1.25, ls="--", zorder=1)
        time_axis.scatter(
            x[events[sl]], truth[sl][events[sl]], s=24, facecolors="none",
            edgecolors=EVENT_MARKER, linewidths=1.0, zorder=5,
        )
        if dataset == "BJTaxi":
            focus = (x >= selection.test_index - 8) & (x <= selection.test_index + 8)
            focused = [v[focus] if v.size == x.size else v for v in window_values]
            upper = max(float(np.max(v)) for v in focused)
            lower = min(float(np.min(v)) for v in focused)
            span = max(upper - lower, 1.0)
            time_axis.set_ylim(lower - 0.07 * span, upper + 0.08 * span)
        else:
            upper = max(float(np.max(v)) for v in window_values)
            lower = min(float(np.min(v)) for v in window_values)
            span = max(upper - min(0.0, lower), 1.0)
            time_axis.set_ylim(min(0.0, lower - 0.04 * span), upper + 0.07 * span)
        time_axis.set_xlim(selection.window_start, selection.window_end - 1)
        time_axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        time_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        time_axis.set_xlabel("Test sample index")
        time_axis.set_ylabel("Flow")
        time_axis.set_title(
            f"({next(letters)}) {descriptor}\nHigh-flow test window", pad=4
        )

        # Right: full chronological test distribution for the identical series.
        kde_axis = axes[row, 1]
        full_values = {"Observed": truth, **predictions}
        grid_max = max(float(np.max(values)) for values in full_values.values())
        shared_grid = np.linspace(0.0, max(grid_max * 1.08, 1.0), 500)
        for label in ("Observed", "STE-Base", "Hyb-STEX"):
            density_x, density = kde_curve(full_values[label], shared_grid)
            color = OBSERVED if label == "Observed" else MODEL_COLORS[label]
            kde_axis.plot(density_x, density, color=color, lw=1.5 if label == "Observed" else 1.35)
        kde_axis.axvline(threshold, color=P90, lw=1.25, ls="--", zorder=1)
        kde_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        kde_axis.set_xlabel("Flow")
        kde_axis.set_ylabel("Density")
        kde_axis.set_xlim(shared_grid[0], shared_grid[-1])
        kde_axis.set_title(
            f"({next(letters)}) {descriptor}\nFull-test KDE", pad=4
        )

        for axis in (time_axis, kde_axis):
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")

    figure.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=5, frameon=True, fancybox=True, edgecolor="#CFCFCF", fontsize=8.0,
        handlelength=2.4, columnspacing=0.9,
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.055, top=0.92, hspace=0.66, wspace=0.27
    )
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    figure.savefig(png_path, dpi=600)
    figure.savefig(
        pdf_path,
        metadata={"Title": "Eight-panel Hyb-STEX test-event and KDE comparison", "Creator": Path(__file__).name},
    )
    plt.close(figure)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()

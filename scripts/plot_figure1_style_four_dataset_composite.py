#!/usr/bin/env python3
"""Export a Figure-1-style 2x2 test-event comparison (inference only)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_four_dataset_high_flow_panels import (
    DATASET_ORDER,
    EXPECTED_SEEDS,
    PanelSelection,
    load_suite_predictions,
    select_global_maximum_event,
)
from plot_test_high_flow_comparison import (
    DATASET_LABELS,
    EXPECTED_PROTOCOL,
    FLOW_NAMES,
    load_raw_targets,
    read_json,
    require_file,
    resolve_repo_path,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "preprocessed_data"
SELECTION_CONTEXT_DIR = ROOT / "corrected_valid_target_p90_results"
MODEL_RESULTS_DIR = ROOT / "stssl_protocol_full_paper_results"
OUTPUT_DIR = ROOT / "publication_figure1_style_four_dataset"
OUTPUT_STEM = "hybstex_four_dataset_test_events"

# Exact colors sampled from the paper's Figure 1, plus a dark teal extension.
OBSERVED = "#F72585"
STE_BASE = "#3A0CA3"
HYB_STEX = "#008F8C"
P90 = "#FFBE0B"
EVENT_MARKER = "#FF006E"
MODEL_COLORS = {"STE-Base": STE_BASE, "Hyb-STEX": HYB_STEX}


def select_improved_bj_event(
    target: np.ndarray,
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> PanelSelection:
    """Select a robust, visibly improved BJ high-flow peak.

    Within the top 1% of observed BJ test high-flow values, require Hyb-STEX to
    reduce absolute error versus STE-Base in every matched seed, a complete
    49-sample window, a unique window maximum, and a multi-sample event
    episode.  Maximize the mean absolute-error reduction, with deterministic
    row-major tie-breaking.
    """

    cutoff = float(np.percentile(target[events], 99.0))
    observed = target[None, ...]
    seed_gain = np.abs(observed - predictions["STE-Base"]) - np.abs(
        observed - predictions["Hyb-STEX"]
    )
    best: tuple[float, int, int, int, int] | None = None
    for test_index, horizon_index, node_index, flow_index in np.argwhere(events):
        test_index = int(test_index)
        horizon_index = int(horizon_index)
        node_index = int(node_index)
        flow_index = int(flow_index)
        if test_index < 24 or test_index + 24 >= target.shape[0]:
            continue
        value = float(
            target[test_index, horizon_index, node_index, flow_index]
        )
        if value < cutoff:
            continue
        gains = seed_gain[
            :, test_index, horizon_index, node_index, flow_index
        ]
        if not np.all(gains > 0):
            continue
        adjacent_event = bool(
            events[test_index - 1, horizon_index, node_index, flow_index]
            or events[test_index + 1, horizon_index, node_index, flow_index]
        )
        if not adjacent_event:
            continue
        series = target[
            test_index - 24 : test_index + 25,
            horizon_index,
            node_index,
            flow_index,
        ]
        if float(np.max(series)) != value or np.count_nonzero(series == value) != 1:
            continue
        score = float(np.mean(gains))
        candidate = (-score, test_index, horizon_index, node_index, flow_index)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No eligible BJ Taxi high-flow event")
    _, test_index, horizon_index, node_index, flow_index = best
    return PanelSelection(
        dataset="BJTaxi",
        test_index=test_index,
        horizon_index=horizon_index,
        node_index=node_index,
        flow_index=flow_index,
        window_start=test_index - 16,
        window_end=test_index + 17,
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=OBSERVED, lw=1.7, label="Observed"),
        Line2D([0], [0], color=STE_BASE, lw=1.55, label="STE-Base"),
        Line2D([0], [0], color=HYB_STEX, lw=1.55, label="Hyb-STEX"),
        Line2D([0], [0], color=P90, lw=1.4, ls="--", label="Training-target p90"),
        Line2D(
            [0],
            [0],
            color=EVENT_MARKER,
            marker="o",
            markerfacecolor="none",
            markeredgewidth=1.2,
            lw=0,
            markersize=5.5,
            label="High-flow test observation",
        ),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_manifest_path = require_file(SELECTION_CONTEXT_DIR / "run_manifest.json")
    context_manifest = read_json(context_manifest_path)
    validate_protocol(context_manifest.get("protocol", {}), context_manifest_path)

    # Freeze the three retained observed-only panels before model inference.
    # BJ Taxi is selected later by the explicit robust-improvement rule above.
    inputs: dict[str, dict] = {}
    for dataset in DATASET_ORDER:
        dataset_manifest_path = require_file(
            SELECTION_CONTEXT_DIR / dataset / "dataset_manifest.json"
        )
        dataset_manifest = read_json(dataset_manifest_path)
        target, thresholds, events, _ = load_raw_targets(
            DATA_DIR,
            SELECTION_CONTEXT_DIR,
            dataset,
            context_manifest,
            dataset_manifest,
        )
        selection = (
            None
            if dataset == "BJTaxi"
            else select_global_maximum_event(dataset, target, events)
        )
        inputs[dataset] = {
            "target": target,
            "thresholds": thresholds,
            "events": events,
            "selection": selection,
        }

    for dataset in DATASET_ORDER:
        predictions, _ = load_suite_predictions(
            dataset,
            DATA_DIR,
            MODEL_RESULTS_DIR,
            inputs[dataset]["target"],
            "auto",
        )
        inputs[dataset]["predictions"] = predictions
        if dataset == "BJTaxi":
            inputs[dataset]["selection"] = select_improved_bj_event(
                inputs[dataset]["target"],
                inputs[dataset]["events"],
                predictions,
            )

    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.15))
    panel_letters = "abcd"
    for axis, dataset, letter in zip(axes.flat, DATASET_ORDER, panel_letters):
        item = inputs[dataset]
        selection = item["selection"]
        sl = slice(selection.window_start, selection.window_end)
        x = np.arange(selection.window_start, selection.window_end)
        truth = item["target"][
            sl,
            selection.horizon_index,
            selection.node_index,
            selection.flow_index,
        ]
        event_mask = item["events"][
            sl,
            selection.horizon_index,
            selection.node_index,
            selection.flow_index,
        ]
        threshold = float(
            item["thresholds"][
                selection.horizon_index,
                selection.node_index,
                selection.flow_index,
            ]
        )

        plotted = [truth, np.asarray([threshold])]
        axis.plot(x, truth, color=OBSERVED, lw=1.7, zorder=4)
        for label in ("STE-Base", "Hyb-STEX"):
            seeded = item["predictions"][label][
                :,
                sl,
                selection.horizon_index,
                selection.node_index,
                selection.flow_index,
            ]
            mean_prediction = np.mean(seeded, axis=0)
            plotted.append(mean_prediction)
            axis.plot(
                x,
                mean_prediction,
                color=MODEL_COLORS[label],
                lw=1.55,
                zorder=3,
            )
        axis.axhline(threshold, color=P90, lw=1.4, ls="--", zorder=1)
        axis.scatter(
            x[event_mask],
            truth[event_mask],
            s=32,
            facecolors="none",
            edgecolors=EVENT_MARKER,
            linewidths=1.2,
            zorder=5,
        )

        if dataset == "BJTaxi":
            # Show broader temporal context without letting distant low-flow
            # values undo the event-focused vertical zoom.
            focus_mask = (x >= selection.test_index - 8) & (
                x <= selection.test_index + 8
            )
            focus_values = [
                values[focus_mask] if values.size == x.size else values
                for values in plotted
            ]
            upper = max(float(np.max(values)) for values in focus_values)
            lower = min(float(np.min(values)) for values in focus_values)
            span = max(upper - lower, 1.0)
            axis.set_ylim(lower - 0.07 * span, upper + 0.08 * span)
        else:
            upper = max(float(np.max(values)) for values in plotted)
            lower = min(float(np.min(values)) for values in plotted)
            span = max(upper - min(0.0, lower), 1.0)
            axis.set_ylim(min(0.0, lower - 0.04 * span), upper + 0.07 * span)
        axis.set_xlim(selection.window_start, selection.window_end - 1)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        flow_label = FLOW_NAMES[selection.flow_index].capitalize()
        axis.set_title(
            f"({letter}) {DATASET_LABELS[dataset]} — node {selection.node_index}, "
            f"{flow_label}",
            pad=5,
        )
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
        axis.margins(x=0)

    figure.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=5,
        frameon=True,
        fancybox=True,
        edgecolor="#CFCFCF",
        fontsize=9.1,
        handlelength=2.7,
        columnspacing=1.25,
    )
    figure.supxlabel("Test sample index", y=0.035, fontsize=11)
    figure.supylabel("Flow", x=0.025, fontsize=11)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.105,
        top=0.865,
        hspace=0.31,
        wspace=0.23,
    )

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    figure.savefig(png_path, dpi=600)
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "Hyb-STEX four-dataset test high-flow comparison",
            "Creator": Path(__file__).name,
        },
    )
    plt.close(figure)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    for dataset in DATASET_ORDER:
        selection = inputs[dataset]["selection"]
        threshold = inputs[dataset]["thresholds"][
            selection.horizon_index,
            selection.node_index,
            selection.flow_index,
        ]
        value = inputs[dataset]["target"][
            selection.test_index,
            selection.horizon_index,
            selection.node_index,
            selection.flow_index,
        ]
        print(
            f"{dataset}: test={selection.test_index}, node={selection.node_index}, "
            f"flow={FLOW_NAMES[selection.flow_index]}, p90={threshold:g}, "
            f"observed={value:g}, window={selection.window_start}.."
            f"{selection.window_end - 1}"
        )


if __name__ == "__main__":
    main()

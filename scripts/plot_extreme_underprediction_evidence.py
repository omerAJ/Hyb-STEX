import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.dataloader import get_dataloader
from lib.utils import load_graph
from model.models import STSSL
from scripts.run_paper_rescue_ablation import load_config, repo_path


MODEL_SPECS = {
    "Base predictor": {
        "results": "paper_rescue_ablation_results/per_seed_metrics.csv",
        "variant": "A_base_mae",
        "mode": "original",
        "phase": "pred",
        "color": "#4B5563",
    },
    "Hyb-STEX": {
        "results": "residual_training_schedule_ablation_results/per_seed_metrics.csv",
        "variant": "D_frozen_residual_event_weighted",
        "mode": "event_weighted_ungated_residual",
        "phase": "pred_2",
        "color": "#D1495B",
    },
}

FLOW_NAMES = ("Inflow", "Outflow")
SEVERITY_LABELS = ("Overall", "Top 25%", "Top 10%", "Top 5%", "Top 1%")
TAIL_LABELS = SEVERITY_LABELS[1:]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper figures verifying high-flow underprediction and correction."
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--data-dir", default="preprocessed_data")
    parser.add_argument("--graph-file", default="preprocessed_data/NYCTaxi/adj_mx.npz")
    parser.add_argument("--output-dir", default="paper_extreme_underprediction_evidence")
    parser.add_argument(
        "--paper-results",
        default="paper_rescue_ablation_results",
        help="Directory containing the base-predictor ablation results.",
    )
    parser.add_argument(
        "--schedule-results",
        default="residual_training_schedule_ablation_results",
        help="Directory containing the selected frozen-residual results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size; defaults to the dataset configuration.",
    )
    parser.add_argument("--window-length", type=int, default=49)
    return parser.parse_args()


def checkpoint_lookup(path, variant):
    checkpoints = {}
    with repo_path(path).open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == variant and row["flow"] == "mean":
                checkpoints[int(row["seed"])] = Path(row["checkpoint"])
    if sorted(checkpoints) != [1, 2, 3]:
        raise ValueError(f"Expected seeds 1, 2, 3 for {variant}; found {sorted(checkpoints)}")
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    return checkpoints


def predict_checkpoint(base_config, dataloader, graph, graph_file, checkpoint, mode, phase):
    config = dict(base_config)
    config.update({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_nodes": len(graph),
        "graph_file": str(graph_file),
        "ablation_mode": mode,
        "event_loss_weight": 0.25,
        "bias_param_scope": "head_only",
        "phase3_mode": "original",
        "classification_loss_weight": 1.0,
        "boost_gate_scale": 1.0,
        "gate_floor": 0.0,
    })
    model = STSSL(SimpleNamespace(**config)).to(config["device"])
    state = torch.load(checkpoint, map_location=config["device"])["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    predictions = []
    targets = []
    events = []
    with torch.no_grad():
        for data, target, event, _ in dataloader["test"]:
            representation, representation_cls = model(data, graph)
            predictions.append(model.predict(representation, representation_cls, phase))
            targets.append(target)
            events.append(event.cpu())
    prediction = dataloader["scaler"].inverse_transform(torch.cat(predictions)).cpu().numpy()
    target = dataloader["scaler"].inverse_transform(torch.cat(targets)).cpu().numpy()
    event = torch.cat(events).numpy()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction[:, 0], target[:, 0], event[:, 0]


def collect_predictions(args):
    base_config = load_config(args.config_filename)
    batch_size = args.batch_size or base_config["test_batch_size"]
    dataloader = get_dataloader(
        args.data_dir,
        base_config["dataset"],
        batch_size,
        batch_size,
        scalar_type="Standard",
        scaler_fit="train",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = load_graph(args.graph_file, device=device)
    predictions = {}
    target = None
    event = None
    for model_name, spec in MODEL_SPECS.items():
        predictions[model_name] = []
        checkpoints = checkpoint_lookup(spec["results"], spec["variant"])
        for seed, checkpoint in sorted(checkpoints.items()):
            pred, current_target, current_event = predict_checkpoint(
                base_config,
                dataloader,
                graph,
                args.graph_file,
                checkpoint,
                spec["mode"],
                spec["phase"],
            )
            predictions[model_name].append(pred)
            if target is None:
                target = current_target
                event = current_event
            elif not np.array_equal(current_event, event) or not np.allclose(current_target, target):
                raise ValueError("Test targets or event labels changed between evaluations")
        predictions[model_name] = np.stack(predictions[model_name], axis=0)
    return predictions, target, event


def severity_masks(target, flow_index):
    y = target[..., flow_index]
    evaluation_mask = y > 5.0
    evaluated_values = y[evaluation_mask]
    masks = [evaluation_mask]
    for percentile in (75, 90, 95, 99):
        threshold = np.percentile(evaluated_values, percentile)
        masks.append(evaluation_mask & (y >= threshold))
    return masks


def calculate_metrics(predictions, target, event):
    rows = []
    for flow_index, flow in enumerate(FLOW_NAMES):
        masks = severity_masks(target, flow_index)
        for model_name, seeded_predictions in predictions.items():
            for seed_index, prediction in enumerate(seeded_predictions, start=1):
                difference = prediction[..., flow_index] - target[..., flow_index]
                for label, mask in zip(SEVERITY_LABELS, masks):
                    selected = difference[mask]
                    rows.append({
                        "model": model_name,
                        "seed": seed_index,
                        "flow": flow,
                        "severity": label,
                        "count": int(mask.sum()),
                        "target_mean": float(target[..., flow_index][mask].mean()),
                        "prediction_mean": float(prediction[..., flow_index][mask].mean()),
                        "mae": float(np.abs(selected).mean()),
                        "signed_error": float(selected.mean()),
                        "underprediction_rate": float((selected < 0).mean()),
                        "median_signed_error": float(np.median(selected)),
                    })
    return rows


def aggregate(rows, model, flow, severity, metric):
    values = [
        row[metric]
        for row in rows
        if row["model"] == model and row["flow"] == flow and row["severity"] == severity
    ]
    return float(np.mean(values)), float(np.std(values))


def save_figure(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mae(rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
    x = np.arange(len(SEVERITY_LABELS))
    for flow, axis in zip(FLOW_NAMES, axes):
        for model_name, spec in MODEL_SPECS.items():
            values = [aggregate(rows, model_name, flow, label, "mae") for label in SEVERITY_LABELS]
            means = np.array([value[0] for value in values])
            stds = np.array([value[1] for value in values])
            axis.plot(x, means, marker="o", linewidth=2.2, color=spec["color"], label=model_name)
            axis.fill_between(x, means - stds, means + stds, color=spec["color"], alpha=0.14)
        axis.set_title(flow)
        axis.set_xticks(x, SEVERITY_LABELS, rotation=25, ha="right")
        axis.set_ylabel("Mean absolute error")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Forecast error rises sharply in the upper tail of observed test flow", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "mae_by_test_flow_percentile")


def plot_underprediction(rows, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7), sharex="col")
    x = np.arange(len(TAIL_LABELS))
    for column, flow in enumerate(FLOW_NAMES):
        for model_name, spec in MODEL_SPECS.items():
            signed = [aggregate(rows, model_name, flow, label, "signed_error") for label in TAIL_LABELS]
            rate = [aggregate(rows, model_name, flow, label, "underprediction_rate") for label in TAIL_LABELS]
            signed_mean = np.array([value[0] for value in signed])
            signed_std = np.array([value[1] for value in signed])
            rate_mean = 100 * np.array([value[0] for value in rate])
            rate_std = 100 * np.array([value[1] for value in rate])
            axes[0, column].plot(x, signed_mean, marker="o", linewidth=2.2, color=spec["color"], label=model_name)
            axes[0, column].fill_between(
                x, signed_mean - signed_std, signed_mean + signed_std,
                color=spec["color"], alpha=0.14,
            )
            axes[1, column].plot(x, rate_mean, marker="o", linewidth=2.2, color=spec["color"])
            axes[1, column].fill_between(
                x, rate_mean - rate_std, rate_mean + rate_std,
                color=spec["color"], alpha=0.14,
            )
        axes[0, column].axhline(0, color="#111827", linewidth=0.8)
        axes[0, column].set_title(flow)
        axes[0, column].set_ylabel("Signed error (prediction - target)")
        axes[1, column].set_ylabel("Underpredicted points (%)")
        axes[1, column].set_xticks(x, TAIL_LABELS, rotation=25, ha="right")
        for row in range(2):
            axes[row, column].grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
            axes[row, column].spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Underprediction is concentrated in the most extreme high-flow observations", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "underprediction_by_test_flow_percentile")


def plot_event_calibration(predictions, target, event, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for flow_index, (flow, axis) in enumerate(zip(FLOW_NAMES, axes)):
        event_mask = event[..., flow_index] == 1
        event_target = target[..., flow_index][event_mask]
        edges = np.unique(np.percentile(event_target, np.linspace(0, 100, 11)))
        lower = min(
            event_target.min(),
            *(prediction[..., flow_index][:, event_mask].min() for prediction in predictions.values()),
        )
        upper = max(
            event_target.max(),
            *(prediction[..., flow_index][:, event_mask].max() for prediction in predictions.values()),
        )
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#9CA3AF", label="Ideal")
        for model_name, spec in MODEL_SPECS.items():
            seed_x = []
            seed_y = []
            for prediction in predictions[model_name]:
                xs = []
                ys = []
                event_prediction = prediction[..., flow_index][event_mask]
                for index in range(len(edges) - 1):
                    if index == len(edges) - 2:
                        mask = (event_target >= edges[index]) & (event_target <= edges[index + 1])
                    else:
                        mask = (event_target >= edges[index]) & (event_target < edges[index + 1])
                    xs.append(event_target[mask].mean())
                    ys.append(event_prediction[mask].mean())
                seed_x.append(xs)
                seed_y.append(ys)
            x_values = np.mean(seed_x, axis=0)
            y_values = np.mean(seed_y, axis=0)
            y_std = np.std(seed_y, axis=0)
            axis.plot(x_values, y_values, marker="o", linewidth=2.2, color=spec["color"], label=model_name)
            axis.fill_between(x_values, y_values - y_std, y_values + y_std, color=spec["color"], alpha=0.14)
        axis.set_title(flow)
        axis.set_xlabel("Mean observed flow within event-target decile")
        axis.set_ylabel("Mean predicted flow")
        axis.grid(color="#E5E7EB", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Predictions fall below observed values in the upper high-flow region", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "event_region_calibration")


def plot_maximum_event_windows(predictions, target, event, output_dir, window_length):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2))
    half = window_length // 2
    for flow_index, (flow, axis) in enumerate(zip(FLOW_NAMES, axes)):
        event_values = np.where(event[..., flow_index] == 1, target[..., flow_index], -np.inf)
        time_index, node_index = np.unravel_index(np.argmax(event_values), event_values.shape)
        start = max(0, time_index - half)
        end = min(target.shape[0], start + window_length)
        start = max(0, end - window_length)
        x = np.arange(start - time_index, end - time_index)
        truth = target[start:end, node_index, flow_index]
        axis.plot(x, truth, color="#111827", linewidth=2.3, label="Observed")
        for model_name, spec in MODEL_SPECS.items():
            seeded = predictions[model_name][:, start:end, node_index, flow_index]
            mean = seeded.mean(axis=0)
            std = seeded.std(axis=0)
            axis.plot(x, mean, color=spec["color"], linewidth=2.0, label=model_name)
            axis.fill_between(x, mean - std, mean + std, color=spec["color"], alpha=0.14)
        event_window = event[start:end, node_index, flow_index] == 1
        axis.scatter(x[event_window], truth[event_window], color="#E9A23B", edgecolor="white", s=42, zorder=5, label="High-flow observation")
        axis.axvline(0, color="#9CA3AF", linestyle="--", linewidth=1)
        axis.set_title(f"{flow}: node {node_index}, window centered on the largest observed event")
        axis.set_xlabel("Test time steps relative to maximum event")
        axis.set_ylabel("Flow")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle("Objective maximum-event examples", fontsize=13, y=0.99)
    fig.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(fig, output_dir, "maximum_observed_event_windows")


def write_metrics(rows, output_dir):
    path = output_dir / "severity_metrics_per_seed.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_key_results(rows):
    print("\nKey aggregate results")
    for flow in FLOW_NAMES:
        for severity in SEVERITY_LABELS:
            parts = []
            for model in MODEL_SPECS:
                mae, _ = aggregate(rows, model, flow, severity, "mae")
                signed, _ = aggregate(rows, model, flow, severity, "signed_error")
                under, _ = aggregate(rows, model, flow, severity, "underprediction_rate")
                parts.append(f"{model}: MAE={mae:.2f}, signed={signed:.2f}, under={100*under:.1f}%")
            print(f"{flow} {severity}: " + " | ".join(parts))


def main():
    args = parse_args()
    # The orchestrator stores each dataset in its own result directory.  Keep
    # the defaults above for standalone use, while allowing that layout here.
    MODEL_SPECS["Base predictor"]["results"] = str(
        repo_path(args.paper_results) / "per_seed_metrics.csv"
    )
    MODEL_SPECS["Hyb-STEX"]["results"] = str(
        repo_path(args.schedule_results) / "per_seed_metrics.csv"
    )
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, target, event = collect_predictions(args)
    rows = calculate_metrics(predictions, target, event)
    write_metrics(rows, output_dir)
    plot_mae(rows, output_dir)
    plot_underprediction(rows, output_dir)
    plot_event_calibration(predictions, target, event, output_dir)
    plot_maximum_event_windows(
        predictions, target, event, output_dir, args.window_length
    )
    print_key_results(rows)
    print(f"\nWrote evidence figures to {output_dir}")


if __name__ == "__main__":
    main()

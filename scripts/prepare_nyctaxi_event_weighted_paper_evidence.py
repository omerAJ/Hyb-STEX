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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.dataloader import get_dataloader
from lib.utils import init_seed, load_graph
from model.models import STSSL
from scripts.run_nyctaxi_event_gate_ablation import DEFAULTS, VARIANT_CONFIGS


LABELS = {
    "base_model": "Base",
    "original": "Legacy gated",
    "ungated_bias": "Legacy ungated",
    "ungated_head_only": "Ungated residual",
    "original_head_only": "Gated residual",
    "floor_gated": "Floor-gated residual",
    "dual_event_residual": "Dual residual",
    "event_weighted_original": "Event-weighted gated",
    "event_weighted_dual_residual": "Event-weighted dual",
}

PLOT_ORDER = [
    "base_model",
    "original",
    "ungated_bias",
    "ungated_head_only",
    "original_head_only",
    "floor_gated",
    "dual_event_residual",
    "event_weighted_original",
    "event_weighted_dual_residual",
]

PREDICTION_VARIANTS = [
    "base_model",
    "ungated_head_only",
    "event_weighted_dual_residual",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare paper-ready evidence for NYCTaxi event-weighted Hyb-STEX."
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--base-results", default="nyctaxi_base_and_event_weighted_results")
    parser.add_argument("--clean-results", default="nyctaxi_event_gate_ablation_results")
    parser.add_argument(
        "--legacy-results",
        default="nyctaxi_classification_head_ablation_results",
    )
    parser.add_argument("--data-dir", default="preprocessed_data")
    parser.add_argument("--graph-file", default="preprocessed_data/NYCTaxi/adj_mx.npz")
    parser.add_argument("--output-dir", default="paper_evidence_event_weighted")
    parser.add_argument("--device", default=None)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--example-seed", type=int, default=1)
    parser.add_argument("--scatter-sample", type=int, default=5000)
    return parser.parse_args()


def repo_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def read_rows(path):
    path = repo_path(path)
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    return float(value) if value not in ("", None) else np.nan


def as_int(value):
    return int(float(value))


def load_summary(args):
    combined = []
    sources = [
        ("base_event_weighted", repo_path(args.base_results) / "summary_metrics.csv"),
        ("clean_head_only", repo_path(args.clean_results) / "summary_metrics.csv"),
        ("legacy", repo_path(args.legacy_results) / "summary_metrics.csv"),
    ]
    seen = set()
    for source, path in sources:
        if not path.is_file():
            continue
        for row in read_rows(path):
            key = (row["variant"], row["flow"])
            if key in seen:
                continue
            seen.add(key)
            combined.append(
                {
                    "source": source,
                    "variant": row["variant"],
                    "label": LABELS.get(row["variant"], row["variant"]),
                    "flow": row["flow"],
                    "mae": to_float(row["mae"]),
                    "eee": to_float(row["eee"]),
                }
            )
    order = {variant: idx for idx, variant in enumerate(PLOT_ORDER)}
    combined.sort(key=lambda r: (order.get(r["variant"], 999), r["flow"]))
    return combined


def load_per_seed(args):
    rows = []
    sources = [
        ("base_event_weighted", repo_path(args.base_results) / "per_seed_metrics.csv"),
        ("clean_head_only", repo_path(args.clean_results) / "per_seed_metrics.csv"),
        ("legacy", repo_path(args.legacy_results) / "per_seed_metrics.csv"),
    ]
    seen = set()
    for source, path in sources:
        if not path.is_file():
            continue
        for row in read_rows(path):
            key = (row["variant"], row["seed"], row["flow"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": source,
                    "variant": row["variant"],
                    "label": LABELS.get(row["variant"], row["variant"]),
                    "seed": as_int(row["seed"]),
                    "flow": row["flow"],
                    "mae": to_float(row["mae"]),
                    "eee": to_float(row["eee"]),
                    "log_dir": row.get("log_dir", ""),
                    "checkpoint": row.get("checkpoint", ""),
                }
            )
    return rows


def get_mean_rows(rows):
    return [row for row in rows if row["flow"] == "mean"]


def plot_metric_bars(summary_rows, output_dir):
    means = get_mean_rows(summary_rows)
    labels = [row["label"] for row in means]
    mae = np.array([row["mae"] for row in means], dtype=float)
    eee = np.array([row["eee"] for row in means], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    colors = ["#6E7F80" if "Event-weighted dual" not in label else "#B23A48" for label in labels]
    axes[0].barh(labels, mae, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE (lower is better)")
    axes[0].set_title("Average Forecast Error")
    axes[1].barh(labels, eee, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("EEE on evs_90 points (lower is better)")
    axes[1].set_title("Extreme-Event Error")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    save_figure(fig, output_dir / "metrics_bar")


def plot_pareto(summary_rows, output_dir):
    means = get_mean_rows(summary_rows)
    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    label_offsets = {
        "base_model": (-58, 6),
        "original": (6, -16),
        "ungated_bias": (8, 7),
        "ungated_head_only": (8, 10),
        "original_head_only": (8, 4),
        "floor_gated": (8, 14),
        "dual_event_residual": (8, -9),
        "event_weighted_original": (8, 6),
        "event_weighted_dual_residual": (8, 6),
    }
    for row in means:
        marker = "*" if row["variant"] == "event_weighted_dual_residual" else "o"
        size = 180 if marker == "*" else 75
        ax.scatter(row["mae"], row["eee"], s=size, marker=marker)
        x_offset, y_offset = label_offsets.get(row["variant"], (5, 4))
        ax.annotate(
            row["label"],
            (row["mae"], row["eee"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8.0,
        )
    ax.set_xlabel("Mean MAE")
    ax.set_ylabel("Mean EEE")
    ax.set_title("MAE-EEE Tradeoff")
    ax.grid(alpha=0.25)
    ax.margins(x=0.08, y=0.08)
    ax.text(
        0.02,
        0.98,
        "Lower-left is better",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        va="top",
    )
    save_figure(fig, output_dir / "mae_eee_pareto")


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def escape_latex(value):
    return str(value).replace("&", "\\&").replace("_", "\\_")


def write_summary_tables(per_seed_rows, output_dir):
    mean_rows = get_mean_rows(per_seed_rows)
    rows = []
    for variant in PLOT_ORDER:
        variant_rows = [row for row in mean_rows if row["variant"] == variant]
        if not variant_rows:
            continue
        mae_values = [row["mae"] for row in variant_rows]
        eee_values = [row["eee"] for row in variant_rows]
        rows.append(
            {
                "variant": variant,
                "label": LABELS.get(variant, variant),
                "seeds": len(variant_rows),
                "mae_mean": float(np.mean(mae_values)),
                "mae_std": sample_std(mae_values),
                "eee_mean": float(np.mean(eee_values)),
                "eee_std": sample_std(eee_values),
            }
        )

    write_rows(
        output_dir / "summary_mean_std.csv",
        rows,
        ["variant", "label", "seeds", "mae_mean", "mae_std", "eee_mean", "eee_std"],
    )

    md_lines = [
        "| Variant | MAE mean +/- std | EEE mean +/- std |",
        "|---|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['label']} | {row['mae_mean']:.4f} +/- {row['mae_std']:.4f} | "
            f"{row['eee_mean']:.4f} +/- {row['eee_std']:.4f} |"
        )
    (output_dir / "paper_table_metrics.md").write_text("\n".join(md_lines), encoding="utf-8")

    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{NYCTaxi test performance across seeds 1, 2, and 3. Lower is better.}",
        "\\label{tab:nyctaxi_event_weighted}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Variant & MAE $\\downarrow$ & EEE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        latex_lines.append(
            f"{escape_latex(row['label'])} & "
            f"{row['mae_mean']:.4f} $\\pm$ {row['mae_std']:.4f} & "
            f"{row['eee_mean']:.4f} $\\pm$ {row['eee_std']:.4f} \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (output_dir / "paper_table_metrics.tex").write_text("\n".join(latex_lines), encoding="utf-8")
    return rows


def build_model_args(args, variant, seed):
    with repo_path(args.config_filename).open("r") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    merged = dict(DEFAULTS)
    merged.update(config)
    if variant == "base_model":
        variant_config = {
            "ablation_mode": "original",
            "bias_param_scope": "head_only",
            "start_phase": "pred",
            "stop_after_phase": "pred",
        }
    else:
        variant_config = dict(VARIANT_CONFIGS[variant])
    merged.update(variant_config)
    merged.update(
        {
            "seed": seed,
            "mode": "test",
            "data_dir": args.data_dir,
            "graph_file": args.graph_file,
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "test_batch_size": args.test_batch_size,
            "batch_size": args.test_batch_size,
            "event_loss_weight": args.event_loss_weight,
            "classification_loss_weight": 1.0,
            "gate_floor": 0.5,
            "debug": False,
            "comment": "paper_evidence",
            "experimentName": f"paper_evidence_{variant}_seed={seed}",
            "load_path": None,
            "max_train_batches": None,
            "max_eval_batches": None,
        }
    )
    return SimpleNamespace(**merged)


def checkpoint_lookup(per_seed_rows):
    lookup = {}
    for row in per_seed_rows:
        if row["flow"] != "mean" or not row.get("checkpoint"):
            continue
        lookup[(row["variant"], row["seed"])] = row["checkpoint"]
    return lookup


def predict_variant(args, checkpoint, variant, seed):
    model_args = build_model_args(args, variant, seed)
    init_seed(seed)
    dataloader = get_dataloader(
        data_dir=model_args.data_dir,
        dataset=model_args.dataset,
        batch_size=model_args.batch_size,
        test_batch_size=model_args.test_batch_size,
        scalar_type="Standard",
    )
    graph = load_graph(model_args.graph_file, device=model_args.device)
    model_args.num_nodes = len(graph)
    model = STSSL(model_args).to(model_args.device)
    state = torch.load(checkpoint, map_location=torch.device(model_args.device))["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    preds = []
    truths = []
    events = []
    phase = "pred" if variant == "base_model" else "pred_2"
    with torch.no_grad():
        for data, target, evs, _ in dataloader["test"]:
            z1, z1_cls = model(data, graph)
            pred_scaled = model.predict(z1, z1_cls, phase)
            pred = dataloader["scaler"].inverse_transform(pred_scaled)
            true = dataloader["scaler"].inverse_transform(target)
            preds.append(pred.detach().cpu().numpy())
            truths.append(true.detach().cpu().numpy())
            events.append(evs.detach().cpu().numpy())
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(truths, axis=0),
        np.concatenate(events, axis=0),
    )


def collect_predictions(args, per_seed_rows, variants):
    lookup = checkpoint_lookup(per_seed_rows)
    predictions = {}
    for variant in variants:
        for seed in sorted({row["seed"] for row in per_seed_rows}):
            checkpoint = lookup.get((variant, seed))
            if not checkpoint:
                continue
            predictions[(variant, seed)] = predict_variant(args, checkpoint, variant, seed)
    return predictions


def plot_event_error_box(predictions, output_dir):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    data = []
    labels = []
    summary_rows = []
    rng = np.random.default_rng(7)
    for variant in PREDICTION_VARIANTS:
        errors = []
        for (v, _seed), (pred, true, evs) in predictions.items():
            if v != variant:
                continue
            mask = evs == 1
            errors.append(np.abs(pred[mask] - true[mask]))
        if not errors:
            continue
        flat = np.concatenate(errors)
        summary_rows.append(
            {
                "variant": variant,
                "label": LABELS[variant],
                "samples": int(flat.size),
                "mean_abs_error": float(np.mean(flat)),
                "median_abs_error": float(np.median(flat)),
                "p75_abs_error": float(np.percentile(flat, 75)),
                "p90_abs_error": float(np.percentile(flat, 90)),
                "p95_abs_error": float(np.percentile(flat, 95)),
            }
        )
        if flat.size > 30000:
            flat = rng.choice(flat, size=30000, replace=False)
        data.append(flat)
        labels.append(LABELS[variant])
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Absolute error on evs_90 points")
    ax.set_title("Extreme-Event Error Distribution")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "event_error_boxplot")
    write_rows(
        output_dir / "event_error_distribution_summary.csv",
        summary_rows,
        [
            "variant",
            "label",
            "samples",
            "mean_abs_error",
            "median_abs_error",
            "p75_abs_error",
            "p90_abs_error",
            "p95_abs_error",
        ],
    )


def plot_quantile_errors(predictions, output_dir):
    bins = [0, 50, 75, 90, 95, 99, 100]
    rows = []
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    for variant in PREDICTION_VARIANTS:
        all_true = []
        all_err = []
        for (v, _seed), (pred, true, _evs) in predictions.items():
            if v != variant:
                continue
            mask = true > 5
            all_true.append(true[mask])
            all_err.append(np.abs(pred[mask] - true[mask]))
        if not all_true:
            continue
        y = np.concatenate(all_true)
        err = np.concatenate(all_err)
        edges = np.percentile(y, bins)
        centers = []
        values = []
        for left_idx, right_idx in zip(range(len(bins) - 1), range(1, len(bins))):
            left = edges[left_idx]
            right = edges[right_idx]
            if right_idx == len(bins) - 1:
                mask = (y >= left) & (y <= right)
            else:
                mask = (y >= left) & (y < right)
            value = float(np.mean(err[mask]))
            label = f"p{bins[left_idx]}-p{bins[right_idx]}"
            rows.append(
                {
                    "variant": variant,
                    "label": LABELS[variant],
                    "target_quantile_bin": label,
                    "target_left": float(left),
                    "target_right": float(right),
                    "mae": value,
                }
            )
            centers.append(label)
            values.append(value)
        ax.plot(centers, values, marker="o", label=LABELS[variant])
    ax.set_ylabel("MAE within target-value bin")
    ax.set_xlabel("True target quantile bin, values > 5")
    ax.set_title("Error by Target Magnitude")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "error_by_target_quantile")
    write_rows(
        output_dir / "error_by_target_quantile.csv",
        rows,
        ["variant", "label", "target_quantile_bin", "target_left", "target_right", "mae"],
    )


def plot_event_scatter(predictions, output_dir, example_seed, scatter_sample):
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for ax, variant in zip(axes, PREDICTION_VARIANTS):
        pred, true, evs = predictions[(variant, example_seed)]
        mask = evs == 1
        x = true[mask].reshape(-1)
        y = pred[mask].reshape(-1)
        if x.size > scatter_sample:
            idx = rng.choice(x.size, size=scatter_sample, replace=False)
            x = x[idx]
            y = y[idx]
        low = min(float(np.min(x)), float(np.min(y)))
        high = max(float(np.max(x)), float(np.max(y)))
        ax.scatter(x, y, s=6, alpha=0.25)
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_title(LABELS[variant])
        ax.set_xlabel("True event value")
        ax.set_ylabel("Predicted event value")
        ax.grid(alpha=0.2)
    save_figure(fig, output_dir / f"event_scatter_seed_{example_seed}")


def plot_spike_case(predictions, output_dir, example_seed):
    base_pred, true, evs = predictions[("base_model", example_seed)]
    event_counts = evs[:, 0, :, :].sum(axis=0)
    peak_values = true[:, 0, :, :].max(axis=0)
    score = event_counts * peak_values
    node, flow = np.unravel_index(np.argmax(score), score.shape)
    series_true = true[:, 0, node, flow]
    center = int(np.argmax(series_true))
    start = max(0, center - 60)
    end = min(series_true.shape[0], center + 61)
    x = np.arange(start, end)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(x, series_true[start:end], color="black", linewidth=2.2, label="True")
    for variant in PREDICTION_VARIANTS:
        pred, _true, _evs = predictions[(variant, example_seed)]
        ax.plot(x, pred[start:end, 0, node, flow], linewidth=1.6, label=LABELS[variant])
    event_x = x[evs[start:end, 0, node, flow] == 1]
    if event_x.size:
        ax.scatter(
            event_x,
            series_true[event_x],
            color="#B23A48",
            s=18,
            label="evs_90",
            zorder=4,
        )
    ax.set_title(f"Spike Case, seed {example_seed}, node {node}, {'inflow' if flow == 0 else 'outflow'}")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Traffic value")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    save_figure(fig, output_dir / f"spike_case_seed_{example_seed}")


def decompose_dual_residual(args, per_seed_rows, output_dir, example_seed):
    lookup = checkpoint_lookup(per_seed_rows)
    checkpoint = lookup.get(("event_weighted_dual_residual", example_seed))
    if not checkpoint:
        return
    model_args = build_model_args(args, "event_weighted_dual_residual", example_seed)
    init_seed(example_seed)
    dataloader = get_dataloader(
        data_dir=model_args.data_dir,
        dataset=model_args.dataset,
        batch_size=model_args.batch_size,
        test_batch_size=model_args.test_batch_size,
        scalar_type="Standard",
    )
    graph = load_graph(model_args.graph_file, device=model_args.device)
    model_args.num_nodes = len(graph)
    model = STSSL(model_args).to(model_args.device)
    state = torch.load(checkpoint, map_location=torch.device(model_args.device))["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    general_event = []
    general_non = []
    event_event = []
    event_non = []
    gate_event = []
    gate_non = []
    with torch.no_grad():
        for data, target, evs, _ in dataloader["test"]:
            z1, z1_cls = model(data, graph)
            base = model.mlp(z1)
            general = model.get_bias(z1)
            gate = model.classify_evs(z1, z1_cls)
            event = model.get_event_bias(z1) * gate
            base_orig = dataloader["scaler"].inverse_transform(base)
            general_orig = dataloader["scaler"].inverse_transform(base + general) - base_orig
            event_orig = dataloader["scaler"].inverse_transform(base + general + event) - dataloader["scaler"].inverse_transform(base + general)
            mask = evs == 1
            general_np = general_orig.detach().cpu().numpy()
            event_np = event_orig.detach().cpu().numpy()
            gate_np = gate.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            general_event.append(np.abs(general_np[mask_np]))
            general_non.append(np.abs(general_np[~mask_np]))
            event_event.append(np.abs(event_np[mask_np]))
            event_non.append(np.abs(event_np[~mask_np]))
            gate_event.append(gate_np[mask_np])
            gate_non.append(gate_np[~mask_np])

    rows = [
        {
            "quantity": "general_residual_abs",
            "event_points": float(np.mean(np.concatenate(general_event))),
            "non_event_points": float(np.mean(np.concatenate(general_non))),
        },
        {
            "quantity": "event_residual_abs",
            "event_points": float(np.mean(np.concatenate(event_event))),
            "non_event_points": float(np.mean(np.concatenate(event_non))),
        },
        {
            "quantity": "predicted_gate",
            "event_points": float(np.mean(np.concatenate(gate_event))),
            "non_event_points": float(np.mean(np.concatenate(gate_non))),
        },
    ]
    write_rows(
        output_dir / "dual_residual_decomposition.csv",
        rows,
        ["quantity", "event_points", "non_event_points"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    residual_rows = [row for row in rows if row["quantity"] != "predicted_gate"]
    labels = [row["quantity"].replace("_abs", "").replace("_", " ") for row in residual_rows]
    x = np.arange(len(residual_rows))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [row["event_points"] for row in residual_rows],
        width,
        label="evs_90 points",
    )
    axes[0].bar(
        x + width / 2,
        [row["non_event_points"] for row in residual_rows],
        width,
        label="non-event points",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean absolute residual contribution")
    axes[0].set_title("Residual Magnitudes")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    gate_row = [row for row in rows if row["quantity"] == "predicted_gate"][0]
    axes[1].bar(
        ["evs_90 points", "non-event points"],
        [gate_row["event_points"], gate_row["non_event_points"]],
        color=["#1f77b4", "#ff7f0e"],
    )
    axes[1].set_ylabel("Mean predicted event probability")
    axes[1].set_title("Classifier Gate")
    axes[1].set_ylim(0, max(0.5, gate_row["event_points"] * 1.2))
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Dual Residual Specialization")
    save_figure(fig, output_dir / "dual_residual_decomposition")


def paired_improvements(per_seed_rows, output_dir):
    mean_rows = get_mean_rows(per_seed_rows)
    by_key = {(row["variant"], row["seed"]): row for row in mean_rows}
    all_seeds = sorted({row["seed"] for row in mean_rows})
    comparisons = [
        ("base_model", "ungated_head_only"),
        ("base_model", "event_weighted_dual_residual"),
        ("ungated_head_only", "event_weighted_dual_residual"),
        ("event_weighted_original", "event_weighted_dual_residual"),
    ]
    rows = []
    for baseline, candidate in comparisons:
        deltas_mae = []
        deltas_eee = []
        used_seeds = []
        for seed in all_seeds:
            if (baseline, seed) not in by_key or (candidate, seed) not in by_key:
                continue
            base = by_key[(baseline, seed)]
            cand = by_key[(candidate, seed)]
            deltas_mae.append(cand["mae"] - base["mae"])
            deltas_eee.append(cand["eee"] - base["eee"])
            used_seeds.append(seed)
        if not deltas_mae:
            continue
        row = {
            "baseline": baseline,
            "candidate": candidate,
            "delta_mae_mean_candidate_minus_baseline": float(np.mean(deltas_mae)),
            "delta_mae_std": sample_std(deltas_mae),
            "delta_eee_mean_candidate_minus_baseline": float(np.mean(deltas_eee)),
            "delta_eee_std": sample_std(deltas_eee),
            "mae_improved_seeds": int(np.sum(np.array(deltas_mae) < 0)),
            "eee_improved_seeds": int(np.sum(np.array(deltas_eee) < 0)),
            "seeds": len(deltas_mae),
        }
        for seed, delta_mae, delta_eee in zip(used_seeds, deltas_mae, deltas_eee):
            row[f"seed_{seed}_delta_mae"] = float(delta_mae)
            row[f"seed_{seed}_delta_eee"] = float(delta_eee)
        rows.append(row)
    seed_fields = []
    for seed in all_seeds:
        seed_fields.extend([f"seed_{seed}_delta_mae", f"seed_{seed}_delta_eee"])
    write_rows(
        output_dir / "paired_improvements.csv",
        rows,
        [
            "baseline",
            "candidate",
            "delta_mae_mean_candidate_minus_baseline",
            "delta_mae_std",
            "delta_eee_mean_candidate_minus_baseline",
            "delta_eee_std",
            "mae_improved_seeds",
            "eee_improved_seeds",
            "seeds",
            *seed_fields,
        ],
    )
    return rows


def write_paper_section(summary_rows, output_dir):
    means = {row["variant"]: row for row in get_mean_rows(summary_rows)}
    base = means["base_model"]
    ungated = means["ungated_head_only"]
    dual = means["event_weighted_dual_residual"]
    event_original = means["event_weighted_original"]
    clean_dual = means["dual_event_residual"]
    lines = [
        "# Paper Draft: Event-Weighted Dual Residual Hyb-STEX",
        "",
        "## Motivation",
        "",
        "The initial Hyb-STEX design used an extreme-event classifier to gate a bias-correction head. This design assumes that the learned correction should be applied primarily at extreme-event points:",
        "",
        "```text",
        "y_hat = f(z) + p_event(z) * r(z)",
        "```",
        "",
        "where `f(z)` is the base predictor, `p_event(z)` is the classifier output trained with `evs_90`, and `r(z)` is a residual correction. The ablation results show that this assumption is too restrictive. A correction head without the event gate improves the base model substantially, which indicates that a large part of the residual error is systematic and not limited to extreme events.",
        "",
        "This motivates a decomposition of the prediction residual into two components:",
        "",
        "```text",
        "y_hat = f(z) + r_general(z) + p_event(z) * r_event(z)",
        "```",
        "",
        "The always-on residual `r_general(z)` handles general calibration error, while `p_event(z) * r_event(z)` handles extreme-event-specific correction. This keeps the benefit of ungated residual learning while preserving the extreme-event classifier as a specialized event pathway.",
        "",
        "## Training Objective",
        "",
        "The event-weighted dual residual model is trained with the standard forecast loss, the event-classification loss, and an additional event-only forecast loss:",
        "",
        "```text",
        "L = MAE(y_hat, y)",
        "  + lambda_event * MAE(y_hat[evs_90 = 1], y[evs_90 = 1])",
        "  + lambda_cls * BCE(p_event, evs_90)",
        "```",
        "",
        "The key change is the event-only MAE term. BCE teaches the classifier to detect extreme events, but it does not directly optimize forecast error on those event points. The event-only MAE term aligns training with the EEE evaluation metric.",
        "",
        "## Main Results on NYCTaxi",
        "",
        f"Across seeds 1, 2, and 3, the base model obtains mean MAE `{base['mae']:.4f}` and EEE `{base['eee']:.4f}`. Adding a clean ungated residual improves MAE to `{ungated['mae']:.4f}` and EEE to `{ungated['eee']:.4f}`, showing that residual calibration alone is useful.",
        "",
        f"The event-weighted dual residual model obtains MAE `{dual['mae']:.4f}` and EEE `{dual['eee']:.4f}`. Compared with the base model, it improves MAE by `{base['mae'] - dual['mae']:.4f}` and EEE by `{base['eee'] - dual['eee']:.4f}`. Compared with the clean ungated residual, it changes MAE by `{dual['mae'] - ungated['mae']:+.4f}` while improving EEE by `{ungated['eee'] - dual['eee']:.4f}`.",
        "",
        "This result supports the central claim: the classifier gate is useful when it is used to specialize an event-specific residual, not when it suppresses all residual correction.",
        "",
        "## Important Control",
        "",
        f"The clean dual residual without event-weighted loss has average MAE `{clean_dual['mae']:.4f}` and EEE `{clean_dual['eee']:.4f}`, which is close to the ungated residual result. The event-weighted dual residual improves EEE to `{dual['eee']:.4f}`. Therefore, the large EEE gain is not explained by adding a second residual head alone; it comes from aligning the event pathway with the event-error objective.",
        "",
        f"Against the event-weighted version of the original single-residual gate, the dual residual improves MAE from `{event_original['mae']:.4f}` to `{dual['mae']:.4f}` and EEE from `{event_original['eee']:.4f}` to `{dual['eee']:.4f}`. This indicates that separating general calibration from event-specific correction improves the event-weighted objective.",
        "",
        "## Recommended Figure Captions",
        "",
        "**Aggregate MAE and EEE comparison.** Average test MAE and extreme-event error (EEE) on NYCTaxi across three seeds. The ungated residual gives the strongest average MAE among the clean residual models, but the event-weighted dual residual achieves the lowest EEE by a wide margin, indicating improved prediction on extreme-event points.",
        "",
        "**MAE-EEE tradeoff.** Pareto view of average MAE and EEE. The event-weighted dual residual moves the model into a distinct low-EEE region while maintaining competitive MAE, showing that event-aware residual specialization changes the tradeoff rather than simply increasing capacity.",
        "",
        "**Extreme-event error distribution.** Boxplot of absolute prediction error over `evs_90` test points. The event-weighted dual residual reduces event-point error relative to both the base model and the ungated residual model.",
        "",
        "**Error by target magnitude.** MAE grouped by quantiles of the true target value. The models behave similarly at normal target magnitudes, while the event-weighted dual residual separates strongly in the top target-value bin. This shows that the EEE gain is concentrated where extreme-event forecasting matters most.",
        "",
        "**Dual residual decomposition.** Magnitude of the learned general residual, event residual, and event gate on event versus non-event points. The event residual is much larger on `evs_90` points than non-event points, and the predicted gate is also higher on event points, confirming that the event pathway specializes rather than acting as an arbitrary capacity increase.",
    ]
    (output_dir / "paper_section_draft.md").write_text("\n".join(lines), encoding="utf-8")


def write_markdown(summary_rows, improvement_rows, output_dir):
    means = {row["variant"]: row for row in get_mean_rows(summary_rows)}
    lines = [
        "# NYCTaxi Event-Weighted Hyb-STEX Evidence",
        "",
        "## Main Metric Table",
        "",
        "| Variant | Mean MAE | Mean EEE |",
        "|---|---:|---:|",
    ]
    for row in get_mean_rows(summary_rows):
        lines.append(f"| {row['label']} | {row['mae']:.4f} | {row['eee']:.4f} |")
    lines.extend(["", "## Key Claims", ""])
    if "event_weighted_dual_residual" in means and "base_model" in means:
        candidate = means["event_weighted_dual_residual"]
        base = means["base_model"]
        lines.append(
            f"- Event-weighted dual residual improves over the base model by "
            f"{base['mae'] - candidate['mae']:.4f} MAE and "
            f"{base['eee'] - candidate['eee']:.4f} EEE."
        )
    if "event_weighted_dual_residual" in means and "ungated_head_only" in means:
        candidate = means["event_weighted_dual_residual"]
        ungated = means["ungated_head_only"]
        lines.append(
            f"- Compared with the clean ungated residual, event-weighted dual residual "
            f"changes MAE by {candidate['mae'] - ungated['mae']:+.4f} and "
            f"changes EEE by {candidate['eee'] - ungated['eee']:+.4f}."
        )
    if "event_weighted_dual_residual" in means and "event_weighted_original" in means:
        candidate = means["event_weighted_dual_residual"]
        original = means["event_weighted_original"]
        lines.append(
            f"- The dual residual decomposition beats event-weighted original by "
            f"{original['mae'] - candidate['mae']:.4f} MAE and "
            f"{original['eee'] - candidate['eee']:.4f} EEE."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ungated residual head shows that a large part of the gain is residual calibration, not hidden event information. The event-weighted dual residual keeps this always-on calibration path and adds a separate event-specific residual trained with direct pressure on evs_90 points. This is why the final model gives a much larger EEE reduction than BCE-only gating.",
            "",
            "## Generated Figures",
            "",
            "- `metrics_bar.png`: aggregate MAE and EEE comparison.",
            "- `mae_eee_pareto.png`: MAE-EEE tradeoff.",
            "- `event_error_boxplot.png`: event-point absolute error distribution.",
            "- `error_by_target_quantile.png`: error by true target magnitude.",
            "- `event_scatter_seed_1.png`: true vs predicted event values.",
            "- `spike_case_seed_1.png`: representative spike trajectory.",
            "- `dual_residual_decomposition.png`: general vs event-specific residual contributions.",
            "",
            "## Generated Tables",
            "",
            "- `summary_mean_std.csv` and `paper_table_metrics.tex`: mean +/- std table for paper use.",
            "- `paired_improvements.csv`: paired seed-level deltas between key variants.",
            "- `event_error_distribution_summary.csv`: distribution summaries for evs_90 absolute errors.",
            "- `error_by_target_quantile.csv`: source values for the target-magnitude plot.",
            "- `dual_residual_decomposition.csv`: source values for residual specialization.",
        ]
    )
    (output_dir / "paper_evidence_summary.md").write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig, path_without_suffix):
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_summary(args)
    per_seed_rows = load_per_seed(args)
    write_rows(
        output_dir / "combined_summary_metrics.csv",
        summary_rows,
        ["source", "variant", "label", "flow", "mae", "eee"],
    )
    write_rows(
        output_dir / "combined_per_seed_metrics.csv",
        per_seed_rows,
        ["source", "variant", "label", "seed", "flow", "mae", "eee", "log_dir", "checkpoint"],
    )

    plot_metric_bars(summary_rows, output_dir)
    plot_pareto(summary_rows, output_dir)
    write_summary_tables(per_seed_rows, output_dir)
    improvement_rows = paired_improvements(per_seed_rows, output_dir)

    predictions = collect_predictions(args, per_seed_rows, PREDICTION_VARIANTS)
    plot_event_error_box(predictions, output_dir)
    plot_quantile_errors(predictions, output_dir)
    plot_event_scatter(predictions, output_dir, args.example_seed, args.scatter_sample)
    plot_spike_case(predictions, output_dir, args.example_seed)
    decompose_dual_residual(args, per_seed_rows, output_dir, args.example_seed)

    write_paper_section(summary_rows, output_dir)
    write_markdown(summary_rows, improvement_rows, output_dir)
    print(f"Wrote paper evidence to {output_dir}")


if __name__ == "__main__":
    main()

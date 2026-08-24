import argparse
import csv
import hashlib
import math
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
from lib.utils import init_seed, load_graph
from model.models import STSSL
from scripts.prepare_nyctaxi_event_weighted_paper_evidence import (
    LABELS,
    build_model_args,
    read_rows,
    repo_path,
    save_figure,
    write_rows,
)


FLOW_NAMES = ("inflow", "outflow")
PLOT_VARIANTS = [
    "base_model",
    "ungated_head_only",
    "ungated_residual_frozen_base",
    "event_weighted_dual_residual",
]
LOCAL_LABELS = dict(LABELS)
LOCAL_LABELS["ungated_residual_frozen_base"] = "Frozen-base ungated"
MODEL_VARIANT = {
    "ungated_residual_frozen_base": "ungated_head_only",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Mine full-test-set predictions for high-delta cases and diagnose why "
            "ungated residual correction improves over the base model."
        )
    )
    parser.add_argument("--config-filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--base-results", default="nyctaxi_base_and_event_weighted_results")
    parser.add_argument("--clean-results", default="nyctaxi_event_gate_ablation_results")
    parser.add_argument(
        "--frozen-results",
        default="nyctaxi_frozen_ungated_residual_results",
        help="Optional results directory from run_nyctaxi_frozen_ungated_residual.py.",
    )
    parser.add_argument("--data-dir", default="preprocessed_data")
    parser.add_argument("--graph-file", default="preprocessed_data/NYCTaxi/adj_mx.npz")
    parser.add_argument("--output-dir", default="nyctaxi_residual_gain_diagnostics")
    parser.add_argument("--device", default=None)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--example-seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--case-context", type=int, default=48)
    parser.add_argument("--min-case-gap", type=int, default=36)
    parser.add_argument("--cascade-max-per-node-flow", type=int, default=2)
    return parser.parse_args()


def as_int(value):
    return int(float(value))


def to_float(value):
    return float(value) if value not in ("", None) else np.nan


def load_per_seed(args):
    sources = [
        ("base_event_weighted", repo_path(args.base_results) / "per_seed_metrics.csv"),
        ("clean_head_only", repo_path(args.clean_results) / "per_seed_metrics.csv"),
        ("frozen_base", repo_path(args.frozen_results) / "per_seed_metrics.csv"),
    ]
    rows = []
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
                    "label": LOCAL_LABELS.get(row["variant"], row["variant"]),
                    "seed": as_int(row["seed"]),
                    "flow": row["flow"],
                    "mae": to_float(row["mae"]),
                    "eee": to_float(row["eee"]),
                    "log_dir": row.get("log_dir", ""),
                    "checkpoint": row.get("checkpoint", ""),
                }
            )
    return rows


def checkpoint_lookup(per_seed_rows):
    lookup = {}
    for row in per_seed_rows:
        if row["flow"] == "mean" and row.get("checkpoint"):
            lookup[(row["variant"], row["seed"])] = row["checkpoint"]
    return lookup


def prediction_phase(variant):
    if variant == "base_model":
        return "pred"
    return "pred_2"


def build_args_for_variant(args, variant, seed):
    model_variant = MODEL_VARIANT.get(variant, variant)
    return build_model_args(args, model_variant, seed)


def inverse_delta(scaler, base_scaled, delta_scaled):
    return scaler.inverse_transform(base_scaled + delta_scaled) - scaler.inverse_transform(base_scaled)


def predict_components(args, checkpoint, variant, seed):
    model_args = build_args_for_variant(args, variant, seed)
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

    out = {
        "base": [],
        "full": [],
        "general_residual": [],
        "event_residual": [],
        "gate": [],
        "true": [],
        "evs": [],
    }
    phase = prediction_phase(variant)
    with torch.no_grad():
        for data, target, evs, _ in dataloader["test"]:
            z1, z1_cls = model(data, graph)
            base_scaled = model.mlp(z1)
            base_orig = dataloader["scaler"].inverse_transform(base_scaled)

            general_scaled = torch.zeros_like(base_scaled)
            event_scaled = torch.zeros_like(base_scaled)
            gate = torch.zeros_like(base_scaled)
            if phase == "pred":
                full_scaled = base_scaled
            else:
                general_scaled = model.get_bias(z1)
                if model.uses_ungated_bias_ablation():
                    full_scaled = base_scaled + general_scaled
                else:
                    gate = model.classify_evs(z1, z1_cls)
                    if model.uses_dual_event_residual_ablation():
                        event_scaled = model.get_event_bias(z1) * gate
                        full_scaled = base_scaled + general_scaled + event_scaled
                    else:
                        full_scaled = base_scaled + general_scaled * gate

            out["base"].append(base_orig.detach().cpu().numpy())
            out["full"].append(
                dataloader["scaler"].inverse_transform(full_scaled).detach().cpu().numpy()
            )
            out["general_residual"].append(
                inverse_delta(dataloader["scaler"], base_scaled, general_scaled)
                .detach()
                .cpu()
                .numpy()
            )
            if model.uses_dual_event_residual_ablation():
                event_orig = (
                    dataloader["scaler"].inverse_transform(base_scaled + general_scaled + event_scaled)
                    - dataloader["scaler"].inverse_transform(base_scaled + general_scaled)
                )
            else:
                event_orig = torch.zeros_like(base_scaled)
            out["event_residual"].append(event_orig.detach().cpu().numpy())
            out["gate"].append(gate.detach().cpu().numpy())
            out["true"].append(
                dataloader["scaler"].inverse_transform(target).detach().cpu().numpy()
            )
            out["evs"].append(evs.detach().cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in out.items()}


def collect_components(args, per_seed_rows):
    lookup = checkpoint_lookup(per_seed_rows)
    components = {}
    for seed in args.seeds:
        for variant in PLOT_VARIANTS:
            checkpoint = lookup.get((variant, seed))
            if not checkpoint:
                continue
            components[(variant, seed)] = predict_components(args, checkpoint, variant, seed)
    return components


def mean_abs(pred, true, mask):
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def component_metric_rows(components, output_dir):
    rows = []
    for seed in sorted({seed for _variant, seed in components}):
        base = components.get(("base_model", seed))
        ungated = components.get(("ungated_head_only", seed))
        frozen = components.get(("ungated_residual_frozen_base", seed))
        event_dual = components.get(("event_weighted_dual_residual", seed))
        if base is None or ungated is None:
            continue
        true = base["true"]
        evs = base["evs"]
        model_outputs = {
            "phase1_base": base["full"],
            "ungated_final_base_only": ungated["base"],
            "ungated_full": ungated["full"],
        }
        if frozen is not None:
            model_outputs["frozen_base_ungated_full"] = frozen["full"]
        if event_dual is not None:
            model_outputs["event_weighted_dual_full"] = event_dual["full"]

        for component, pred in model_outputs.items():
            for flow_idx, flow in enumerate(FLOW_NAMES):
                mae_mask = true[..., flow_idx] > 5
                event_mask = evs[..., flow_idx] == 1
                non_event_mask = mae_mask & ~event_mask
                rows.append(
                    {
                        "seed": seed,
                        "component": component,
                        "flow": flow,
                        "mae": mean_abs(pred[..., flow_idx], true[..., flow_idx], mae_mask),
                        "eee": mean_abs(pred[..., flow_idx], true[..., flow_idx], event_mask),
                        "non_event_mae": mean_abs(
                            pred[..., flow_idx],
                            true[..., flow_idx],
                            non_event_mask,
                        ),
                    }
                )
            flow_rows = rows[-2:]
            rows.append(
                {
                    "seed": seed,
                    "component": component,
                    "flow": "mean",
                    "mae": float(np.nanmean([row["mae"] for row in flow_rows])),
                    "eee": float(np.nanmean([row["eee"] for row in flow_rows])),
                    "non_event_mae": float(
                        np.nanmean([row["non_event_mae"] for row in flow_rows])
                    ),
                }
            )

    write_rows(
        output_dir / "base_to_ungated_component_metrics.csv",
        rows,
        ["seed", "component", "flow", "mae", "eee", "non_event_mae"],
    )

    summary = []
    for component in sorted({row["component"] for row in rows}):
        for flow in (*FLOW_NAMES, "mean"):
            selected = [row for row in rows if row["component"] == component and row["flow"] == flow]
            if not selected:
                continue
            summary.append(
                {
                    "seed": "average",
                    "component": component,
                    "flow": flow,
                    "mae": float(np.nanmean([row["mae"] for row in selected])),
                    "eee": float(np.nanmean([row["eee"] for row in selected])),
                    "non_event_mae": float(
                        np.nanmean([row["non_event_mae"] for row in selected])
                    ),
                }
            )
    write_rows(
        output_dir / "base_to_ungated_component_summary.csv",
        summary,
        ["seed", "component", "flow", "mae", "eee", "non_event_mae"],
    )
    return rows, summary


def residual_alignment_rows(components, output_dir):
    rows = []
    for seed in sorted({seed for variant, seed in components if variant == "ungated_head_only"}):
        comp = components[("ungated_head_only", seed)]
        true = comp["true"]
        base = comp["base"]
        full = comp["full"]
        residual = full - base
        evs = comp["evs"]
        for flow_idx, flow in enumerate(FLOW_NAMES):
            masks = {
                "all_gt5": true[..., flow_idx] > 5,
                "event": evs[..., flow_idx] == 1,
                "non_event_gt5": (true[..., flow_idx] > 5) & (evs[..., flow_idx] == 0),
            }
            for subset, mask in masks.items():
                if not np.any(mask):
                    continue
                error_before = true[..., flow_idx][mask] - base[..., flow_idx][mask]
                residual_values = residual[..., flow_idx][mask]
                before_abs = np.abs(error_before)
                after_abs = np.abs(true[..., flow_idx][mask] - full[..., flow_idx][mask])
                if np.std(error_before) > 0 and np.std(residual_values) > 0:
                    corr = float(np.corrcoef(error_before, residual_values)[0, 1])
                else:
                    corr = np.nan
                active = np.abs(residual_values) > 1e-6
                rows.append(
                    {
                        "seed": seed,
                        "flow": flow,
                        "subset": subset,
                        "samples": int(mask.sum()),
                        "mae_before_residual": float(np.mean(before_abs)),
                        "mae_after_residual": float(np.mean(after_abs)),
                        "delta_after_minus_before": float(np.mean(after_abs) - np.mean(before_abs)),
                        "mean_signed_base_error_true_minus_pred": float(np.mean(error_before)),
                        "mean_signed_residual": float(np.mean(residual_values)),
                        "mean_abs_residual": float(np.mean(np.abs(residual_values))),
                        "residual_error_correlation": corr,
                        "correct_direction_rate_active": (
                            float(np.mean(error_before[active] * residual_values[active] > 0))
                            if np.any(active)
                            else np.nan
                        ),
                    }
                )
    write_rows(
        output_dir / "ungated_residual_alignment.csv",
        rows,
        [
            "seed",
            "flow",
            "subset",
            "samples",
            "mae_before_residual",
            "mae_after_residual",
            "delta_after_minus_before",
            "mean_signed_base_error_true_minus_pred",
            "mean_signed_residual",
            "mean_abs_residual",
            "residual_error_correlation",
            "correct_direction_rate_active",
        ],
    )
    return rows


def quantile_driver_plot(components, output_dir):
    bins = [0, 50, 75, 90, 95, 99, 100]
    true_values = []
    phase1_values = []
    final_base_values = []
    full_values = []
    residual_values = []
    for seed in sorted({seed for variant, seed in components if variant == "ungated_head_only"}):
        base = components.get(("base_model", seed))
        ungated = components.get(("ungated_head_only", seed))
        if base is None or ungated is None:
            continue
        true = base["true"]
        mask = true > 5
        true_values.append(true[mask])
        phase1_values.append(base["full"][mask])
        final_base_values.append(ungated["base"][mask])
        full_values.append(ungated["full"][mask])
        residual_values.append((ungated["full"] - ungated["base"])[mask])

    if not true_values:
        return []

    true_all = np.concatenate(true_values)
    phase1_all = np.concatenate(phase1_values)
    final_base_all = np.concatenate(final_base_values)
    full_all = np.concatenate(full_values)
    residual_all = np.concatenate(residual_values)
    edges = np.percentile(true_all, bins)

    rows = []
    labels = []
    phase1_mae = []
    final_base_mae = []
    full_mae = []
    signed_error = []
    signed_residual = []
    for idx in range(len(bins) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == len(bins) - 2:
            mask = (true_all >= left) & (true_all <= right)
        else:
            mask = (true_all >= left) & (true_all < right)
        label = f"p{bins[idx]}-p{bins[idx + 1]}"
        labels.append(label)
        p1 = float(np.mean(np.abs(phase1_all[mask] - true_all[mask])))
        fb = float(np.mean(np.abs(final_base_all[mask] - true_all[mask])))
        uf = float(np.mean(np.abs(full_all[mask] - true_all[mask])))
        err = float(np.mean(true_all[mask] - final_base_all[mask]))
        resid = float(np.mean(residual_all[mask]))
        phase1_mae.append(p1)
        final_base_mae.append(fb)
        full_mae.append(uf)
        signed_error.append(err)
        signed_residual.append(resid)
        rows.append(
            {
                "target_quantile_bin": label,
                "target_left": float(left),
                "target_right": float(right),
                "phase1_base_mae": p1,
                "ungated_final_base_only_mae": fb,
                "ungated_full_mae": uf,
                "phase1_to_final_base_delta": fb - p1,
                "residual_addition_delta": uf - fb,
                "mean_signed_error_before_residual_true_minus_pred": err,
                "mean_signed_residual": resid,
            }
        )

    write_rows(
        output_dir / "base_to_ungated_driver_by_target_quantile.csv",
        rows,
        [
            "target_quantile_bin",
            "target_left",
            "target_right",
            "phase1_base_mae",
            "ungated_final_base_only_mae",
            "ungated_full_mae",
            "phase1_to_final_base_delta",
            "residual_addition_delta",
            "mean_signed_error_before_residual_true_minus_pred",
            "mean_signed_residual",
        ],
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    ax.plot(labels, phase1_mae, marker="o", label="Phase-1 base")
    ax.plot(labels, final_base_mae, marker="o", label="Ungated final base only")
    ax.plot(labels, full_mae, marker="o", label="Ungated full")
    ax.set_title("Base-to-Ungated Gain by Target Magnitude")
    ax.set_xlabel("True target quantile bin, values > 5")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "base_to_ungated_mae_by_target_quantile")

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, signed_error, width, label="Mean base error: true - pred")
    ax.bar(x + width / 2, signed_residual, width, label="Mean learned residual")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Ungated Residual Tracks Signed Base Error")
    ax.set_xlabel("True target quantile bin, values > 5")
    ax.set_ylabel("Signed value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "ungated_residual_calibration_by_target_quantile")
    return rows


def select_cases(true, pred_a, pred_b, valid_mask, top_k, min_gap):
    score = np.abs(pred_a - true) - np.abs(pred_b - true)
    valid = valid_mask & (score > 0)
    candidate_indices = np.argwhere(valid)
    if candidate_indices.size == 0:
        return []
    candidate_scores = score[valid]
    order = np.argsort(candidate_scores)[::-1]
    selected = []
    for idx in order:
        sample, horizon, node, flow = [int(v) for v in candidate_indices[idx]]
        if any(
            node == prev["node"]
            and flow == prev["flow"]
            and abs(sample - prev["sample"]) < min_gap
            for prev in selected
        ):
            continue
        selected.append(
            {
                "sample": sample,
                "horizon": horizon,
                "node": node,
                "flow": flow,
                "score": float(candidate_scores[idx]),
            }
        )
        if len(selected) >= top_k:
            break
    return selected


def select_cascade_cases(
    true,
    base_pred,
    ungated_pred,
    event_dual_pred,
    evs,
    top_k,
    min_gap,
    max_per_node_flow,
):
    base_err = np.abs(base_pred - true)
    ungated_err = np.abs(ungated_pred - true)
    dual_err = np.abs(event_dual_pred - true)
    base_to_ungated_gain = base_err - ungated_err
    ungated_to_dual_gain = ungated_err - dual_err
    valid = (evs == 1) & (base_to_ungated_gain > 0) & (ungated_to_dual_gain > 0)
    candidate_indices = np.argwhere(valid)
    if candidate_indices.size == 0:
        return [], {
            "event_points": int(np.sum(evs == 1)),
            "cascade_points": 0,
            "cascade_fraction": 0.0,
        }

    # Prefer cases where both steps are meaningful instead of one huge step and one tiny step.
    score = np.sqrt(base_to_ungated_gain * ungated_to_dual_gain)
    candidate_scores = score[valid]
    order = np.argsort(candidate_scores)[::-1]
    selected = []
    per_node_flow = {}
    for idx in order:
        sample, horizon, node, flow = [int(v) for v in candidate_indices[idx]]
        key = (node, flow)
        if per_node_flow.get(key, 0) >= max_per_node_flow:
            continue
        if any(
            node == prev["node"]
            and flow == prev["flow"]
            and abs(sample - prev["sample"]) < min_gap
            for prev in selected
        ):
            continue
        selected.append(
            {
                "sample": sample,
                "horizon": horizon,
                "node": node,
                "flow": flow,
                "score": float(candidate_scores[idx]),
                "base_abs_error": float(base_err[sample, horizon, node, flow]),
                "ungated_abs_error": float(ungated_err[sample, horizon, node, flow]),
                "event_weighted_dual_abs_error": float(dual_err[sample, horizon, node, flow]),
                "base_to_ungated_gain": float(
                    base_to_ungated_gain[sample, horizon, node, flow]
                ),
                "ungated_to_event_weighted_dual_gain": float(
                    ungated_to_dual_gain[sample, horizon, node, flow]
                ),
                "base_to_event_weighted_dual_gain": float(
                    base_err[sample, horizon, node, flow]
                    - dual_err[sample, horizon, node, flow]
                ),
            }
        )
        per_node_flow[key] = per_node_flow.get(key, 0) + 1
        if len(selected) >= top_k:
            break

    event_points = int(np.sum(evs == 1))
    cascade_points = int(candidate_indices.shape[0])
    return selected, {
        "event_points": event_points,
        "cascade_points": cascade_points,
        "cascade_fraction": cascade_points / event_points if event_points else 0.0,
    }


def plot_case_grid(comps, cases, comparison_name, output_dir, context):
    if not cases:
        return
    true = next(iter(comps.values()))["true"]
    evs = next(iter(comps.values()))["evs"]
    rows = math.ceil(len(cases) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14.2, 3.6 * rows + 0.5))
    axes = np.asarray(axes).reshape(-1)
    variants_to_plot = [variant for variant in PLOT_VARIANTS if variant in comps]
    styles = {
        "base_model": {"color": "#1f77b4", "linewidth": 1.4, "linestyle": "-"},
        "ungated_head_only": {"color": "#ff7f0e", "linewidth": 1.5, "linestyle": "-"},
        "ungated_residual_frozen_base": {"color": "#9467bd", "linewidth": 1.4, "linestyle": "--"},
        "event_weighted_dual_residual": {"color": "#2ca02c", "linewidth": 1.6, "linestyle": "-"},
    }
    for ax, case in zip(axes, cases):
        sample = case["sample"]
        node = case["node"]
        flow = case["flow"]
        start = max(0, sample - context)
        end = min(true.shape[0], sample + context + 1)
        x = np.arange(start, end)
        ax.plot(x, true[start:end, 0, node, flow], color="black", linewidth=2.1, label="True")
        for variant in variants_to_plot:
            ax.plot(
                x,
                comps[variant]["full"][start:end, 0, node, flow],
                label=LOCAL_LABELS.get(variant, variant),
                **styles.get(variant, {}),
            )
        event_mask = evs[start:end, 0, node, flow] == 1
        if np.any(event_mask):
            ax.scatter(
                x[event_mask],
                true[start:end, 0, node, flow][event_mask],
                color="#B23A48",
                s=18,
                zorder=4,
                label="evs_90",
            )
        ax.axvline(sample, color="#444444", linewidth=1, alpha=0.8)
        horizon = case["horizon"]
        focal_true = true[sample, horizon, node, flow]
        ax.scatter([sample], [focal_true], color="black", s=54, zorder=5)
        for variant in variants_to_plot:
            style = styles.get(variant, {})
            ax.scatter(
                [sample],
                [comps[variant]["full"][sample, horizon, node, flow]],
                color=style.get("color", "#333333"),
                s=34,
                zorder=5,
            )
        flow_name = FLOW_NAMES[flow]
        if "ungated_to_event_weighted_dual_gain" in case:
            title = (
                f"node {node}, {flow_name}, t={sample}; "
                f"base>{case['base_to_ungated_gain']:.1f}, "
                f"dual>{case['ungated_to_event_weighted_dual_gain']:.1f}"
            )
        else:
            title = f"node {node}, {flow_name}, t={sample}, improvement={case['score']:.1f}"
        ax.set_title(
            title,
            fontsize=10,
        )
        ax.set_xlabel("Test sample index")
        ax.set_ylabel("Traffic value")
        ax.grid(alpha=0.22)
    for ax in axes[len(cases) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(comparison_name.replace("_", " "), y=0.995)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.965))
    save_figure(fig, output_dir / f"{comparison_name}_top_cases")


def top_case_plots(components, args, output_dir):
    seed = args.example_seed
    comps = {
        variant: components[(variant, seed)]
        for variant in PLOT_VARIANTS
        if (variant, seed) in components
    }
    if "base_model" not in comps or "ungated_head_only" not in comps:
        return []

    true = comps["base_model"]["true"]
    evs = comps["base_model"]["evs"]
    comparisons = [
        (
            "base_to_ungated_all_gt5",
            "base_model",
            "ungated_head_only",
            true > 5,
        ),
    ]
    if "event_weighted_dual_residual" in comps:
        comparisons.extend(
            [
                (
                    "base_to_event_weighted_dual_events",
                    "base_model",
                    "event_weighted_dual_residual",
                    evs == 1,
                ),
                (
                    "ungated_to_event_weighted_dual_events",
                    "ungated_head_only",
                    "event_weighted_dual_residual",
                    evs == 1,
                ),
            ]
        )
    if "ungated_residual_frozen_base" in comps:
        comparisons.append(
            (
                "base_to_frozen_ungated_all_gt5",
                "base_model",
                "ungated_residual_frozen_base",
                true > 5,
            )
        )

    rows = []
    for comparison_name, baseline, candidate, valid_mask in comparisons:
        cases = select_cases(
            true,
            comps[baseline]["full"],
            comps[candidate]["full"],
            valid_mask,
            args.top_k,
            args.min_case_gap,
        )
        for rank, case in enumerate(cases, start=1):
            sample = case["sample"]
            horizon = case["horizon"]
            node = case["node"]
            flow = case["flow"]
            row = {
                "comparison": comparison_name,
                "rank": rank,
                "sample": sample,
                "horizon": horizon,
                "node": node,
                "flow": FLOW_NAMES[flow],
                "true_value": float(true[sample, horizon, node, flow]),
                "evs_90": int(evs[sample, horizon, node, flow]),
                "baseline": baseline,
                "candidate": candidate,
                "baseline_abs_error": float(
                    abs(comps[baseline]["full"][sample, horizon, node, flow] - true[sample, horizon, node, flow])
                ),
                "candidate_abs_error": float(
                    abs(comps[candidate]["full"][sample, horizon, node, flow] - true[sample, horizon, node, flow])
                ),
                "improvement": case["score"],
            }
            for variant in PLOT_VARIANTS:
                if variant in comps:
                    row[f"{variant}_prediction"] = float(
                        comps[variant]["full"][sample, horizon, node, flow]
                    )
            rows.append(row)
        plot_case_grid(comps, cases, comparison_name, output_dir, args.case_context)

    fieldnames = [
        "comparison",
        "rank",
        "sample",
        "horizon",
        "node",
        "flow",
        "true_value",
        "evs_90",
        "baseline",
        "candidate",
        "baseline_abs_error",
        "candidate_abs_error",
        "improvement",
        *[f"{variant}_prediction" for variant in PLOT_VARIANTS],
    ]
    write_rows(output_dir / f"top_case_improvements_seed_{seed}.csv", rows, fieldnames)
    return rows


def cascade_case_plots(components, args, output_dir):
    required = {"base_model", "ungated_head_only", "event_weighted_dual_residual"}
    available_seeds = sorted(
        seed
        for seed in args.seeds
        if all((variant, seed) in components for variant in required)
    )
    rows = []
    summary_rows = []
    for seed in available_seeds:
        comps = {variant: components[(variant, seed)] for variant in PLOT_VARIANTS if (variant, seed) in components}
        true = comps["base_model"]["true"]
        evs = comps["base_model"]["evs"]
        cases, summary = select_cascade_cases(
            true,
            comps["base_model"]["full"],
            comps["ungated_head_only"]["full"],
            comps["event_weighted_dual_residual"]["full"],
            evs,
            args.top_k,
            args.min_case_gap,
            args.cascade_max_per_node_flow,
        )
        summary_rows.append(
            {
                "seed": seed,
                "event_points": summary["event_points"],
                "cascade_points": summary["cascade_points"],
                "cascade_fraction": summary["cascade_fraction"],
                "selected_cases": len(cases),
            }
        )
        comparison_name = f"ungated_then_event_weighted_dual_extreme_cascade_seed_{seed}"
        for rank, case in enumerate(cases, start=1):
            sample = case["sample"]
            horizon = case["horizon"]
            node = case["node"]
            flow = case["flow"]
            row = {
                "seed": seed,
                "rank": rank,
                "sample": sample,
                "horizon": horizon,
                "node": node,
                "flow": FLOW_NAMES[flow],
                "true_value": float(true[sample, horizon, node, flow]),
                "evs_90": int(evs[sample, horizon, node, flow]),
                "base_prediction": float(comps["base_model"]["full"][sample, horizon, node, flow]),
                "ungated_prediction": float(
                    comps["ungated_head_only"]["full"][sample, horizon, node, flow]
                ),
                "event_weighted_dual_prediction": float(
                    comps["event_weighted_dual_residual"]["full"][sample, horizon, node, flow]
                ),
                "base_abs_error": case["base_abs_error"],
                "ungated_abs_error": case["ungated_abs_error"],
                "event_weighted_dual_abs_error": case["event_weighted_dual_abs_error"],
                "base_to_ungated_gain": case["base_to_ungated_gain"],
                "ungated_to_event_weighted_dual_gain": case[
                    "ungated_to_event_weighted_dual_gain"
                ],
                "base_to_event_weighted_dual_gain": case["base_to_event_weighted_dual_gain"],
                "score": case["score"],
            }
            rows.append(row)
        plot_case_grid(comps, cases, comparison_name, output_dir, args.case_context)

    write_rows(
        output_dir / "ungated_then_event_weighted_dual_extreme_cascade_cases.csv",
        rows,
        [
            "seed",
            "rank",
            "sample",
            "horizon",
            "node",
            "flow",
            "true_value",
            "evs_90",
            "base_prediction",
            "ungated_prediction",
            "event_weighted_dual_prediction",
            "base_abs_error",
            "ungated_abs_error",
            "event_weighted_dual_abs_error",
            "base_to_ungated_gain",
            "ungated_to_event_weighted_dual_gain",
            "base_to_event_weighted_dual_gain",
            "score",
        ],
    )
    write_rows(
        output_dir / "ungated_then_event_weighted_dual_extreme_cascade_summary.csv",
        summary_rows,
        ["seed", "event_points", "cascade_points", "cascade_fraction", "selected_cases"],
    )
    return rows, summary_rows


def sample_hashes(samples):
    return {
        hashlib.sha1(np.ascontiguousarray(samples[idx]).view(np.uint8)).hexdigest()
        for idx in range(samples.shape[0])
    }


def leakage_checks(args, output_dir):
    data_root = repo_path(args.data_dir) / "NYCTaxi"
    arrays = {}
    rows = []
    for split in ("train", "val", "test"):
        loaded = np.load(data_root / f"{split}.npz")
        arrays[split] = {"x": loaded["x"], "y": loaded["y"], "evs": loaded["evs_90"]}
        x = arrays[split]["x"]
        y = arrays[split]["y"]
        rows.append({"check": f"{split}_x_shape", "value": str(tuple(x.shape)), "detail": ""})
        rows.append({"check": f"{split}_y_shape", "value": str(tuple(y.shape)), "detail": ""})
        last = x[:, -1:, :, :]
        full_last_match = np.all(last == y, axis=(1, 2, 3))
        mean_abs_last = float(np.mean(np.abs(last - y)))
        any_step_full = []
        step_mae = []
        for step in range(x.shape[1]):
            current = x[:, step : step + 1, :, :]
            any_step_full.append(np.all(current == y, axis=(1, 2, 3)))
            step_mae.append(float(np.mean(np.abs(current - y))))
        any_step_full = np.any(np.stack(any_step_full, axis=0), axis=0)
        rows.extend(
            [
                {
                    "check": f"{split}_fraction_target_equals_last_input_full_sample",
                    "value": float(np.mean(full_last_match)),
                    "detail": "Exact full-sample equality check.",
                },
                {
                    "check": f"{split}_fraction_target_equals_any_input_step_full_sample",
                    "value": float(np.mean(any_step_full)),
                    "detail": "Exact full-sample equality across any input timestep.",
                },
                {
                    "check": f"{split}_mean_abs_target_minus_last_input",
                    "value": mean_abs_last,
                    "detail": "Raw, unscaled arrays.",
                },
                {
                    "check": f"{split}_min_mean_abs_target_minus_any_input_step",
                    "value": float(np.min(step_mae)),
                    "detail": f"best_step={int(np.argmin(step_mae))}",
                },
            ]
        )

    hashes = {split: sample_hashes(arrays[split]["x"]) for split in arrays}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hashes[left] & hashes[right]
        rows.append(
            {
                "check": f"exact_duplicate_x_windows_{left}_vs_{right}",
                "value": len(overlap),
                "detail": "SHA1 of complete raw x window.",
            }
        )
    rows.append(
        {
            "check": "scaler_fit_scope",
            "value": "train+val",
            "detail": "lib.dataloader.normalize_data concatenates x_train and x_val only.",
        }
    )
    write_rows(output_dir / "dataset_leakage_checks.csv", rows, ["check", "value", "detail"])
    return rows


def write_summary(output_dir, component_summary, alignment_rows, quantile_rows, case_rows, leakage_rows):
    mean_rows = {
        row["component"]: row
        for row in component_summary
        if row["flow"] == "mean"
    }
    lines = [
        "# NYCTaxi Residual Gain Diagnostics",
        "",
        "## Base-to-Ungated Decomposition",
        "",
        "| Component | MAE | EEE | Non-event MAE |",
        "|---|---:|---:|---:|",
    ]
    for component in [
        "phase1_base",
        "ungated_final_base_only",
        "ungated_full",
        "frozen_base_ungated_full",
        "event_weighted_dual_full",
    ]:
        row = mean_rows.get(component)
        if not row:
            continue
        lines.append(
            f"| {component} | {row['mae']:.4f} | {row['eee']:.4f} | {row['non_event_mae']:.4f} |"
        )

    lines.extend(["", "## Interpretation Checks", ""])
    if "phase1_base" in mean_rows and "ungated_full" in mean_rows:
        base = mean_rows["phase1_base"]
        full = mean_rows["ungated_full"]
        lines.append(
            f"- Clean ungated improves over phase-1 base by {base['mae'] - full['mae']:.4f} MAE "
            f"and {base['eee'] - full['eee']:.4f} EEE."
        )
    if "ungated_final_base_only" in mean_rows and "ungated_full" in mean_rows:
        base_only = mean_rows["ungated_final_base_only"]
        full = mean_rows["ungated_full"]
        lines.append(
            f"- Within the ungated checkpoint, adding the residual changes MAE by "
            f"{full['mae'] - base_only['mae']:+.4f} and EEE by {full['eee'] - base_only['eee']:+.4f}."
        )
    if "frozen_base_ungated_full" in mean_rows:
        frozen = mean_rows["frozen_base_ungated_full"]
        base = mean_rows["phase1_base"]
        lines.append(
            f"- Frozen-base residual control improves over phase-1 base by "
            f"{base['mae'] - frozen['mae']:.4f} MAE and {base['eee'] - frozen['eee']:.4f} EEE."
        )

    all_alignment = [
        row for row in alignment_rows if row["flow"] == "inflow" and row["subset"] == "all_gt5"
    ]
    if all_alignment:
        corr = np.nanmean([row["residual_error_correlation"] for row in all_alignment])
        direction = np.nanmean([row["correct_direction_rate_active"] for row in all_alignment])
        lines.append(
            f"- Ungated residual alignment, inflow all_gt5: mean correlation {corr:.4f}, "
            f"active correct-direction rate {direction:.4f}."
        )

    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `*_top_cases.png/pdf`: full-test-set examples ranked by actual error improvement.",
            "- `top_case_improvements_seed_*.csv`: exact rows behind the selected cases.",
            "- `base_to_ungated_component_summary.csv`: phase-1 base, final base-only, and full residual decomposition.",
            "- `ungated_residual_alignment.csv`: whether the residual points in the direction of the base error.",
            "- `base_to_ungated_driver_by_target_quantile.csv`: where the gain occurs by target magnitude.",
            "- `dataset_leakage_checks.csv`: exact-window and target-in-input checks.",
        ]
    )
    (output_dir / "residual_gain_diagnostics_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = load_per_seed(args)
    components = collect_components(args, per_seed_rows)
    if not components:
        raise RuntimeError("No checkpoints were found for diagnostics.")

    component_rows, component_summary = component_metric_rows(components, output_dir)
    alignment_rows = residual_alignment_rows(components, output_dir)
    quantile_rows = quantile_driver_plot(components, output_dir)
    case_rows = top_case_plots(components, args, output_dir)
    cascade_case_rows, cascade_summary_rows = cascade_case_plots(components, args, output_dir)
    leakage_rows = leakage_checks(args, output_dir)
    write_summary(
        output_dir,
        component_summary,
        alignment_rows,
        quantile_rows,
        case_rows + cascade_case_rows,
        leakage_rows,
    )
    print(f"Wrote residual gain diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

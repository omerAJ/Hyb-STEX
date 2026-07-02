"""
Train and compare NYCTaxi three-phase residual variants.

Default command from the repo root:
  C:\\Users\\PCF\\.conda\\envs\\sds-test\\python.exe scripts\\run_nyctaxi_three_phase_residual_comparison.py

This script compares:
  - ungated_3phase: base + residual
  - gated_3phase: base + sigmoid(event_gate) * residual

It also imports the latest original 4-phase Hyb-STEX results, when available,
so the final tables include original_4phase as a baseline.

For ungated_3phase, the script can reuse rows from the latest
nyctaxi_ungated_residual_results run for matching seeds and train only missing
seeds. Use --no-reuse-existing-ungated to force a fresh ungated run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_nyctaxi_drift_experiment import (  # noqa: E402
    DEFAULT_MAIN_OPTIONS,
    EVENT_THRESHOLD_SOURCE_CHOICES,
    FLOW_NAMES,
    StandardScaler,
    add_repo_to_path,
    best_f1_threshold,
    build_dataloaders,
    f1_counts,
    get_model_params_grouped,
    init_seed,
    load_arrays,
    load_config,
    load_graph,
    parse_csv_list,
    parse_seed_list,
    preflight_environment,
    repo_root,
)


VARIANT_CHOICES = ("ungated_3phase", "gated_3phase")
PRIMARY_BASELINE_VARIANT = "original_4phase"
RESULT_COLUMNS = [
    "variant",
    "seed",
    "flow",
    "primary_mae",
    "primary_eee",
    "base_mae",
    "base_eee",
    "residual_mae",
    "residual_eee",
    "soft_mae",
    "soft_eee",
    "oracle_mae",
    "oracle_eee",
    "always_on_mae",
    "always_on_eee",
    "f1_tuned",
    "f1_05",
    "precision_tuned",
    "recall_tuned",
    "threshold",
    "val_best_f1",
    "event_prevalence",
    "best_val_loss",
    "best_val_epoch",
    "checkpoint",
    "log_dir",
    "source",
    "phase_results",
]
SUMMARY_METRICS = [
    "primary_mae",
    "primary_eee",
    "f1_tuned",
    "f1_05",
    "precision_tuned",
    "recall_tuned",
    "base_mae",
    "base_eee",
    "residual_mae",
    "residual_eee",
    "soft_mae",
    "soft_eee",
    "oracle_mae",
    "oracle_eee",
    "always_on_mae",
    "always_on_eee",
]


def build_run_args(
    configs: dict[str, Any],
    data_dir: Path,
    seed: int,
    device: str,
    variant: str,
    max_epochs: int | None,
) -> argparse.Namespace:
    run_configs = dict(configs)
    for key, value in DEFAULT_MAIN_OPTIONS.items():
        run_configs.setdefault(key, value)
    run_configs["seed"] = seed
    run_configs["mode"] = "train"
    run_configs["device"] = device
    run_configs["data_dir"] = str(data_dir)
    run_configs["dataset"] = "NYCTaxi"
    run_configs["graph_file"] = str(data_dir / "NYCTaxi" / "adj_mx.npz")
    run_configs["best_path"] = None
    run_configs["debug"] = False
    run_configs["phase3_mode"] = variant
    run_configs["comment"] = f"nyctaxi_{variant}"
    run_configs["experimentName"] = f"nyctaxi_{variant}_seed={seed}"
    if max_epochs is not None:
        run_configs["epochs"] = max_epochs
        run_configs["num_epochs"] = max_epochs
    return argparse.Namespace(**run_configs)


def set_group_trainable(
    pred_params: list[torch.nn.Parameter],
    classifier_params: list[torch.nn.Parameter],
    bias_params: list[torch.nn.Parameter],
    train_pred: bool,
    train_classifier: bool,
    train_bias: bool,
) -> None:
    for param in pred_params:
        param.requires_grad = train_pred
    for param in classifier_params:
        param.requires_grad = train_classifier
    for param in bias_params:
        param.requires_grad = train_bias


def build_phase_optimizer(
    pred_params: list[torch.nn.Parameter],
    classifier_params: list[torch.nn.Parameter],
    bias_params: list[torch.nn.Parameter],
    lr: float,
    train_pred: bool,
    train_classifier: bool,
    train_bias: bool,
    pred_lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    if train_pred:
        groups.append(
            {
                "params": pred_params,
                "lr": lr * pred_lr_scale,
                "eps": 1.0e-8,
                "weight_decay": 0,
                "amsgrad": False,
            }
        )
    if train_classifier:
        groups.append(
            {
                "params": classifier_params,
                "lr": lr,
                "eps": 1.0e-8,
                "weight_decay": 0,
                "amsgrad": True,
            }
        )
    if train_bias:
        groups.append(
            {
                "params": bias_params,
                "lr": lr,
                "eps": 1.0e-8,
                "weight_decay": 1.0e-8,
                "amsgrad": True,
            }
        )
    if not groups:
        raise ValueError("At least one parameter group must be trainable.")
    return torch.optim.Adam(groups)


def masked_mae_tensor(pred: torch.Tensor, true: torch.Tensor, mask_value: float = 5.0) -> torch.Tensor:
    mask = true > mask_value
    if not torch.any(mask):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return torch.mean(torch.abs(true[mask] - pred[mask]))


def weighted_mae_loss(
    pred_scaled: torch.Tensor,
    true_scaled: torch.Tensor,
    scaler: StandardScaler,
    yita: float,
) -> torch.Tensor:
    pred = scaler.inverse_transform(pred_scaled)
    true = scaler.inverse_transform(true_scaled)
    return (
        yita * masked_mae_tensor(pred[..., 0], true[..., 0])
        + (1.0 - yita) * masked_mae_tensor(pred[..., 1], true[..., 1])
    )


def gate_logits(model: torch.nn.Module, repr1: torch.Tensor) -> torch.Tensor:
    return model.mlp_cls(model.ff_to_cls(repr1))


def forward_parts(
    model: torch.nn.Module,
    data: torch.Tensor,
    graph: torch.Tensor,
    gate_temperature: float,
) -> dict[str, torch.Tensor]:
    repr1, _ = model(data, graph)
    logits = gate_logits(model, repr1)
    probs = torch.sigmoid(logits / gate_temperature)
    base = model.mlp(repr1)
    bias = model.get_bias(repr1)
    return {
        "base": base,
        "bias": bias,
        "logits": logits,
        "probs": probs,
        "residual": base + bias,
        "soft": base + (bias * probs),
    }


def prediction_for_mode(parts: dict[str, torch.Tensor], evs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "base":
        return parts["base"]
    if mode == "ungated":
        return parts["residual"]
    if mode == "gated_oracle":
        return parts["base"] + (parts["bias"] * evs)
    if mode == "gated_soft":
        return parts["soft"]
    raise ValueError(f"Unsupported phase mode: {mode}")


def make_pos_weight(arrays: dict[str, np.ndarray], device: str, mode: str, cap: float | None) -> torch.Tensor:
    if mode == "none":
        return torch.ones((1, 1, 1, len(FLOW_NAMES)), dtype=torch.float32, device=device)
    labels = torch.as_tensor(arrays["evs_train"], dtype=torch.float32, device=device)
    pos = labels.sum(dim=(0, 1, 2))
    total_per_flow = labels.shape[0] * labels.shape[1] * labels.shape[2]
    neg = torch.as_tensor(float(total_per_flow), dtype=torch.float32, device=device) - pos
    pos_weight = neg / torch.clamp(pos, min=1.0)
    if cap is not None and cap > 0:
        pos_weight = torch.clamp(pos_weight, max=float(cap))
    return pos_weight.view(1, 1, 1, -1)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.where(labels > 0.5, pos_weight, torch.ones_like(logits))
    return torch.mean(loss * weights)


def train_or_eval_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    graph: torch.Tensor,
    scaler: StandardScaler,
    args: argparse.Namespace,
    mode: str,
    lambda_cls: float,
    pos_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    gate_temperature: float,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_losses: list[float] = []
    pred_losses: list[float] = []
    cls_losses: list[float] = []

    for data, target, evs, _ in dataloader:
        if optimizer is not None:
            optimizer.zero_grad()

        parts = forward_parts(model, data, graph, gate_temperature)
        pred = prediction_for_mode(parts, evs, mode)
        pred_loss = weighted_mae_loss(pred, target, scaler, float(args.yita))

        if lambda_cls > 0:
            cls_loss = weighted_bce_with_logits(parts["logits"], evs, pos_weight)
        else:
            cls_loss = torch.zeros((), dtype=pred_loss.dtype, device=pred_loss.device)

        loss = pred_loss + (float(lambda_cls) * cls_loss)
        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss encountered in mode={mode}.")

        if optimizer is not None:
            loss.backward()
            if bool(args.grad_norm):
                trainable = [param for param in model.parameters() if param.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
            optimizer.step()

        total_losses.append(float(loss.item()))
        pred_losses.append(float(pred_loss.item()))
        cls_losses.append(float(cls_loss.item()))

    return (
        float(np.mean(total_losses)),
        float(np.mean(pred_losses)),
        float(np.mean(cls_losses)),
    )


def train_phase(
    model: torch.nn.Module,
    dataloaders: dict[str, Any],
    graph: torch.Tensor,
    scaler: StandardScaler,
    args: argparse.Namespace,
    pred_params: list[torch.nn.Parameter],
    classifier_params: list[torch.nn.Parameter],
    bias_params: list[torch.nn.Parameter],
    phase_name: str,
    mode: str,
    train_pred: bool,
    train_classifier: bool,
    train_bias: bool,
    lambda_cls: float,
    pos_weight: torch.Tensor,
    patience: int,
    checkpoint_dir: Path,
    pred_lr_scale: float = 1.0,
    gate_temperature: float = 1.0,
) -> dict[str, Any]:
    set_group_trainable(
        pred_params,
        classifier_params,
        bias_params,
        train_pred=train_pred,
        train_classifier=train_classifier,
        train_bias=train_bias,
    )
    optimizer = build_phase_optimizer(
        pred_params,
        classifier_params,
        bias_params,
        float(args.lr_init),
        train_pred=train_pred,
        train_classifier=train_classifier,
        train_bias=train_bias,
        pred_lr_scale=pred_lr_scale,
    )

    best_loss = math.inf
    best_epoch = 0
    not_improved = 0
    start_time = time.time()
    best_path = checkpoint_dir / f"best_model_{phase_name}.pth"
    val_loader = dataloaders["val"] if dataloaders["val"] is not None else dataloaders["test"]

    print(
        f"\n--- {phase_name}: mode={mode} "
        f"train_pred={train_pred} train_cls={train_classifier} train_bias={train_bias} "
        f"lambda_cls={lambda_cls} pred_lr_scale={pred_lr_scale} ---"
    )
    for epoch in range(1, int(args.epochs) + 1):
        train_total, train_pred_loss, train_cls_loss = train_or_eval_epoch(
            model=model,
            dataloader=dataloaders["train"],
            graph=graph,
            scaler=scaler,
            args=args,
            mode=mode,
            lambda_cls=lambda_cls,
            pos_weight=pos_weight,
            optimizer=optimizer,
            gate_temperature=gate_temperature,
        )
        with torch.no_grad():
            val_total, val_pred_loss, val_cls_loss = train_or_eval_epoch(
                model=model,
                dataloader=val_loader,
                graph=graph,
                scaler=scaler,
                args=args,
                mode=mode,
                lambda_cls=lambda_cls,
                pos_weight=pos_weight,
                optimizer=None,
                gate_temperature=gate_temperature,
            )

        print(
            f"{phase_name} epoch={epoch} "
            f"train_total={train_total:.5f} train_pred={train_pred_loss:.5f} train_cls={train_cls_loss:.5f} "
            f"val_total={val_total:.5f} val_pred={val_pred_loss:.5f} val_cls={val_cls_loss:.5f}"
        )

        selection_loss = val_pred_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            not_improved = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                best_path,
            )
        else:
            not_improved += 1

        if bool(args.early_stop) and not_improved == patience:
            print(f"{phase_name}: validation prediction loss did not improve for {patience} epochs; stopping.")
            break

    state = torch.load(best_path, map_location=torch.device(args.device))
    model.load_state_dict(state["model"])
    elapsed_min = (time.time() - start_time) / 60.0
    print(f"{phase_name}: best_val_pred_loss={best_loss:.5f} best_epoch={best_epoch} time_min={elapsed_min:.2f}")
    return {
        "phase": phase_name,
        "mode": mode,
        "best_val_loss": float(best_loss),
        "best_val_epoch": int(best_epoch),
        "checkpoint": str(best_path.resolve()),
        "lambda_cls": float(lambda_cls),
        "pred_lr_scale": float(pred_lr_scale),
    }


def mae_value(pred: torch.Tensor, true: torch.Tensor, mask_value: float = 5.0) -> float:
    mask = true > mask_value
    if not torch.any(mask):
        return float("nan")
    return float(torch.mean(torch.abs(true[mask] - pred[mask])).item())


def eee_value(pred: torch.Tensor, true: torch.Tensor, evs: torch.Tensor) -> float:
    mask = evs == 1
    if not torch.any(mask):
        return float("nan")
    return float(torch.mean(torch.abs(true[mask] - pred[mask])).item())


def collect_outputs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    graph: torch.Tensor,
    scaler: StandardScaler,
    gate_temperature: float,
) -> dict[str, torch.Tensor]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        "true": [],
        "evs": [],
        "probs": [],
        "base": [],
        "residual": [],
        "soft": [],
        "oracle": [],
        "always_on": [],
    }
    with torch.no_grad():
        for data, target, evs, _ in dataloader:
            parts = forward_parts(model, data, graph, gate_temperature)
            collected["true"].append(target)
            collected["evs"].append(evs)
            collected["probs"].append(parts["probs"])
            collected["base"].append(parts["base"])
            collected["residual"].append(parts["residual"])
            collected["soft"].append(parts["soft"])
            collected["oracle"].append(parts["base"] + (parts["bias"] * evs))
            collected["always_on"].append(parts["residual"])

    outputs = {
        "true": scaler.inverse_transform(torch.cat(collected["true"], dim=0)),
        "evs": torch.cat(collected["evs"], dim=0),
        "probs": torch.cat(collected["probs"], dim=0),
    }
    for name in ("base", "residual", "soft", "oracle", "always_on"):
        outputs[name] = scaler.inverse_transform(torch.cat(collected[name], dim=0))
    return outputs


def blank_result_row() -> dict[str, Any]:
    row: dict[str, Any] = {column: float("nan") for column in RESULT_COLUMNS}
    for column in ("variant", "flow", "checkpoint", "log_dir", "source", "phase_results"):
        row[column] = ""
    return row


def evaluate_trained_model(
    model: torch.nn.Module,
    dataloaders: dict[str, Any],
    graph: torch.Tensor,
    scaler: StandardScaler,
    seed: int,
    variant: str,
    phase_results: list[dict[str, Any]],
    checkpoint_dir: Path,
    gate_temperature: float,
) -> list[dict[str, Any]]:
    val_outputs = collect_outputs(model, dataloaders["val"], graph, scaler, gate_temperature)
    test_outputs = collect_outputs(model, dataloaders["test"], graph, scaler, gate_temperature)
    final_phase = phase_results[-1]
    rows: list[dict[str, Any]] = []

    for flow_index, flow_name in enumerate(FLOW_NAMES):
        row = blank_result_row()
        row["variant"] = variant
        row["seed"] = seed
        row["flow"] = flow_name
        row["best_val_loss"] = float(final_phase["best_val_loss"])
        row["best_val_epoch"] = int(final_phase["best_val_epoch"])
        row["checkpoint"] = str(final_phase["checkpoint"])
        row["log_dir"] = str(checkpoint_dir.resolve())
        row["source"] = "trained"
        row["phase_results"] = json.dumps(phase_results)

        true = test_outputs["true"][..., flow_index]
        evs = test_outputs["evs"][..., flow_index]
        row["event_prevalence"] = float(evs.detach().cpu().numpy().mean())

        for forecast_name in ("base", "residual", "soft", "oracle", "always_on"):
            pred = test_outputs[forecast_name][..., flow_index]
            row[f"{forecast_name}_mae"] = mae_value(pred, true)
            row[f"{forecast_name}_eee"] = eee_value(pred, true, evs)

        if variant == "ungated_3phase":
            row["primary_mae"] = row["residual_mae"]
            row["primary_eee"] = row["residual_eee"]
        else:
            row["primary_mae"] = row["soft_mae"]
            row["primary_eee"] = row["soft_eee"]

            val_probs = val_outputs["probs"][..., flow_index].detach().cpu().numpy().reshape(-1)
            val_labels = val_outputs["evs"][..., flow_index].detach().cpu().numpy().reshape(-1)
            test_probs = test_outputs["probs"][..., flow_index].detach().cpu().numpy().reshape(-1)
            test_labels = test_outputs["evs"][..., flow_index].detach().cpu().numpy().reshape(-1)
            val_best_f1, threshold = best_f1_threshold(val_probs, val_labels)
            tuned = f1_counts(test_probs, test_labels, threshold)
            fixed = f1_counts(test_probs, test_labels, 0.5)
            row["threshold"] = threshold
            row["val_best_f1"] = val_best_f1
            row["f1_tuned"] = tuned["f1"]
            row["f1_05"] = fixed["f1"]
            row["precision_tuned"] = tuned["precision"]
            row["recall_tuned"] = tuned["recall"]

        rows.append(row)
    return rows


def train_one_variant(
    configs: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    seed: int,
    variant: str,
    device: str,
    event_percentile: float,
    event_threshold_source: str,
    max_epochs: int | None,
    phase_patience: int,
    lambda_cls_phase2: float,
    lambda_cls_phase3: float,
    cls_pos_weight_mode: str,
    cls_pos_weight_cap: float | None,
    phase3_pred_lr_scale: float,
    gate_temperature: float,
) -> list[dict[str, Any]]:
    from model.models import STSSL

    init_seed(seed, device)
    arrays = load_arrays(data_dir, event_percentile, event_threshold_source)
    dataloaders, scaler = build_dataloaders(
        arrays,
        batch_size=int(configs["batch_size"]),
        test_batch_size=int(configs["test_batch_size"]),
        device=device,
    )
    args = build_run_args(configs, data_dir, seed, device, variant, max_epochs)
    graph = load_graph(Path(args.graph_file), device)
    args.num_nodes = len(graph)
    args.ipe = len(dataloaders["train"])

    model = STSSL(args).to(device)
    pred_params, classifier_params, bias_params = get_model_params_grouped(model)
    pos_weight = make_pos_weight(arrays, device, cls_pos_weight_mode, cls_pos_weight_cap)

    checkpoint_dir = output_dir / "checkpoints" / variant / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase_results: list[dict[str, Any]] = []

    phase_results.append(
        train_phase(
            model=model,
            dataloaders=dataloaders,
            graph=graph,
            scaler=scaler,
            args=args,
            pred_params=pred_params,
            classifier_params=classifier_params,
            bias_params=bias_params,
            phase_name="base_pretrain",
            mode="base",
            train_pred=True,
            train_classifier=False,
            train_bias=False,
            lambda_cls=0.0,
            pos_weight=pos_weight,
            patience=phase_patience,
            checkpoint_dir=checkpoint_dir,
            pred_lr_scale=1.0,
            gate_temperature=gate_temperature,
        )
    )

    if variant == "ungated_3phase":
        phase_results.append(
            train_phase(
                model=model,
                dataloaders=dataloaders,
                graph=graph,
                scaler=scaler,
                args=args,
                pred_params=pred_params,
                classifier_params=classifier_params,
                bias_params=bias_params,
                phase_name="residual_joint",
                mode="ungated",
                train_pred=True,
                train_classifier=False,
                train_bias=True,
                lambda_cls=0.0,
                pos_weight=pos_weight,
                patience=phase_patience,
                checkpoint_dir=checkpoint_dir,
                pred_lr_scale=1.0,
                gate_temperature=gate_temperature,
            )
        )
        phase_results.append(
            train_phase(
                model=model,
                dataloaders=dataloaders,
                graph=graph,
                scaler=scaler,
                args=args,
                pred_params=pred_params,
                classifier_params=classifier_params,
                bias_params=bias_params,
                phase_name="residual_bias_finetune",
                mode="ungated",
                train_pred=False,
                train_classifier=False,
                train_bias=True,
                lambda_cls=0.0,
                pos_weight=pos_weight,
                patience=phase_patience,
                checkpoint_dir=checkpoint_dir,
                pred_lr_scale=1.0,
                gate_temperature=gate_temperature,
            )
        )
    elif variant == "gated_3phase":
        phase_results.append(
            train_phase(
                model=model,
                dataloaders=dataloaders,
                graph=graph,
                scaler=scaler,
                args=args,
                pred_params=pred_params,
                classifier_params=classifier_params,
                bias_params=bias_params,
                phase_name="gated_oracle_warmup",
                mode="gated_oracle",
                train_pred=False,
                train_classifier=True,
                train_bias=True,
                lambda_cls=lambda_cls_phase2,
                pos_weight=pos_weight,
                patience=phase_patience,
                checkpoint_dir=checkpoint_dir,
                pred_lr_scale=1.0,
                gate_temperature=gate_temperature,
            )
        )
        phase_results.append(
            train_phase(
                model=model,
                dataloaders=dataloaders,
                graph=graph,
                scaler=scaler,
                args=args,
                pred_params=pred_params,
                classifier_params=classifier_params,
                bias_params=bias_params,
                phase_name="gated_soft_joint",
                mode="gated_soft",
                train_pred=True,
                train_classifier=True,
                train_bias=True,
                lambda_cls=lambda_cls_phase3,
                pos_weight=pos_weight,
                patience=phase_patience,
                checkpoint_dir=checkpoint_dir,
                pred_lr_scale=phase3_pred_lr_scale,
                gate_temperature=gate_temperature,
            )
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    rows = evaluate_trained_model(
        model=model,
        dataloaders=dataloaders,
        graph=graph,
        scaler=scaler,
        seed=seed,
        variant=variant,
        phase_results=phase_results,
        checkpoint_dir=checkpoint_dir,
        gate_temperature=gate_temperature,
    )

    del model, graph, dataloaders
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(rows[0].keys()) if rows else RESULT_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_results(root: Path, result_dir_name: str) -> Path | None:
    candidates = sorted(
        (root / result_dir_name).glob("*/results.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_optional_results_path(raw: str, root: Path, result_dir_name: str) -> Path | None:
    if raw.lower() == "none":
        return None
    if raw.lower() == "auto":
        return find_latest_results(root, result_dir_name)
    path = Path(raw).resolve()
    return path if path.is_file() else None


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def transform_original_rows(path: Path | None, seeds: set[int]) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for original in read_csv_dicts(path):
        if original.get("phase3_mode") != "original":
            continue
        seed = to_int(original.get("seed"))
        if seed not in seeds:
            continue
        row = blank_result_row()
        row["variant"] = PRIMARY_BASELINE_VARIANT
        row["seed"] = seed
        row["flow"] = original.get("flow", "")
        row["primary_mae"] = to_float(original.get("soft_mae"))
        row["primary_eee"] = to_float(original.get("soft_eee"))
        for name in ("base", "soft", "oracle", "always_on"):
            row[f"{name}_mae"] = to_float(original.get(f"{name}_mae"))
            row[f"{name}_eee"] = to_float(original.get(f"{name}_eee"))
        row["f1_tuned"] = to_float(original.get("test_f1_tuned"))
        row["f1_05"] = to_float(original.get("test_f1_05"))
        row["precision_tuned"] = to_float(original.get("test_precision_tuned"))
        row["recall_tuned"] = to_float(original.get("test_recall_tuned"))
        row["threshold"] = to_float(original.get("threshold"))
        row["val_best_f1"] = to_float(original.get("val_best_f1"))
        row["event_prevalence"] = to_float(original.get("test_prevalence"))
        row["best_val_loss"] = to_float(original.get("best_val_loss"))
        row["best_val_epoch"] = to_int(original.get("best_val_epoch"))
        row["checkpoint"] = original.get("checkpoint", "")
        row["log_dir"] = original.get("log_dir", "")
        row["source"] = str(path.resolve())
        rows.append(row)
    return rows


def transform_existing_ungated_rows(path: Path | None, seeds: set[int]) -> dict[int, list[dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for existing in read_csv_dicts(path):
        seed = to_int(existing.get("seed"))
        if seed not in seeds:
            continue
        row = blank_result_row()
        row["variant"] = "ungated_3phase"
        row["seed"] = seed
        row["flow"] = existing.get("flow", "")
        row["primary_mae"] = to_float(existing.get("residual_mae"))
        row["primary_eee"] = to_float(existing.get("residual_eee"))
        row["base_mae"] = to_float(existing.get("base_mae"))
        row["base_eee"] = to_float(existing.get("base_eee"))
        row["residual_mae"] = to_float(existing.get("residual_mae"))
        row["residual_eee"] = to_float(existing.get("residual_eee"))
        row["always_on_mae"] = row["residual_mae"]
        row["always_on_eee"] = row["residual_eee"]
        row["event_prevalence"] = to_float(existing.get("event_prevalence"))
        row["best_val_loss"] = to_float(existing.get("best_val_loss"))
        row["best_val_epoch"] = to_int(existing.get("best_val_epoch"))
        row["checkpoint"] = existing.get("checkpoint", "")
        row["log_dir"] = existing.get("log_dir", "")
        row["source"] = str(path.resolve())
        row["phase_results"] = existing.get("phase_results", "")
        grouped[seed].append(row)

    complete: dict[int, list[dict[str, Any]]] = {}
    expected_flows = set(FLOW_NAMES)
    for seed, rows in grouped.items():
        flows = {str(row["flow"]) for row in rows}
        if expected_flows.issubset(flows):
            complete[seed] = rows
    return complete


def resolve_seeds(raw: str, baseline_path: Path | None) -> list[int]:
    if raw.lower() != "auto":
        return parse_seed_list(raw)
    if baseline_path is not None and baseline_path.is_file():
        seeds = sorted(
            {
                to_int(row.get("seed"))
                for row in read_csv_dicts(baseline_path)
                if row.get("phase3_mode") == "original"
            }
        )
        seeds = [seed for seed in seeds if seed > 0]
        if seeds:
            return seeds
    return [1, 2, 3]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return float(np.mean(array)), std


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["flow"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (variant, flow), group in sorted(grouped.items()):
        item: dict[str, Any] = {"variant": variant, "flow": flow, "n": len(group)}
        for metric in SUMMARY_METRICS:
            mean, std = mean_std([to_float(row.get(metric)) for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)
    return summary


def build_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_key = {
        (to_int(row.get("seed")), str(row.get("flow"))): row
        for row in rows
        if row.get("variant") == PRIMARY_BASELINE_VARIANT
    }
    comparison: list[dict[str, Any]] = []
    for row in rows:
        variant = str(row.get("variant"))
        if variant == PRIMARY_BASELINE_VARIANT:
            continue
        key = (to_int(row.get("seed")), str(row.get("flow")))
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        comparison.append(
            {
                "variant": variant,
                "seed": row["seed"],
                "flow": row["flow"],
                "original_4phase_mae": to_float(baseline.get("primary_mae")),
                "variant_mae": to_float(row.get("primary_mae")),
                "delta_mae_variant_minus_original": to_float(row.get("primary_mae"))
                - to_float(baseline.get("primary_mae")),
                "original_4phase_eee": to_float(baseline.get("primary_eee")),
                "variant_eee": to_float(row.get("primary_eee")),
                "delta_eee_variant_minus_original": to_float(row.get("primary_eee"))
                - to_float(baseline.get("primary_eee")),
                "original_4phase_f1_tuned": to_float(baseline.get("f1_tuned")),
                "variant_f1_tuned": to_float(row.get("f1_tuned")),
                "delta_f1_variant_minus_original": to_float(row.get("f1_tuned"))
                - to_float(baseline.get("f1_tuned")),
            }
        )
    return comparison


def summarize_comparison(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison:
        grouped[(str(row["variant"]), str(row["flow"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (variant, flow), group in sorted(grouped.items()):
        item: dict[str, Any] = {"variant": variant, "flow": flow, "n": len(group)}
        for metric in (
            "delta_mae_variant_minus_original",
            "delta_eee_variant_minus_original",
            "delta_f1_variant_minus_original",
        ):
            mean, std = mean_std([to_float(row.get(metric)) for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)
    return summary


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary = summarize(rows)
    comparison = build_comparison(rows)
    comparison_summary = summarize_comparison(comparison)

    write_csv(output_dir / "results.csv", rows, RESULT_COLUMNS)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "comparison.csv", comparison)
    write_csv(output_dir / "comparison_summary.csv", comparison_summary)

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results": rows,
                "summary": summary,
                "comparison": comparison,
                "comparison_summary": comparison_summary,
            },
            handle,
            indent=2,
        )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    with (output_dir / "results.md").open("w", encoding="utf-8") as handle:
        handle.write("# NYCTaxi Three-Phase Residual Comparison\n\n")
        handle.write("## Primary Summary\n\n")
        handle.write(
            markdown_table(
                summary,
                [
                    "variant",
                    "flow",
                    "n",
                    "primary_mae_mean",
                    "primary_mae_std",
                    "primary_eee_mean",
                    "primary_eee_std",
                    "f1_tuned_mean",
                    "f1_05_mean",
                    "base_mae_mean",
                    "oracle_mae_mean",
                    "always_on_mae_mean",
                ],
            )
        )
        handle.write("\n\n## Delta vs Original 4-Phase\n\n")
        handle.write(
            markdown_table(
                comparison_summary,
                [
                    "variant",
                    "flow",
                    "n",
                    "delta_mae_variant_minus_original_mean",
                    "delta_mae_variant_minus_original_std",
                    "delta_eee_variant_minus_original_mean",
                    "delta_eee_variant_minus_original_std",
                    "delta_f1_variant_minus_original_mean",
                ],
            )
        )
        handle.write("\n\n## Per Seed\n\n")
        handle.write(
            markdown_table(
                rows,
                [
                    "variant",
                    "seed",
                    "flow",
                    "primary_mae",
                    "primary_eee",
                    "f1_tuned",
                    "f1_05",
                    "threshold",
                    "base_mae",
                    "residual_mae",
                    "soft_mae",
                    "oracle_mae",
                    "always_on_mae",
                    "source",
                ],
            )
        )
        handle.write("\n")


def preflight_model(
    configs: dict[str, Any],
    data_dir: Path,
    device: str,
    event_percentile: float,
    event_threshold_source: str,
) -> None:
    from model.models import STSSL

    arrays = load_arrays(data_dir, event_percentile, event_threshold_source)
    args = build_run_args(configs, data_dir, seed=1, device=device, variant="gated_3phase", max_epochs=1)
    graph = load_graph(Path(args.graph_file), device)
    args.num_nodes = len(graph)
    model = STSSL(args).to(device)
    pred_params, classifier_params, bias_params = get_model_params_grouped(model)
    pos_weight = make_pos_weight(arrays, device, mode="auto", cap=20.0)
    print("Preflight model constructed for three-phase residual comparison.")
    print(f"NYCTaxi train x shape: {arrays['x_train'].shape}")
    print(f"NYCTaxi train y shape: {arrays['y_train'].shape}")
    print(f"NYCTaxi event threshold source: {event_threshold_source}")
    print(f"NYCTaxi train event prevalence: {arrays['evs_train'].mean():.6f}")
    print(f"NYCTaxi val event prevalence: {arrays['evs_val'].mean():.6f}")
    print(f"NYCTaxi test event prevalence: {arrays['evs_test'].mean():.6f}")
    print(f"Classifier positive weights: {pos_weight.detach().cpu().numpy().reshape(-1).tolist()}")
    print(f"Parameter groups: pred={len(pred_params)}, cls={len(classifier_params)}, bias={len(bias_params)}")
    del model, graph
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "configs" / "NYCTaxi.yaml"))
    parser.add_argument("--data-dir", default=str(root / "preprocessed_data"))
    parser.add_argument("--output-dir", default=str(root / "nyctaxi_three_phase_residual_results"))
    parser.add_argument(
        "--seeds",
        default="auto",
        help="Comma-separated seeds, or 'auto' to reuse seeds from the latest original 4-phase run.",
    )
    parser.add_argument("--variants", default="ungated_3phase,gated_3phase")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--event-percentile", default=90.0, type=float)
    parser.add_argument(
        "--event-threshold-source",
        default="file",
        choices=EVENT_THRESHOLD_SOURCE_CHOICES,
        help=(
            "Event-label threshold source. 'file' uses saved evs_90 arrays; "
            "'train' regenerates paper-correct labels in memory; "
            "'split' reproduces the old notebook rule."
        ),
    )
    parser.add_argument("--max-epochs", default=None, type=int, help="Optional smoke-test epoch cap per phase.")
    parser.add_argument("--phase-patience", default=30, type=int)
    parser.add_argument("--lambda-cls-phase2", default=1.0, type=float)
    parser.add_argument("--lambda-cls-phase3", default=0.05, type=float)
    parser.add_argument("--cls-pos-weight-mode", default="auto", choices=["auto", "none"])
    parser.add_argument("--cls-pos-weight-cap", default=20.0, type=float)
    parser.add_argument("--phase3-pred-lr-scale", default=0.1, type=float)
    parser.add_argument("--gate-temperature", default=1.0, type=float)
    parser.add_argument(
        "--original-results",
        default="auto",
        help="Path to original drift experiment results.csv, 'auto', or 'none'.",
    )
    parser.add_argument(
        "--existing-ungated-results",
        default="auto",
        help="Path to previous ungated residual results.csv, 'auto', or 'none'.",
    )
    parser.add_argument("--no-reuse-existing-ungated", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    config_path = Path(args.config).resolve()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    original_path = resolve_optional_results_path(args.original_results, root, "nyctaxi_drift_results")
    seeds = resolve_seeds(args.seeds, original_path)
    seed_set = set(seeds)

    preflight_environment(data_dir, device)
    add_repo_to_path(root)
    configs = load_config(config_path)
    preflight_model(configs, data_dir, device, args.event_percentile, args.event_threshold_source)
    if args.preflight_only:
        print("Preflight complete. No training was run.")
        return 0

    variants = parse_csv_list(args.variants)
    invalid = [variant for variant in variants if variant not in VARIANT_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}. Choices: {VARIANT_CHOICES}")

    print(f"Resolved seeds: {seeds}")
    if original_path is not None:
        print(f"Original 4-phase comparison file: {original_path}")
    else:
        print("No original 4-phase comparison file found/provided.")

    existing_ungated_path = resolve_optional_results_path(
        args.existing_ungated_results,
        root,
        "nyctaxi_ungated_residual_results",
    )
    existing_ungated: dict[int, list[dict[str, Any]]] = {}
    if not args.no_reuse_existing_ungated and "ungated_3phase" in variants:
        existing_ungated = transform_existing_ungated_rows(existing_ungated_path, seed_set)
        if existing_ungated_path is not None:
            reused = sorted(existing_ungated.keys())
            print(f"Existing ungated comparison file: {existing_ungated_path}")
            print(f"Reusable complete ungated seeds: {reused}")
    elif "ungated_3phase" in variants:
        print("Existing ungated reuse disabled; ungated_3phase will be trained fresh.")

    output_dir = Path(args.output_dir).resolve() / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(transform_original_rows(original_path, seed_set))
    write_outputs(output_dir, rows, args)

    for variant in variants:
        for seed in seeds:
            if variant == "ungated_3phase" and seed in existing_ungated:
                print(f"\n=== Reusing ungated_3phase seed={seed} from existing results ===")
                rows.extend(existing_ungated[seed])
                write_outputs(output_dir, rows, args)
                print(
                    markdown_table(
                        existing_ungated[seed],
                        ["variant", "seed", "flow", "primary_mae", "primary_eee", "source"],
                    )
                )
                continue

            print(f"\n=== NYCTaxi variant={variant} seed={seed} device={device} ===")
            run_rows = train_one_variant(
                configs=configs,
                data_dir=data_dir,
                output_dir=output_dir,
                seed=seed,
                variant=variant,
                device=device,
                event_percentile=args.event_percentile,
                event_threshold_source=args.event_threshold_source,
                max_epochs=args.max_epochs,
                phase_patience=int(args.phase_patience),
                lambda_cls_phase2=float(args.lambda_cls_phase2),
                lambda_cls_phase3=float(args.lambda_cls_phase3),
                cls_pos_weight_mode=str(args.cls_pos_weight_mode),
                cls_pos_weight_cap=float(args.cls_pos_weight_cap) if args.cls_pos_weight_cap is not None else None,
                phase3_pred_lr_scale=float(args.phase3_pred_lr_scale),
                gate_temperature=float(args.gate_temperature),
            )
            rows.extend(run_rows)
            write_outputs(output_dir, rows, args)
            print(
                markdown_table(
                    run_rows,
                    [
                        "variant",
                        "seed",
                        "flow",
                        "primary_mae",
                        "primary_eee",
                        "f1_tuned",
                        "f1_05",
                        "base_mae",
                        "residual_mae",
                        "soft_mae",
                        "oracle_mae",
                    ],
                )
            )

    write_outputs(output_dir, rows, args)
    print(f"\nSaved results to: {output_dir}")
    print((output_dir / "results.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import copy
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STLAFORMER_ROOT = REPO_ROOT / "external" / "STLAformer"
if str(STLAFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(STLAFORMER_ROOT))
STLAFORMER_MODEL_ROOT = STLAFORMER_ROOT / "model"
if str(STLAFORMER_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(STLAFORMER_MODEL_ROOT))

from lib.data_prepare import get_dataloaders_from_index_data  # type: ignore  # noqa: E402
from lib.metrics import MAE, MAPE, RMSE, RMSE_MAE_MAPE  # type: ignore  # noqa: E402
from model.STLformer import STLformer  # type: ignore  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_log(message: str, log_file=None) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {message}"
    print(line)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    overall_rmse, overall_mae, overall_mape = RMSE_MAE_MAPE(y_true, y_pred)
    per_horizon = []
    for horizon in range(y_pred.shape[1]):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, horizon, :], y_pred[:, horizon, :])
        per_horizon.append(
            {
                "horizon": horizon + 1,
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
            }
        )
    return {
        "overall": {
            "rmse": float(overall_rmse),
            "mae": float(overall_mae),
            "mape": float(overall_mape),
        },
        "per_horizon": per_horizon,
    }


def metric_delta(improved: dict, baseline: dict) -> dict:
    overall = {
        key: float(improved["overall"][key] - baseline["overall"][key])
        for key in ("rmse", "mae", "mape")
    }
    per_horizon = []
    for improved_h, baseline_h in zip(improved["per_horizon"], baseline["per_horizon"]):
        per_horizon.append(
            {
                "horizon": improved_h["horizon"],
                "rmse": float(improved_h["rmse"] - baseline_h["rmse"]),
                "mae": float(improved_h["mae"] - baseline_h["mae"]),
                "mape": float(improved_h["mape"] - baseline_h["mape"]),
            }
        )
    return {"overall": overall, "per_horizon": per_horizon}


def to_numpy_predictions(model, loader, scaler, device, correction_head=None, thresholds=None):
    model.eval()
    if correction_head is not None:
        correction_head.eval()
    y_all = []
    base_all = []
    corrected_all = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            base_norm, repr_seq = model(x_batch, return_repr=True)
            base = scaler.inverse_transform(base_norm)
            y_all.append(y_batch.cpu().numpy())
            base_all.append(base.cpu().numpy())
            if correction_head is not None:
                corrected = correction_head.correct(base, repr_seq, thresholds)
                corrected_all.append(corrected.cpu().numpy())
    y_true = np.vstack(y_all).squeeze(-1)
    base_pred = np.vstack(base_all).squeeze(-1)
    corrected_pred = None
    if corrected_all:
        corrected_pred = np.vstack(corrected_all).squeeze(-1)
    return y_true, base_pred, corrected_pred


class SignedCorrectionHead(nn.Module):
    def __init__(self, in_steps: int, model_dim: int, out_steps: int, dropout: float):
        super().__init__()
        flat_dim = in_steps * model_dim
        hidden_dim = flat_dim
        self.out_steps = out_steps
        self.up_gate = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps),
        )
        self.down_gate = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps),
        )
        self.up_mag = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps),
        )
        self.down_mag = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_steps),
        )

    def forward(self, repr_seq):
        batch_size, in_steps, num_nodes, model_dim = repr_seq.shape
        flat = repr_seq.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, in_steps * model_dim)
        up_logit = self.up_gate(flat).transpose(1, 2).unsqueeze(-1)
        down_logit = self.down_gate(flat).transpose(1, 2).unsqueeze(-1)
        up_mag = F.softplus(self.up_mag(flat)).transpose(1, 2).unsqueeze(-1)
        down_mag = F.softplus(self.down_mag(flat)).transpose(1, 2).unsqueeze(-1)
        return {
            "up_logit": up_logit,
            "down_logit": down_logit,
            "up_prob": torch.sigmoid(up_logit),
            "down_prob": torch.sigmoid(down_logit),
            "up_mag": up_mag,
            "down_mag": down_mag,
        }

    def correct(self, base_pred, repr_seq, thresholds):
        outputs = self.forward(repr_seq)
        corrected = (
            base_pred
            + outputs["up_prob"] * (thresholds["u_up"] + outputs["up_mag"])
            - outputs["down_prob"] * (thresholds["u_down"] + outputs["down_mag"])
        )
        return torch.clamp_min(corrected, 0.0)


def compute_signed_thresholds(model, loader, scaler, device, quantile):
    pos_residuals = []
    neg_residuals = []
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            base_norm = model(x_batch)
            base = scaler.inverse_transform(base_norm)
            pos_residuals.append(torch.clamp(y_batch - base, min=0.0).cpu())
            neg_residuals.append(torch.clamp(base - y_batch, min=0.0).cpu())
    pos_tensor = torch.cat(pos_residuals, dim=0)
    neg_tensor = torch.cat(neg_residuals, dim=0)
    u_up = torch.quantile(pos_tensor, quantile, dim=0, keepdim=True)
    u_down = torch.quantile(neg_tensor, quantile, dim=0, keepdim=True)
    pos_indicator = (pos_tensor > u_up).float()
    neg_indicator = (neg_tensor > u_down).float()
    pos_rate = float(pos_indicator.mean().item())
    neg_rate = float(neg_indicator.mean().item())
    thresholds = {
        "u_up": u_up.to(device),
        "u_down": u_down.to(device),
        "pos_rate": pos_rate,
        "neg_rate": neg_rate,
        "pos_weight_up": float((1.0 - pos_rate) / max(pos_rate, 1e-6)),
        "pos_weight_down": float((1.0 - neg_rate) / max(neg_rate, 1e-6)),
    }
    return thresholds


def evaluate_correction(model, correction_head, loader, scaler, device, thresholds):
    y_true, base_pred, corrected_pred = to_numpy_predictions(
        model, loader, scaler, device, correction_head=correction_head, thresholds=thresholds
    )
    base_summary = summarize_metrics(y_true, base_pred)
    corrected_summary = summarize_metrics(y_true, corrected_pred)
    return {
        "base": base_summary,
        "corrected": corrected_summary,
        "delta": metric_delta(corrected_summary, base_summary),
    }


def train_correction(
    model,
    correction_head,
    train_loader,
    val_loader,
    scaler,
    device,
    thresholds,
    epochs,
    patience,
    lr,
    weight_decay,
    lambda_cls,
    lambda_mag,
    lambda_mae,
    pos_weight_scale,
    log_file,
):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        correction_head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    pos_weight_up = torch.tensor(
        [thresholds["pos_weight_up"] * pos_weight_scale], device=device, dtype=torch.float32
    )
    pos_weight_down = torch.tensor(
        [thresholds["pos_weight_down"] * pos_weight_scale], device=device, dtype=torch.float32
    )
    bce_up = nn.BCEWithLogitsLoss(pos_weight=pos_weight_up)
    bce_down = nn.BCEWithLogitsLoss(pos_weight=pos_weight_down)
    best_state = copy.deepcopy(correction_head.state_dict())
    best_val = math.inf
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        correction_head.train()
        train_totals = {"loss": 0.0, "mae": 0.0, "cls": 0.0, "mag": 0.0}
        batch_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                base_norm, repr_seq = model(x_batch, return_repr=True)
                base = scaler.inverse_transform(base_norm)
            outputs = correction_head(repr_seq)
            pos_residual = torch.clamp(y_batch - base, min=0.0)
            neg_residual = torch.clamp(base - y_batch, min=0.0)
            up_target = (pos_residual > thresholds["u_up"]).float()
            down_target = (neg_residual > thresholds["u_down"]).float()
            up_excess = torch.clamp(pos_residual - thresholds["u_up"], min=0.0)
            down_excess = torch.clamp(neg_residual - thresholds["u_down"], min=0.0)
            corrected = correction_head.correct(base, repr_seq, thresholds)
            mae_loss = torch.mean(torch.abs(corrected - y_batch))
            cls_loss = bce_up(outputs["up_logit"], up_target) + bce_down(outputs["down_logit"], down_target)
            up_mag_loss = F.l1_loss(outputs["up_mag"] * up_target, up_excess * up_target)
            down_mag_loss = F.l1_loss(outputs["down_mag"] * down_target, down_excess * down_target)
            mag_loss = up_mag_loss + down_mag_loss
            loss = lambda_mae * mae_loss + lambda_cls * cls_loss + lambda_mag * mag_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(correction_head.parameters(), 5.0)
            optimizer.step()

            train_totals["loss"] += float(loss.item())
            train_totals["mae"] += float(mae_loss.item())
            train_totals["cls"] += float(cls_loss.item())
            train_totals["mag"] += float(mag_loss.item())
            batch_count += 1

        val_report = evaluate_correction(model, correction_head, val_loader, scaler, device, thresholds)
        val_mae = val_report["corrected"]["overall"]["mae"]
        history.append(
            {
                "epoch": epoch,
                "train": {key: value / max(batch_count, 1) for key, value in train_totals.items()},
                "val_corrected_mae": val_mae,
                "val_delta_mae": val_report["delta"]["overall"]["mae"],
            }
        )
        print_log(
            "Correction e{:03d} | train loss={:.4f} mae={:.4f} cls={:.4f} mag={:.4f} | "
            "val corrected MAE={:.4f} delta MAE={:.4f}".format(
                epoch,
                train_totals["loss"] / max(batch_count, 1),
                train_totals["mae"] / max(batch_count, 1),
                train_totals["cls"] / max(batch_count, 1),
                train_totals["mag"] / max(batch_count, 1),
                val_mae,
                val_report["delta"]["overall"]["mae"],
            ),
            log_file=log_file,
        )
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(correction_head.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print_log(
                    f"Early stopping correction at epoch {epoch} after {patience} stale epochs.",
                    log_file=log_file,
                )
                break

    correction_head.load_state_dict(best_state)
    return correction_head, {"best_val_mae": best_val, "best_epoch": best_epoch, "history": history}


def load_official_config(dataset):
    cfg_path = STLAFORMER_ROOT / "model" / "STLformer.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        full_cfg = yaml.safe_load(handle)
    return full_cfg[dataset]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS04")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "external" / "STLAformer-data" / "data"))
    parser.add_argument("--baseline-ckpt", default=str(STLAFORMER_ROOT / "pre-trained" / "PEMS04.pt"))
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "study_runs" / "STLAformerStudy"),
    )
    parser.add_argument("--mode", choices=["baseline_eval", "train_correction", "full"], default="full")
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cls", type=float, default=0.05)
    parser.add_argument("--lambda-mag", type=float, default=0.10)
    parser.add_argument("--lambda-mae", type=float, default=1.0)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--tag", default="official_pretrained_signed_correction")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset.upper()
    cfg = load_official_config(dataset)
    data_path = Path(args.data_dir) / dataset

    run_root = Path(args.output_root) / dataset
    run_dir = run_root / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        print_log(f"Dataset={dataset} device={device} seed={args.seed}", log_file=log_file)
        print_log(f"Data path={data_path}", log_file=log_file)
        print_log(f"Baseline checkpoint={args.baseline_ckpt}", log_file=log_file)
        print_log(
            f"Correction config: q={args.quantile} epochs={args.epochs} patience={args.patience} "
            f"lr={args.lr} lambda_mae={args.lambda_mae} lambda_cls={args.lambda_cls} lambda_mag={args.lambda_mag}",
            log_file=log_file,
        )

        train_loader, val_loader, test_loader, scaler = get_dataloaders_from_index_data(
            str(data_path), batch_size=cfg.get("batch_size"), log=None
        )
        model = STLformer(**cfg["model_args"]).to(device)
        state_dict = torch.load(args.baseline_ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print_log("Loaded official baseline checkpoint.", log_file=log_file)

        y_true_val, base_val, _ = to_numpy_predictions(model, val_loader, scaler, device)
        y_true_test, base_test, _ = to_numpy_predictions(model, test_loader, scaler, device)
        baseline = {
            "val": summarize_metrics(y_true_val, base_val),
            "test": summarize_metrics(y_true_test, base_test),
        }
        print_log(
            "Baseline test | RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.4f}".format(
                **baseline["test"]["overall"]
            ),
            log_file=log_file,
        )

        results = {
            "dataset": dataset,
            "seed": args.seed,
            "device": str(device),
            "baseline_checkpoint": args.baseline_ckpt,
            "baseline": baseline,
        }

        if args.mode in {"train_correction", "full"}:
            thresholds = compute_signed_thresholds(model, train_loader, scaler, device, args.quantile)
            print_log(
                "Signed thresholds | up_rate={:.6f} down_rate={:.6f} up_pos_weight={:.4f} down_pos_weight={:.4f}".format(
                    thresholds["pos_rate"],
                    thresholds["neg_rate"],
                    thresholds["pos_weight_up"],
                    thresholds["pos_weight_down"],
                ),
                log_file=log_file,
            )
            correction_head = SignedCorrectionHead(
                in_steps=cfg["model_args"]["in_steps"],
                model_dim=model.model_dim,
                out_steps=cfg["model_args"]["out_steps"],
                dropout=cfg["model_args"]["dropout"],
            ).to(device)
            correction_head, train_info = train_correction(
                model=model,
                correction_head=correction_head,
                train_loader=train_loader,
                val_loader=val_loader,
                scaler=scaler,
                device=device,
                thresholds=thresholds,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lambda_cls=args.lambda_cls,
                lambda_mag=args.lambda_mag,
                lambda_mae=args.lambda_mae,
                pos_weight_scale=args.pos_weight_scale,
                log_file=log_file,
            )
            correction_report = {
                "val": evaluate_correction(model, correction_head, val_loader, scaler, device, thresholds),
                "test": evaluate_correction(model, correction_head, test_loader, scaler, device, thresholds),
            }
            print_log(
                "Corrected test | RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.4f} | delta MAE={delta_mae:.4f}".format(
                    rmse=correction_report["test"]["corrected"]["overall"]["rmse"],
                    mae=correction_report["test"]["corrected"]["overall"]["mae"],
                    mape=correction_report["test"]["corrected"]["overall"]["mape"],
                    delta_mae=correction_report["test"]["delta"]["overall"]["mae"],
                ),
                log_file=log_file,
            )
            correction_ckpt = run_dir / "signed_correction_head.pt"
            torch.save(correction_head.state_dict(), correction_ckpt)
            results["thresholds"] = {
                "quantile": args.quantile,
                "up_rate": thresholds["pos_rate"],
                "down_rate": thresholds["neg_rate"],
                "up_pos_weight": thresholds["pos_weight_up"],
                "down_pos_weight": thresholds["pos_weight_down"],
            }
            results["correction_training"] = train_info
            results["correction"] = correction_report
            results["correction_checkpoint"] = str(correction_ckpt)

        results_path = run_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print_log(f"Saved results to {results_path}", log_file=log_file)


if __name__ == "__main__":
    main()

"""
Generate plots highlighting Hyb-STEX vs original ST-SSL on extreme events.

Default target is NYCTaxi because the Hyb-STEX checkpoint exists in saved_weights.
Run from repo root:

  python scripts/plot_hybstex_vs_stssl_extremes.py

Outputs:
  plots/hybstex_vs_stssl_NYCTaxi_<timestamp>/
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


FLOW_NAMES = ("inflow", "outflow")
MODEL_LABELS = {
    "stssl": "ST-SSL",
    "hyb_base": "Hyb-STEX base",
    "hyb": "Hyb-STEX",
}
COLORS = {
    "true": "#111827",
    "stssl": "#ef7d22",
    "hyb_base": "#64748b",
    "hyb": "#1d9a6c",
    "event": "#dc2626",
    "win": "#16a34a",
    "over": "#f97316",
    "miss": "#2563eb",
}


class StandardScaler:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_torch(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std + self.mean


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32)


def make_evs(y: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    evs = np.zeros_like(y, dtype=np.float32)
    for horizon_idx in range(y.shape[1]):
        for node_idx in range(y.shape[2]):
            for flow_idx in range(y.shape[3]):
                series = y[:, horizon_idx, node_idx, flow_idx]
                threshold = np.percentile(series, percentile)
                evs[:, horizon_idx, node_idx, flow_idx] = (series > threshold).astype(np.float32)
    return evs


def load_split_arrays(data_dir: Path, dataset: str, percentile: float) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        npz = np.load(data_dir / dataset / f"{split}.npz")
        arrays[f"x_{split}"] = npz["x"].astype(np.float32)
        arrays[f"y_{split}"] = npz["y"].astype(np.float32)
        arrays[f"evs_{split}"] = npz["evs_90"].astype(np.float32) if "evs_90" in npz.files else make_evs(arrays[f"y_{split}"], percentile)
    return arrays


def load_scaler(arrays: dict[str, np.ndarray]) -> StandardScaler:
    train_val = np.concatenate([arrays["x_train"], arrays["x_val"]], axis=0)
    return StandardScaler(mean=float(train_val.mean()), std=float(train_val.std()))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def add_hyb_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "S_Loss": 0,
        "T_Loss": 0,
        "seed": 1,
        "comment": "plot",
        "cheb_order": 3,
        "graph_init": "8_neighbours",
        "self_attention_flag": True,
        "cross_attention_flag": False,
        "feedforward_flag": False,
        "layer_norm_flag": False,
        "additional_sa_flag": False,
        "learnable_flag": False,
        "pos_emb_flag": False,
        "rank": 0,
        "add_8": False,
        "add_eye": False,
        "add_x_encoder": False,
        "freeze_encoder": False,
        "threshold_adj_mx": False,
        "affinity_conv": False,
        "loss": "mae",
        "load_path": None,
        "variant": None,
    }
    out = dict(cfg)
    out.update({k: out.get(k, v) for k, v in defaults.items()})
    return out


def predict_hybstex(args: argparse.Namespace) -> None:
    root = repo_root()
    torch.from_numpy = lambda arr: torch.tensor(arr)

    sys.path.insert(0, str(root))
    from model.models import STSSL

    data_dir = Path(args.data_dir).resolve()
    arrays = load_split_arrays(data_dir, args.dataset, args.ev_percentile)
    scaler = load_scaler(arrays)

    cfg = add_hyb_defaults(load_config(root / "configs" / f"{args.dataset}.yaml"))
    cfg["device"] = args.device
    cfg["data_dir"] = str(data_dir)
    cfg["dataset"] = args.dataset
    cfg["graph_file"] = str(data_dir / args.dataset / "adj_mx.npz")
    cfg["num_nodes"] = int(np.load(cfg["graph_file"])["adj_mx"].shape[0])

    graph = torch.tensor(np.load(cfg["graph_file"])["adj_mx"], dtype=torch.float32, device=args.device)
    model = STSSL(argparse.Namespace(**cfg)).to(args.device)
    state = torch.load(args.hyb_checkpoint, map_location=torch.device(args.device))["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    x_all = np.concatenate([arrays["x_val"], arrays["x_test"]], axis=0)
    x_norm = scaler.transform(x_all).astype(np.float32)
    loader = torch.utils.data.DataLoader(
        torch.as_tensor(x_norm, dtype=torch.float32, device=args.device),
        batch_size=args.batch_size,
        shuffle=False,
    )

    pred_base = []
    pred_hyb = []
    with torch.no_grad():
        for x in loader:
            z, z_cls = model(x, graph)
            pred_base.append(model.predict(z, z_cls, "pred"))
            pred_hyb.append(model.predict(z, z_cls, "bias"))

    base = scaler.inverse_torch(torch.cat(pred_base, dim=0))
    hyb = scaler.inverse_torch(torch.cat(pred_hyb, dim=0))
    np.savez_compressed(args.output_npz, hyb_base=torch_to_numpy(base), hyb=torch_to_numpy(hyb))


def predict_stssl(args: argparse.Namespace) -> None:
    stssl_dir = Path(args.stssl_dir).resolve()
    sys.path.insert(0, str(stssl_dir))

    from model.models import STSSL

    data_dir = Path(args.data_dir).resolve()
    arrays = load_split_arrays(data_dir, args.dataset, args.ev_percentile)
    scaler = load_scaler(arrays)

    cfg = load_config(stssl_dir / "configs" / f"{args.dataset}.yaml")
    cfg["device"] = args.device
    cfg["data_dir"] = str(data_dir)
    cfg["dataset"] = args.dataset
    cfg["graph_file"] = str(data_dir / args.dataset / "adj_mx.npz")
    cfg["num_nodes"] = int(np.load(cfg["graph_file"])["adj_mx"].shape[0])

    graph = torch.tensor(np.load(cfg["graph_file"])["adj_mx"], dtype=torch.float32, device=args.device)
    model = STSSL(argparse.Namespace(**cfg)).to(args.device)
    state = torch.load(args.stssl_checkpoint, map_location=torch.device(args.device))["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    x_all = np.concatenate([arrays["x_val"], arrays["x_test"]], axis=0)
    x_norm = scaler.transform(x_all).astype(np.float32)
    loader = torch.utils.data.DataLoader(
        torch.as_tensor(x_norm, dtype=torch.float32, device=args.device),
        batch_size=args.batch_size,
        shuffle=False,
    )

    pred = []
    with torch.no_grad():
        for x in loader:
            # Only repr1 is used by ST-SSL's predictor. Calling encoder directly avoids
            # stochastic augmentation during plotting.
            z = model.encoder(x, graph)
            pred.append(model.predict(z, None))
    pred = scaler.inverse_torch(torch.cat(pred, dim=0))
    np.savez_compressed(args.output_npz, stssl=torch_to_numpy(pred))


def write_shared_truth(args: argparse.Namespace, cache_dir: Path) -> None:
    data_dir = Path(args.data_dir).resolve()
    arrays = load_split_arrays(data_dir, args.dataset, args.ev_percentile)
    y = np.concatenate([arrays["y_val"], arrays["y_test"]], axis=0)
    evs = np.concatenate([arrays["evs_val"], arrays["evs_test"]], axis=0)
    np.savez_compressed(
        cache_dir / "truth.npz",
        true=y.astype(np.float32),
        evs=evs.astype(np.float32),
        val_len=np.asarray([arrays["y_val"].shape[0]], dtype=np.int64),
    )


def latest_best_stssl_checkpoint(results_csv: Path, dataset: str) -> Path:
    rows = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] == dataset:
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No {dataset} rows found in {results_csv}")

    by_seed: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    best_seed = min(
        by_seed,
        key=lambda seed: float(np.mean([float(row["eee"]) for row in by_seed[seed]])),
    )
    return Path(by_seed[best_seed][0]["checkpoint"])


def eee_by_node(true: np.ndarray, pred: np.ndarray, evs: np.ndarray) -> np.ndarray:
    err = np.abs(pred - true)
    out = np.full((true.shape[1], true.shape[2]), np.nan, dtype=np.float64)
    for node in range(true.shape[1]):
        for flow in range(true.shape[2]):
            mask = evs[:, node, flow] == 1
            if mask.any():
                out[node, flow] = float(err[mask, node, flow].mean())
    return out


def mae_by_node(true: np.ndarray, pred: np.ndarray, mask_value: float = 5.0) -> np.ndarray:
    err = np.abs(pred - true)
    out = np.full((true.shape[1], true.shape[2]), np.nan, dtype=np.float64)
    for node in range(true.shape[1]):
        for flow in range(true.shape[2]):
            mask = true[:, node, flow] > mask_value
            if mask.any():
                out[node, flow] = float(err[mask, node, flow].mean())
    return out


def event_value_stats(true: np.ndarray, evs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event_mean = np.full((true.shape[1], true.shape[2]), np.nan, dtype=np.float64)
    event_max = np.full((true.shape[1], true.shape[2]), np.nan, dtype=np.float64)
    for node in range(true.shape[1]):
        for flow in range(true.shape[2]):
            mask = evs[:, node, flow] == 1
            if mask.any():
                values = true[mask, node, flow]
                event_mean[node, flow] = float(values.mean())
                event_max[node, flow] = float(values.max())
    return event_mean, event_max


def kde1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.zeros_like(grid)
    std = float(values.std(ddof=1))
    bw = 1.06 * std * (len(values) ** (-1 / 5)) if std > 0 else max(float(values.mean()) * 0.05, 1.0)
    bw = max(bw, 1e-3)
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z * z).mean(axis=1) / (bw * math.sqrt(2 * math.pi))


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_node_timeseries(out: Path, t: np.ndarray, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, evs: np.ndarray, node: int, flow: int, val_len: int, title_extra: str) -> None:
    flow_name = FLOW_NAMES[flow]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, true[:, node, flow], color=COLORS["true"], lw=1.8, label="True")
    ax.plot(t, stssl[:, node, flow], color=COLORS["stssl"], lw=1.2, alpha=0.9, label="ST-SSL")
    ax.plot(t, base[:, node, flow], color=COLORS["hyb_base"], lw=1.0, alpha=0.75, ls="--", label="Hyb base")
    ax.plot(t, hyb[:, node, flow], color=COLORS["hyb"], lw=1.5, alpha=0.95, label="Hyb-STEX")
    mask = evs[:, node, flow] == 1
    ax.scatter(t[mask], true[mask, node, flow], s=24, facecolors="none", edgecolors=COLORS["event"], lw=1.4, label="90th pct events")
    ax.axvline(val_len - 0.5, color="#94a3b8", lw=1.0, ls=":")
    ax.set_title(f"Node {node} {flow_name}: Hyb-STEX vs ST-SSL on extreme events {title_extra}")
    ax.set_xlabel("Validation + test timestep")
    ax.set_ylabel(flow_name)
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.2)
    savefig(out / f"node_{node:03d}_{flow_name}_timeseries.png")


def plot_node_zoom(out: Path, t: np.ndarray, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, evs: np.ndarray, node: int, flow: int, radius: int = 36) -> None:
    flow_name = FLOW_NAMES[flow]
    event_mask = evs[:, node, flow] == 1
    if event_mask.any():
        improvement = np.abs(stssl[:, node, flow] - true[:, node, flow]) - np.abs(hyb[:, node, flow] - true[:, node, flow])
        idxs = np.where(event_mask)[0]
        center = int(idxs[np.nanargmax(improvement[idxs])])
    else:
        center = int(np.argmax(true[:, node, flow]))
    s = max(0, center - radius)
    e = min(len(t), center + radius + 1)
    xs = t[s:e]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(xs, true[s:e, node, flow], color=COLORS["true"], lw=2.0, label="True")
    ax.plot(xs, stssl[s:e, node, flow], color=COLORS["stssl"], lw=1.5, label="ST-SSL")
    ax.plot(xs, base[s:e, node, flow], color=COLORS["hyb_base"], lw=1.1, ls="--", label="Hyb base")
    ax.plot(xs, hyb[s:e, node, flow], color=COLORS["hyb"], lw=1.8, label="Hyb-STEX")
    mask = evs[s:e, node, flow] == 1
    ax.scatter(xs[mask], true[s:e][mask, node, flow], s=35, facecolors="none", edgecolors=COLORS["event"], lw=1.5)
    ax.set_title(f"Node {node} {flow_name}: zoom around strongest Hyb-STEX extreme-event gain")
    ax.set_xlabel("Validation + test timestep")
    ax.set_ylabel(flow_name)
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.25)
    savefig(out / f"node_{node:03d}_{flow_name}_zoom.png")


def plot_node_tail(out: Path, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, evs: np.ndarray, node: int, flow: int) -> None:
    flow_name = FLOW_NAMES[flow]
    series = {
        "True": true[:, node, flow],
        "ST-SSL": stssl[:, node, flow],
        "Hyb base": base[:, node, flow],
        "Hyb-STEX": hyb[:, node, flow],
    }
    lo = min(float(v.min()) for v in series.values())
    hi = max(float(v.max()) for v in series.values())
    grid = np.linspace(lo, hi, 220)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(series["True"], bins=36, density=True, color="#cbd5e1", alpha=0.45, label="True hist")
    ax.plot(grid, kde1d(series["True"], grid), color=COLORS["true"], lw=2.0, label="True KDE")
    ax.plot(grid, kde1d(series["ST-SSL"], grid), color=COLORS["stssl"], lw=1.8, label="ST-SSL KDE")
    ax.plot(grid, kde1d(series["Hyb base"], grid), color=COLORS["hyb_base"], lw=1.4, ls="--", label="Hyb base KDE")
    ax.plot(grid, kde1d(series["Hyb-STEX"], grid), color=COLORS["hyb"], lw=1.8, label="Hyb-STEX KDE")
    threshold = np.percentile(true[:, node, flow], 90)
    ax.axvline(threshold, color=COLORS["event"], lw=1.2, ls=":", label="true 90th pct")
    ax.set_title(f"Node {node} {flow_name}: distribution tail capture")
    ax.set_xlabel(flow_name)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    savefig(out / f"node_{node:03d}_{flow_name}_tail_kde.png")


def plot_node_scatter(out: Path, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, evs: np.ndarray, node: int, flow: int) -> None:
    flow_name = FLOW_NAMES[flow]
    mask = evs[:, node, flow] == 1
    if mask.sum() < 2:
        return
    y = true[mask, node, flow]
    fig, ax = plt.subplots(figsize=(5.8, 5.5))
    ax.scatter(y, stssl[mask, node, flow], s=28, alpha=0.7, color=COLORS["stssl"], label="ST-SSL")
    ax.scatter(y, base[mask, node, flow], s=22, alpha=0.55, color=COLORS["hyb_base"], label="Hyb base")
    ax.scatter(y, hyb[mask, node, flow], s=28, alpha=0.75, color=COLORS["hyb"], label="Hyb-STEX")
    lo = min(float(y.min()), float(stssl[mask, node, flow].min()), float(hyb[mask, node, flow].min()))
    hi = max(float(y.max()), float(stssl[mask, node, flow].max()), float(hyb[mask, node, flow].max()))
    ax.plot([lo, hi], [lo, hi], color="#334155", lw=1.0, ls=":")
    ax.set_title(f"Node {node} {flow_name}: event-only parity")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    savefig(out / f"node_{node:03d}_{flow_name}_event_scatter.png")


def plot_heatmaps(out: Path, values: dict[str, np.ndarray], row: int, col: int, flow: int) -> None:
    flow_name = FLOW_NAMES[flow]
    items = [
        ("ST-SSL EEE", values["stssl_eee"][:, flow]),
        ("Hyb-STEX EEE", values["hyb_eee"][:, flow]),
        ("EEE improvement", values["improvement"][:, flow]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, vals) in zip(axes, items):
        grid = vals.reshape(row, col)
        cmap = "RdYlGn" if "improvement" in title else "magma"
        im = ax.imshow(grid, aspect="auto", cmap=cmap)
        ax.set_title(f"{flow_name} {title}")
        ax.set_xlabel("grid col")
        ax.set_ylabel("grid row")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(out / f"nyctaxi_{flow_name}_eee_heatmaps.png")


def plot_city_series(out: Path, t: np.ndarray, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, flow: int, val_len: int) -> None:
    flow_name = FLOW_NAMES[flow]
    y = true[:, :, flow].sum(axis=1)
    s = stssl[:, :, flow].sum(axis=1)
    b = base[:, :, flow].sum(axis=1)
    h = hyb[:, :, flow].sum(axis=1)
    threshold = np.percentile(y, 90)
    event = y > threshold
    st_err = np.abs(s - y)
    hyb_err = np.abs(h - y)
    win = event & (hyb_err < st_err)
    over = event & (hyb_err >= st_err) & (h > y)
    miss = event & (hyb_err >= st_err) & (h <= y)

    fig, ax = plt.subplots(figsize=(15, 5.0))
    ax.plot(t, y, color=COLORS["true"], lw=1.8, label="True city total")
    ax.plot(t, s, color=COLORS["stssl"], lw=1.1, alpha=0.9, label="ST-SSL")
    ax.plot(t, b, color=COLORS["hyb_base"], lw=1.0, ls="--", alpha=0.7, label="Hyb base")
    ax.plot(t, h, color=COLORS["hyb"], lw=1.4, label="Hyb-STEX")
    ax.scatter(t[win], y[win], s=28, color=COLORS["win"], label="Hyb wins on event")
    ax.scatter(t[over], y[over], s=32, color=COLORS["over"], marker="x", label="over-correction")
    ax.scatter(t[miss], y[miss], s=32, color=COLORS["miss"], marker="^", label="missed/lagged")
    ax.axhline(threshold, color=COLORS["event"], lw=1.0, ls=":", label="city 90th pct")
    ax.axvline(val_len - 0.5, color="#94a3b8", lw=1.0, ls=":")
    ax.set_title(f"NYCTaxi city-wide total {flow_name}: validation + test")
    ax.set_xlabel("Validation + test timestep")
    ax.set_ylabel(f"total {flow_name}")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    savefig(out / f"city_total_{flow_name}_timeseries.png")

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(t, st_err, color=COLORS["stssl"], lw=1.0, label="|ST-SSL - true|")
    ax.plot(t, hyb_err, color=COLORS["hyb"], lw=1.0, label="|Hyb-STEX - true|")
    ax.fill_between(t, st_err, hyb_err, where=st_err > hyb_err, color=COLORS["win"], alpha=0.18, label="Hyb error reduction")
    ax.fill_between(t, st_err, hyb_err, where=st_err <= hyb_err, color=COLORS["over"], alpha=0.12, label="Hyb worse/equal")
    ax.axvline(val_len - 0.5, color="#94a3b8", lw=1.0, ls=":")
    ax.set_title(f"NYCTaxi city-wide total {flow_name}: absolute error comparison")
    ax.set_xlabel("Validation + test timestep")
    ax.set_ylabel("absolute error")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    savefig(out / f"city_total_{flow_name}_absolute_error.png")


def plot_city_zoom(out: Path, t: np.ndarray, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, flow: int, radius: int = 48) -> None:
    flow_name = FLOW_NAMES[flow]
    y = true[:, :, flow].sum(axis=1)
    s = stssl[:, :, flow].sum(axis=1)
    b = base[:, :, flow].sum(axis=1)
    h = hyb[:, :, flow].sum(axis=1)
    event = y > np.percentile(y, 90)
    gain = np.abs(s - y) - np.abs(h - y)
    idxs = np.where(event)[0]
    center = int(idxs[np.argmax(gain[idxs])]) if len(idxs) else int(np.argmax(y))
    lo, hi = max(0, center - radius), min(len(t), center + radius + 1)
    xs = t[lo:hi]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(xs, y[lo:hi], color=COLORS["true"], lw=2.0, label="True city total")
    ax.plot(xs, s[lo:hi], color=COLORS["stssl"], lw=1.4, label="ST-SSL")
    ax.plot(xs, b[lo:hi], color=COLORS["hyb_base"], lw=1.0, ls="--", label="Hyb base")
    ax.plot(xs, h[lo:hi], color=COLORS["hyb"], lw=1.7, label="Hyb-STEX")
    ax.scatter([center], [y[center]], s=60, facecolors="none", edgecolors=COLORS["event"], lw=1.8)
    ax.set_title(f"NYCTaxi city-wide {flow_name}: zoom around strongest Hyb-STEX event gain")
    ax.set_xlabel("Validation + test timestep")
    ax.set_ylabel(f"total {flow_name}")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    savefig(out / f"city_total_{flow_name}_best_gain_zoom.png")


def plot_city_event_error_summary(out: Path, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, flow: int) -> None:
    flow_name = FLOW_NAMES[flow]
    y = true[:, :, flow].sum(axis=1)
    s = stssl[:, :, flow].sum(axis=1)
    b = base[:, :, flow].sum(axis=1)
    h = hyb[:, :, flow].sum(axis=1)
    event = y > np.percentile(y, 90)
    st_err = np.abs(s - y)
    hyb_err = np.abs(h - y)
    base_err = np.abs(b - y)
    gain = st_err - hyb_err

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(gain[event], bins=30, color=COLORS["hyb"], alpha=0.75)
    ax.axvline(0, color="#334155", lw=1.0, ls=":")
    ax.set_title(f"NYCTaxi city-wide {flow_name}: event error reduction distribution")
    ax.set_xlabel("ST-SSL absolute error - Hyb-STEX absolute error")
    ax.set_ylabel("event timesteps")
    ax.grid(alpha=0.25)
    savefig(out / f"city_total_{flow_name}_event_error_reduction_hist.png")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = np.where(gain[event] >= 0, COLORS["win"], COLORS["over"])
    ax.scatter(y[event], gain[event], s=28, color=colors, alpha=0.78)
    ax.axhline(0, color="#334155", lw=1.0, ls=":")
    ax.set_title(f"NYCTaxi city-wide {flow_name}: event magnitude vs Hyb-STEX error reduction")
    ax.set_xlabel(f"true total {flow_name} on event timesteps")
    ax.set_ylabel("ST-SSL absolute error - Hyb-STEX absolute error")
    ax.grid(alpha=0.25)
    savefig(out / f"city_total_{flow_name}_event_gain_scatter.png")

    labels = ["ST-SSL", "Hyb base", "Hyb-STEX"]
    vals = [float(st_err[event].mean()), float(base_err[event].mean()), float(hyb_err[event].mean())]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(labels, vals, color=[COLORS["stssl"], COLORS["hyb_base"], COLORS["hyb"]])
    ax.set_title(f"NYCTaxi city-wide {flow_name}: mean absolute error on city events")
    ax.set_ylabel("mean absolute error")
    ax.grid(axis="y", alpha=0.25)
    savefig(out / f"city_total_{flow_name}_event_mae_bars.png")


def plot_city_ranked_window(
    out: Path,
    t: np.ndarray,
    true: np.ndarray,
    stssl: np.ndarray,
    base: np.ndarray,
    hyb: np.ndarray,
    flow: int,
    center: int,
    tag: str,
    rank: int,
    radius: int = 36,
) -> None:
    flow_name = FLOW_NAMES[flow]
    y = true[:, :, flow].sum(axis=1)
    s = stssl[:, :, flow].sum(axis=1)
    b = base[:, :, flow].sum(axis=1)
    h = hyb[:, :, flow].sum(axis=1)
    lo, hi = max(0, center - radius), min(len(t), center + radius + 1)
    xs = t[lo:hi]
    gain = abs(s[center] - y[center]) - abs(h[center] - y[center])

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(xs, y[lo:hi], color=COLORS["true"], lw=2.0, label="True city total")
    ax.plot(xs, s[lo:hi], color=COLORS["stssl"], lw=1.4, label="ST-SSL")
    ax.plot(xs, b[lo:hi], color=COLORS["hyb_base"], lw=1.0, ls="--", label="Hyb base")
    ax.plot(xs, h[lo:hi], color=COLORS["hyb"], lw=1.7, label="Hyb-STEX")
    ax.scatter([center], [y[center]], s=60, facecolors="none", edgecolors=COLORS["event"], lw=1.8)
    ax.axvline(center, color="#94a3b8", lw=1.0, ls=":")
    ax.set_title(f"NYCTaxi city-wide {flow_name}: {tag} event zoom #{rank} (error reduction {gain:.1f})")
    ax.set_xlabel("Validation + test timestep")
    ax.set_ylabel(f"total {flow_name}")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    clean_tag = tag.lower().replace("/", "_").replace(" ", "_")
    savefig(out / f"city_total_{flow_name}_{clean_tag}_zoom_{rank:02d}_t{center:04d}.png")


def plot_city_ranked_windows(out: Path, t: np.ndarray, true: np.ndarray, stssl: np.ndarray, base: np.ndarray, hyb: np.ndarray, flow: int, per_class: int) -> None:
    flow_name = FLOW_NAMES[flow]
    y = true[:, :, flow].sum(axis=1)
    s = stssl[:, :, flow].sum(axis=1)
    h = hyb[:, :, flow].sum(axis=1)
    event = y > np.percentile(y, 90)
    gain = np.abs(s - y) - np.abs(h - y)
    event_idxs = np.where(event)[0]

    groups = [
        ("Hyb win", event_idxs[gain[event_idxs] > 0], True),
        ("over-correction", event_idxs[(gain[event_idxs] <= 0) & (h[event_idxs] > y[event_idxs])], False),
        ("missed/lagged", event_idxs[(gain[event_idxs] <= 0) & (h[event_idxs] <= y[event_idxs])], False),
    ]
    for tag, idxs, descending in groups:
        if len(idxs) == 0:
            continue
        order = np.argsort(gain[idxs])
        if descending:
            order = order[::-1]
        for rank, center in enumerate(idxs[order][:per_class], start=1):
            plot_city_ranked_window(out / f"{flow_name}_ranked_event_windows", t, true, stssl, base, hyb, flow, int(center), tag, rank)


def plot_top_bars(out: Path, ranking: np.ndarray, flow: int, top_k: int = 25) -> None:
    flow_name = FLOW_NAMES[flow]
    top = ranking[:top_k]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    labels = [str(int(row["node"])) for row in top]
    vals = [float(row["eee_improvement"]) for row in top]
    ax.bar(labels, vals, color=COLORS["hyb"])
    ax.axhline(0, color="#334155", lw=0.8)
    ax.set_title(f"Top node-level EEE reductions: {flow_name}")
    ax.set_xlabel("node")
    ax.set_ylabel("ST-SSL EEE - Hyb-STEX EEE")
    ax.grid(axis="y", alpha=0.25)
    savefig(out / f"top_node_eee_reductions_{flow_name}.png")


def write_node_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "node",
            "flow",
            "stssl_eee",
            "hyb_eee",
            "hyb_base_eee",
            "eee_improvement",
            "base_to_hyb_improvement",
            "stssl_mae",
            "hyb_mae",
            "event_count",
            "event_true_mean",
            "event_true_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_plots(args: argparse.Namespace) -> Path:
    root = repo_root()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = mkdir(Path(args.output_dir).resolve() / f"hybstex_vs_stssl_{args.dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    cache_dir = mkdir(out_dir / "_prediction_cache")

    stssl_checkpoint = Path(args.stssl_checkpoint).resolve() if args.stssl_checkpoint else latest_best_stssl_checkpoint(Path(args.stssl_results_csv).resolve(), args.dataset).resolve()
    hyb_checkpoint = Path(args.hyb_checkpoint).resolve()
    data_dir = Path(args.data_dir).resolve()

    write_shared_truth(args, cache_dir)
    hyb_npz = cache_dir / "hybstex_preds.npz"
    stssl_npz = cache_dir / "stssl_preds.npz"

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
    ]
    if not hyb_npz.exists() or args.regenerate_predictions:
        subprocess.run(
            common
            + [
                "predict-hyb",
                "--dataset",
                args.dataset,
                "--data-dir",
                str(data_dir),
                "--device",
                device,
                "--batch-size",
                str(args.batch_size),
                "--ev-percentile",
                str(args.ev_percentile),
                "--hyb-checkpoint",
                str(hyb_checkpoint),
                "--output-npz",
                str(hyb_npz),
            ],
            check=True,
            cwd=str(root),
        )
    if not stssl_npz.exists() or args.regenerate_predictions:
        subprocess.run(
            common
            + [
                "predict-stssl",
                "--dataset",
                args.dataset,
                "--data-dir",
                str(data_dir),
                "--stssl-dir",
                str(Path(args.stssl_dir).resolve()),
                "--device",
                device,
                "--batch-size",
                str(args.batch_size),
                "--ev-percentile",
                str(args.ev_percentile),
                "--stssl-checkpoint",
                str(stssl_checkpoint),
                "--output-npz",
                str(stssl_npz),
            ],
            check=True,
            cwd=str(root),
        )

    truth_npz = np.load(cache_dir / "truth.npz")
    hyb_npz_loaded = np.load(hyb_npz)
    stssl_npz_loaded = np.load(stssl_npz)

    true = truth_npz["true"][:, 0, :, :]
    evs = truth_npz["evs"][:, 0, :, :]
    val_len = int(truth_npz["val_len"][0])
    hyb = hyb_npz_loaded["hyb"][:, 0, :, :]
    base = hyb_npz_loaded["hyb_base"][:, 0, :, :]
    stssl = stssl_npz_loaded["stssl"][:, 0, :, :]
    t = np.arange(true.shape[0])

    stssl_eee = eee_by_node(true, stssl, evs)
    hyb_eee = eee_by_node(true, hyb, evs)
    base_eee = eee_by_node(true, base, evs)
    stssl_mae = mae_by_node(true, stssl)
    hyb_mae = mae_by_node(true, hyb)
    improvement = stssl_eee - hyb_eee
    base_to_hyb = base_eee - hyb_eee
    event_counts = evs.sum(axis=0)
    event_mean, event_max = event_value_stats(true, evs)

    ranking_rows: list[dict[str, Any]] = []
    for node in range(true.shape[1]):
        for flow in range(true.shape[2]):
            ranking_rows.append(
                {
                    "node": node,
                    "flow": FLOW_NAMES[flow],
                    "stssl_eee": stssl_eee[node, flow],
                    "hyb_eee": hyb_eee[node, flow],
                    "hyb_base_eee": base_eee[node, flow],
                    "eee_improvement": improvement[node, flow],
                    "base_to_hyb_improvement": base_to_hyb[node, flow],
                    "stssl_mae": stssl_mae[node, flow],
                    "hyb_mae": hyb_mae[node, flow],
                    "event_count": int(event_counts[node, flow]),
                    "event_true_mean": event_mean[node, flow],
                    "event_true_max": event_max[node, flow],
                }
            )
    ranking_rows.sort(key=lambda row: (np.nan_to_num(row["eee_improvement"], nan=-1e9)), reverse=True)
    write_node_ranking(out_dir / "node_eee_ranking.csv", ranking_rows)

    meta = {
        "dataset": args.dataset,
        "device": device,
        "hyb_checkpoint": str(hyb_checkpoint),
        "stssl_checkpoint": str(stssl_checkpoint),
        "data_dir": str(data_dir),
        "val_len": val_len,
        "n_steps": int(true.shape[0]),
        "min_events_for_node_plots": args.min_events,
        "min_event_max_for_node_plots": args.min_event_max,
        "city_zooms_per_class": args.city_zooms_per_class,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    heat_dir = mkdir(out_dir / "heatmaps")
    city_dir = mkdir(out_dir / "city_wide")
    summary_dir = mkdir(out_dir / "summary")
    values = {"stssl_eee": stssl_eee, "hyb_eee": hyb_eee, "improvement": improvement}
    row_count, col_count = args.grid_rows, args.grid_cols
    for flow in range(2):
        plot_heatmaps(heat_dir, values, row_count, col_count, flow)
        plot_city_series(city_dir, t, true, stssl, base, hyb, flow, val_len)
        plot_city_zoom(city_dir, t, true, stssl, base, hyb, flow)
        plot_city_event_error_summary(city_dir, true, stssl, base, hyb, flow)
        plot_city_ranked_windows(city_dir, t, true, stssl, base, hyb, flow, args.city_zooms_per_class)

        flow_rows = [
            row
            for row in ranking_rows
            if row["flow"] == FLOW_NAMES[flow]
            and np.isfinite(row["eee_improvement"])
            and row["event_count"] >= args.min_events
            and np.isfinite(row["event_true_max"])
            and row["event_true_max"] >= args.min_event_max
        ]
        flow_rows.sort(key=lambda row: row["eee_improvement"], reverse=True)
        dtype = [("node", int), ("eee_improvement", float)]
        ranking_array = np.asarray([(row["node"], row["eee_improvement"]) for row in flow_rows], dtype=dtype)
        plot_top_bars(summary_dir, ranking_array, flow, top_k=min(args.top_nodes, 30))

        selected = flow_rows[: args.top_nodes]
        for row in selected:
            node = int(row["node"])
            title_extra = f"(EEE gain {row['eee_improvement']:.2f}, events {row['event_count']}, max {row['event_true_max']:.0f})"
            node_root = mkdir(out_dir / "node_wise" / FLOW_NAMES[flow] / f"node_{node:03d}")
            plot_node_timeseries(node_root, t, true, stssl, base, hyb, evs, node, flow, val_len, title_extra)
            plot_node_zoom(node_root, t, true, stssl, base, hyb, evs, node, flow)
            plot_node_tail(node_root, true, stssl, base, hyb, evs, node, flow)
            plot_node_scatter(node_root, true, stssl, base, hyb, evs, node, flow)

    if not args.keep_prediction_cache:
        shutil.rmtree(cache_dir)

    print(f"Generated plots in: {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="plot", choices=["plot", "predict-hyb", "predict-stssl"])
    parser.add_argument("--dataset", default="NYCTaxi")
    parser.add_argument("--data-dir", default=str(root / "external" / "ST-SSL" / "data"))
    parser.add_argument("--stssl-dir", default=str(root / "external" / "ST-SSL"))
    parser.add_argument("--stssl-results-csv", default=str(root / "stssl_eee_results" / "20260627-081429" / "results.csv"))
    parser.add_argument("--stssl-checkpoint", default=None)
    parser.add_argument("--hyb-checkpoint", default=str(root / "saved_weights" / "NYCTaxi-p4.pth"))
    parser.add_argument("--output-dir", default=str(root / "plots"))
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ev-percentile", type=float, default=90.0)
    parser.add_argument("--top-nodes", type=int, default=35)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--min-event-max", type=float, default=20.0)
    parser.add_argument("--city-zooms-per-class", type=int, default=6)
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--regenerate-predictions", action="store_true")
    parser.add_argument("--keep-prediction-cache", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "predict-hyb":
        predict_hybstex(args)
        return 0
    if args.mode == "predict-stssl":
        predict_stssl(args)
        return 0
    generate_plots(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

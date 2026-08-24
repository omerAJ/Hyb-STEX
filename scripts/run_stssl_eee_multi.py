"""
Train original ST-SSL baselines and report Hyb-STEX EEE.

Default plan:
  - NYCBike2: seeds 1, 2, 3
  - NYCTaxi:  seeds 1, 2, 3
  - BJTaxi:   seeds 1, 2

Pass ``--datasets NYCBike1 --nycbike1-seeds 1,2,3`` to run the
NYCBike1 baseline only.

The script expects ST-SSL-format data at:
  preprocessed_data/<DATASET>/{train,val,test,adj_mx}.npz

The data files should contain evs_90 labels generated from training-split
thresholds. If labels are missing, this script regenerates them in memory from
training thresholds rather than using split-specific thresholds.

Run from the repo root:
  python scripts/run_stssl_eee_multi.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


DEFAULT_SEEDS = {
    "NYCBike1": [1, 2, 3],
    "NYCBike2": [1, 2, 3],
    "NYCTaxi": [1, 2, 3],
    "BJTaxi": [1, 2],
}

REQUIRED_DATA_FILES = ("train.npz", "val.npz", "test.npz", "adj_mx.npz")


class StandardScaler:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data * self.std) + self.mean


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_seed_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def ensure_stssl_repo(stssl_dir: Path) -> None:
    if (stssl_dir / "main.py").is_file() and (stssl_dir / "model" / "models.py").is_file():
        return

    stssl_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Echo-Ji/ST-SSL.git", str(stssl_dir)],
        check=True,
    )


def missing_data_files(data_dir: Path, datasets: list[str]) -> list[Path]:
    missing: list[Path] = []
    for dataset in datasets:
        for name in REQUIRED_DATA_FILES:
            path = data_dir / dataset / name
            if not path.is_file():
                missing.append(path)
    return missing


def import_hybstex_metrics(root: Path) -> Any:
    metrics_path = root / "lib" / "metrics.py"
    spec = importlib.util.spec_from_file_location("hybstex_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load metrics module from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_dwa(l_old: Any, l_new: Any, temp: float = 2.0) -> list[float]:
    l_old_t = torch.tensor(l_old, dtype=torch.float32)
    l_new_t = torch.tensor(l_new, dtype=torch.float32)
    n_tasks = len(l_new_t)
    ratio = l_old_t / l_new_t
    weights = n_tasks * torch.softmax(ratio / temp, dim=0)
    return [float(value) for value in weights.detach().cpu().tolist()]


def _choice_prob(tensor: torch.Tensor, invert: bool = False) -> np.ndarray:
    probs_t = torch.softmax(tensor, dim=0)
    if invert:
        probs_t = 1.0 - probs_t
    probs = np.asarray(probs_t.detach().cpu().tolist(), dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones_like(probs, dtype=np.float64)
        total = probs.sum()
    return probs / total


def safe_aug_topology(sim_mx: torch.Tensor, input_graph: torch.Tensor, percent: float = 0.2) -> torch.Tensor:
    sim_mx = sim_mx.to(input_graph.device)
    index_list = input_graph.nonzero()
    edge_num = int(index_list.shape[0] / 2)
    add_drop_num = int(edge_num * (percent / 2.0) / 2.0)
    aug_graph = copy.deepcopy(input_graph)
    if edge_num <= 0 or add_drop_num <= 0:
        return aug_graph

    edge_mask = (input_graph > 0).tril(diagonal=-1)
    drop_scores = sim_mx[edge_mask]
    if drop_scores.numel() > 0:
        drop_prob = _choice_prob(drop_scores, invert=True)
        drop_list_np = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
        drop_list = torch.as_tensor(drop_list_np, device=index_list.device, dtype=torch.long)
        drop_index = index_list[drop_list]
        zeros = torch.zeros_like(aug_graph[0, 0])
        aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
        aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing="ij")
    lower_mask = y < x
    x, y = x[lower_mask], y[lower_mask]

    add_mask = torch.ones(sim_mx.size(), dtype=torch.bool, device=sim_mx.device).tril(diagonal=-1)
    add_scores = sim_mx[add_mask]
    if add_scores.numel() > 0:
        add_prob = _choice_prob(add_scores)
        add_list = np.random.choice(int((node_num * node_num - node_num) / 2), size=add_drop_num, p=add_prob)
        add_x = torch.as_tensor(x[add_list], device=aug_graph.device, dtype=torch.long)
        add_y = torch.as_tensor(y[add_list], device=aug_graph.device, dtype=torch.long)
        ones = torch.ones_like(aug_graph[0, 0])
        aug_graph[add_x, add_y] = ones
        aug_graph[add_y, add_x] = ones

    return aug_graph


def safe_aug_traffic(t_sim_mx: torch.Tensor, flow_data: torch.Tensor, percent: float = 0.2) -> torch.Tensor:
    l_steps, batch_size, node_count = t_sim_mx.shape
    mask_num = int(batch_size * l_steps * node_count * percent)
    aug_flow = copy.deepcopy(flow_data)
    if mask_num <= 0:
        return aug_flow

    mask_prob = np.asarray((1.0 - t_sim_mx.permute(1, 0, 2).reshape(-1)).detach().cpu().tolist(), dtype=np.float64)
    mask_prob = mask_prob / mask_prob.sum()

    x, y, z = np.meshgrid(range(batch_size), range(l_steps), range(node_count), indexing="ij")
    mask_list = np.random.choice(batch_size * l_steps * node_count, size=mask_num, p=mask_prob)

    mask_x = torch.as_tensor(x.reshape(-1)[mask_list], device=aug_flow.device, dtype=torch.long)
    mask_y = torch.as_tensor(y.reshape(-1)[mask_list], device=aug_flow.device, dtype=torch.long)
    mask_z = torch.as_tensor(z.reshape(-1)[mask_list], device=aug_flow.device, dtype=torch.long)
    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[mask_x, mask_y, mask_z] = zeros
    return aug_flow


def install_runtime_patches(stssl_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(stssl_dir))

    utils = importlib.import_module("lib.utils")
    utils.dwa = safe_dwa

    aug = importlib.import_module("model.aug")
    aug.aug_topology = safe_aug_topology
    aug.aug_traffic = safe_aug_traffic

    return {
        "STSSL": importlib.import_module("model.models").STSSL,
        "Trainer": importlib.import_module("model.trainer").Trainer,
        "load_graph": utils.load_graph,
    }


def init_seed(seed: int, device: str) -> None:
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)


def load_arrays(data_dir: Path, dataset: str, ev_percentile: float, force_regenerate_evs: bool) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    file_evs: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        npz = np.load(data_dir / dataset / f"{split}.npz")
        arrays[f"x_{split}"] = npz["x"].astype(np.float32)
        arrays[f"y_{split}"] = npz["y"].astype(np.float32)
        if "evs_90" in npz.files:
            file_evs[split] = npz["evs_90"].astype(np.float32)

    use_file_evs = not force_regenerate_evs and all(split in file_evs for split in ("train", "val", "test"))
    if use_file_evs:
        for split in ("train", "val", "test"):
            arrays[f"evs_{split}"] = file_evs[split]
    else:
        thresholds = np.percentile(arrays["y_train"], ev_percentile, axis=0)
        for split in ("train", "val", "test"):
            arrays[f"evs_{split}"] = (arrays[f"y_{split}"] > thresholds).astype(np.float32)
    return arrays


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: str,
    shuffle: bool,
    drop_last: bool,
) -> torch.utils.data.DataLoader:
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def build_dataloaders(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    test_batch_size: int,
    device: str,
) -> tuple[dict[str, Any], StandardScaler]:
    train_val_x = np.concatenate([arrays["x_train"], arrays["x_val"]], axis=0)
    scaler = StandardScaler(mean=float(train_val_x.mean()), std=float(train_val_x.std()))

    normalized = {}
    for split in ("train", "val", "test"):
        normalized[f"x_{split}"] = scaler.transform(arrays[f"x_{split}"]).astype(np.float32)
        normalized[f"y_{split}"] = scaler.transform(arrays[f"y_{split}"]).astype(np.float32)

    dataloaders = {
        "train": make_loader(normalized["x_train"], normalized["y_train"], batch_size, device, True, True),
        "val": make_loader(normalized["x_val"], normalized["y_val"], test_batch_size, device, False, True),
        "test": make_loader(normalized["x_test"], normalized["y_test"], test_batch_size, device, False, False),
        "scaler": scaler,
    }
    return dataloaders, scaler


def evaluate_eee(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    graph: torch.Tensor,
    evs_test: np.ndarray,
    metrics: Any,
    device: str,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            repr1, repr2 = model(data, graph)
            y_pred.append(model.predict(repr1, repr2))
            y_true.append(target)

    pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
    true = scaler.inverse_transform(torch.cat(y_true, dim=0))
    evs = torch.as_tensor(evs_test, dtype=torch.float32, device=device)

    results: dict[str, dict[str, float | int]] = {}
    for idx, flow in enumerate(("inflow", "outflow")):
        mae, eee = metrics.test_metrics(pred[..., idx], true[..., idx], evs=evs[..., idx])
        results[flow] = {
            "mae": float(mae),
            "eee": float(eee),
            "extreme_count": int(evs[..., idx].sum().item()),
        }
    return results


def load_config(stssl_dir: Path, dataset: str) -> dict[str, Any]:
    config_path = stssl_dir / "configs" / f"{dataset}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing ST-SSL config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def run_one(
    modules: dict[str, Any],
    metrics: Any,
    stssl_dir: Path,
    data_dir: Path,
    dataset: str,
    seed: int,
    device: str,
    ev_percentile: float,
    force_regenerate_evs: bool,
    max_epochs: int | None,
) -> list[dict[str, Any]]:
    config = load_config(stssl_dir, dataset)
    arrays = load_arrays(data_dir, dataset, ev_percentile, force_regenerate_evs)

    config["seed"] = seed
    config["mode"] = "train"
    config["device"] = device
    config["data_dir"] = str(data_dir)
    config["dataset"] = dataset
    config["graph_file"] = str(data_dir / dataset / "adj_mx.npz")
    config["best_path"] = None
    config["debug"] = False
    if max_epochs is not None:
        config["epochs"] = int(max_epochs)

    actual_input_length = int(arrays["x_train"].shape[1])
    if int(config.get("input_length", actual_input_length)) != actual_input_length:
        print(
            f"[warn] {dataset} seed {seed}: config input_length={config.get('input_length')} "
            f"but data has {actual_input_length}; using data length."
        )
        config["input_length"] = actual_input_length

    init_seed(seed, device)
    graph = modules["load_graph"](config["graph_file"], device=device)
    config["num_nodes"] = len(graph)
    args = argparse.Namespace(**config)

    dataloaders, scaler = build_dataloaders(
        arrays,
        batch_size=int(config["batch_size"]),
        test_batch_size=int(config["test_batch_size"]),
        device=device,
    )

    model = modules["STSSL"](args).to(device)
    if hasattr(model, "thm") and not hasattr(model.thm, "lbl"):
        lbl_rl = torch.ones(int(config["batch_size"]), int(config["num_nodes"]), device=device)
        lbl_fk = torch.zeros(int(config["batch_size"]), int(config["num_nodes"]), device=device)
        model.thm.lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
    optimizer = torch.optim.Adam(
        params=list(model.parameters()),
        lr=float(config["lr_init"]),
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False,
    )
    trainer = modules["Trainer"](model=model, optimizer=optimizer, dataloader=dataloaders, graph=graph, args=args)
    train_results = trainer.train()

    best_path = Path(trainer.best_path)
    state = torch.load(best_path, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    eee_results = evaluate_eee(model, dataloaders["test"], scaler, graph, arrays["evs_test"], metrics, device)

    rows = []
    for flow, values in eee_results.items():
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "flow": flow,
            "mae": values["mae"],
            "eee": values["eee"],
            "extreme_count": values["extreme_count"],
            "best_val_loss": float(train_results["best_val_loss"]),
            "best_val_epoch": int(train_results["best_val_epoch"]),
            "checkpoint": str(best_path),
            "log_dir": str(Path(trainer.args.log_dir)),
        })
    return rows


def format_float(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["dataset", "seed", "flow", "mae", "eee", "extreme_count", "best_val_epoch", "best_val_loss"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["seed"]),
                    str(row["flow"]),
                    format_float(float(row["mae"])),
                    format_float(float(row["eee"])),
                    str(row["extreme_count"]),
                    str(row["best_val_epoch"]),
                    format_float(float(row["best_val_loss"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["flow"])].append(row)

    summary = []
    for (dataset, flow), group in sorted(grouped.items()):
        maes = np.asarray([float(row["mae"]) for row in group], dtype=np.float64)
        eees = np.asarray([float(row["eee"]) for row in group], dtype=np.float64)
        summary.append({
            "dataset": dataset,
            "flow": flow,
            "n": len(group),
            "mae_mean": float(maes.mean()),
            "mae_std": float(maes.std(ddof=1)) if len(group) > 1 else 0.0,
            "eee_mean": float(eees.mean()),
            "eee_std": float(eees.std(ddof=1)) if len(group) > 1 else 0.0,
        })
    return summary


def summary_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["dataset", "flow", "n", "mae_mean", "mae_std", "eee_mean", "eee_std"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["flow"]),
                    str(row["n"]),
                    format_float(float(row["mae_mean"])),
                    format_float(float(row["mae_std"])),
                    format_float(float(row["eee_mean"])),
                    format_float(float(row["eee_std"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stssl-dir", default=str(root / "external" / "ST-SSL"))
    parser.add_argument("--data-dir", default=str(root / "preprocessed_data"))
    parser.add_argument("--output-dir", default=str(root / "stssl_eee_results"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DEFAULT_SEEDS),
        default=["NYCBike2", "NYCTaxi", "BJTaxi"],
        help="Datasets to run. Defaults to the original NYCBike2, NYCTaxi, and BJTaxi plan.",
    )
    parser.add_argument("--nycbike1-seeds", default="1,2,3")
    parser.add_argument("--nycbike2-seeds", default="1,2,3")
    parser.add_argument("--nyctaxi-seeds", default="1,2,3")
    parser.add_argument("--bjtaxi-seeds", default="1,2")
    parser.add_argument("--ev-percentile", default=90.0, type=float)
    parser.add_argument("--force-regenerate-evs", action="store_true")
    parser.add_argument("--max-epochs", default=None, type=int, help="Optional smoke-test override.")
    args = parser.parse_args()

    stssl_dir = Path(args.stssl_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    ensure_stssl_repo(stssl_dir)

    seed_values = {
        "NYCBike1": args.nycbike1_seeds,
        "NYCBike2": args.nycbike2_seeds,
        "NYCTaxi": args.nyctaxi_seeds,
        "BJTaxi": args.bjtaxi_seeds,
    }
    seed_plan = {
        dataset: parse_seed_list(seed_values[dataset], DEFAULT_SEEDS[dataset])
        for dataset in args.datasets
    }
    datasets = list(seed_plan.keys())
    missing = missing_data_files(data_dir, datasets)
    if missing:
        print("Missing required ST-SSL dataset files:")
        for path in missing:
            print(f"  - {path}")
        print("\nPut the ST-SSL dataset folders under the data dir, or pass --data-dir.")
        print("Expected layout: <data-dir>/<DATASET>/{train,val,test,adj_mx}.npz")
        return 2

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")

    metrics = import_hybstex_metrics(root)
    modules = install_runtime_patches(stssl_dir)

    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for dataset, seeds in seed_plan.items():
        for seed in seeds:
            print(f"\n=== {dataset} seed={seed} device={device} ===")
            rows = run_one(
                modules=modules,
                metrics=metrics,
                stssl_dir=stssl_dir,
                data_dir=data_dir,
                dataset=dataset,
                seed=seed,
                device=device,
                ev_percentile=float(args.ev_percentile),
                force_regenerate_evs=bool(args.force_regenerate_evs),
                max_epochs=args.max_epochs,
            )
            all_rows.extend(rows)
            write_csv(run_dir / "results_partial.csv", all_rows)
            print(markdown_table(rows))

    summary_rows = summarize(all_rows)
    write_csv(run_dir / "results.csv", all_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    with (run_dir / "results.md").open("w", encoding="utf-8") as f:
        f.write("# ST-SSL EEE Results\n\n")
        f.write("## Per Seed\n\n")
        f.write(markdown_table(all_rows))
        f.write("\n\n## Summary\n\n")
        f.write(summary_markdown(summary_rows))
        f.write("\n")
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nPer-seed results:")
    print(markdown_table(all_rows))
    print("\nSummary:")
    print(summary_markdown(summary_rows))
    print(f"\nSaved tables to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the protocol-v2 paper evaluations without mutating legacy artifacts.

The module keeps planning and dry-run operations write-free, then executes a
manifest-locked, resumable train/evaluate state machine for actual runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.event_masks import build_event_masks, corrected_p90_spec, fit_event_thresholds
from lib.metrics import corrected_event_metrics


SCHEMA_VERSION = 2
PROTOCOL = {
    "protocol_id": "train_all_node_flow_p90_valid_v2",
    "percentile": 90.0,
    "valid_min": 5.0,
    "comparison": "gt",
    "fit_population": "all_targets",
}
REQUIRED_FLOWS = ("inflow", "outflow", "mean")
REQUIRED_DATA_FILES = ("train.npz", "val.npz", "test.npz", "adj_mx.npz")
FLOW_NAMES = ("inflow", "outflow")
PER_SEED_FIELDS = (
    "dataset", "variant", "seed", "flow", "mae", "eee", "normal_mae",
    "event_signed_error", "valid_count", "event_count", "normal_count",
    "valid_abs_error_sum", "event_abs_error_sum", "normal_abs_error_sum",
    "event_signed_error_sum", "active_parameters", "train_protocol",
    "evaluation_protocol", "evaluation_protocol_fingerprint",
    "evaluation_fingerprint", "checkpoint", "checkpoint_sha256",
    "prediction", "prediction_sha256",
)
SUMMARY_METRICS = ("mae", "eee", "normal_mae", "event_signed_error")
MODEL_DEFAULTS = {
    "S_Loss": 0,
    "T_Loss": 0,
    "cheb_order": 3,
    "graph_init": "8_neighbours",
    "self_attention_flag": True,
    "cross_attention_flag": False,
    "feedforward_flag": False,
    "layer_norm_flag": False,
    "additional_sa_flag": False,
    "learnable_flag": False,
    "rank": 0,
    "pos_emb_flag": False,
    "add_8": False,
    "add_eye": False,
    "add_x_encoder": False,
    "freeze_encoder": False,
    "threshold_adj_mx": False,
    "affinity_conv": False,
    "loss": "mae",
    "variant": None,
    "phase3_mode": "original",
    "bias_param_scope": "head_only",
    "classification_loss_weight": 1.0,
    "event_loss_weight": 0.0,
    "scaler_fit": "train",
    "max_train_batches": None,
    "max_eval_batches": None,
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config_filename: str


@dataclass(frozen=True)
class VariantSpec:
    name: str
    action: str
    ablation_mode: str | None
    start_phase: str
    stop_after_phase: str
    evaluation_phase: str
    source_study: str | None = None
    source_root_dir: str | None = None
    source_metrics_filename: str = "per_seed_metrics.csv"
    source_variant: str | None = None
    load_source_variant: str | None = None
    event_loss_weight: float = 0.25
    bias_param_scope: str | None = None


@dataclass(frozen=True)
class RunUnit:
    dataset: str
    variant: str
    seed: int
    action: str
    ablation_mode: str | None
    start_phase: str
    stop_after_phase: str
    evaluation_phase: str
    load_source_variant: str | None
    event_loss_weight: float
    bias_param_scope: str | None


@dataclass(frozen=True)
class SourceCheckpoint:
    dataset: str
    variant: str
    seed: int
    csv_path: Path
    csv_sha256: str
    checkpoint: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class DryRunResult:
    plan: tuple[RunUnit, ...]
    source_checkpoints: tuple[SourceCheckpoint, ...]
    manifest: dict


@dataclass(frozen=True)
class ExecutionResult:
    output_dir: Path
    unit_records: tuple[Path, ...]
    trained_units: int
    evaluated_units: int
    skipped_units: int


DATASET_SPECS = (
    DatasetSpec("NYCTaxi", "configs/NYCTaxi.yaml"),
    DatasetSpec("NYCBike1", "configs/NYCBike1.yaml"),
    DatasetSpec("NYCBike2", "configs/NYCBike2.yaml"),
    DatasetSpec("BJTaxi", "configs/BJTaxi.yaml"),
)

VARIANT_SPECS = (
    VariantSpec(
        name="base_mae",
        action="reuse_eval",
        ablation_mode=None,
        start_phase="pred",
        stop_after_phase="pred",
        evaluation_phase="pred",
        source_study="paper_rescue",
        source_root_dir="paper_rescue_ablation_results",
        source_metrics_filename="shared_phase1_metrics.csv",
        source_variant="shared_phase1",
        event_loss_weight=0.0,
        bias_param_scope="head_only",
    ),
    VariantSpec(
        name="event_weighted_base",
        action="train_eval",
        ablation_mode="event_weighted_base",
        start_phase="pred",
        stop_after_phase="pred",
        evaluation_phase="pred",
        event_loss_weight=0.25,
        bias_param_scope="head_only",
    ),
    VariantSpec(
        name="frozen_residual_mae",
        action="reuse_eval",
        ablation_mode="ungated_bias",
        start_phase="pred_2",
        stop_after_phase="pred_2",
        evaluation_phase="pred_2",
        source_study="residual_schedule",
        source_root_dir="residual_training_schedule_ablation_results",
        source_variant="B_frozen_residual_mae",
        event_loss_weight=0.0,
        bias_param_scope="head_only",
    ),
    VariantSpec(
        name="joint_event_weighted_residual",
        action="train_eval",
        ablation_mode="event_weighted_ungated_residual",
        start_phase="bias",
        stop_after_phase="bias",
        evaluation_phase="bias",
        load_source_variant="base_mae",
        bias_param_scope="head_only",
    ),
    VariantSpec(
        name="frozen_event_weighted_residual",
        action="train_eval",
        ablation_mode="event_weighted_ungated_residual",
        start_phase="pred_2",
        stop_after_phase="pred_2",
        evaluation_phase="pred_2",
        load_source_variant="base_mae",
        bias_param_scope="head_only",
    ),
)


def canonical_json(value: object) -> str:
    """Return the one JSON encoding used for all runner fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_spec(dataset: str | DatasetSpec) -> DatasetSpec:
    if isinstance(dataset, DatasetSpec):
        return dataset
    for spec in DATASET_SPECS:
        if spec.name == dataset:
            return spec
    raise ValueError(f"Unknown corrected-runner dataset: {dataset}")


def _variant_spec(variant: str | VariantSpec) -> VariantSpec:
    if isinstance(variant, VariantSpec):
        return variant
    for spec in VARIANT_SPECS:
        if spec.name == variant:
            return spec
    raise ValueError(f"Unknown corrected-runner variant: {variant}")


def _normalize_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized or any(seed < 1 for seed in normalized):
        raise ValueError("seeds must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must not contain duplicates")
    return normalized


def build_execution_plan(seeds: Sequence[int] = (1, 2, 3)) -> tuple[RunUnit, ...]:
    """Build the fixed dataset/seed/variant execution graph in paper-row order."""
    normalized_seeds = _normalize_seeds(seeds)

    return tuple(
        RunUnit(
            dataset=dataset.name,
            variant=variant.name,
            seed=seed,
            action=variant.action,
            ablation_mode=variant.ablation_mode,
            start_phase=variant.start_phase,
            stop_after_phase=variant.stop_after_phase,
            evaluation_phase=variant.evaluation_phase,
            load_source_variant=variant.load_source_variant,
            event_loss_weight=variant.event_loss_weight,
            bias_param_scope=variant.bias_param_scope,
        )
        for dataset in DATASET_SPECS
        for seed in normalized_seeds
        for variant in VARIANT_SPECS
    )


def _source_csv_path(source_root: Path, dataset: DatasetSpec, variant: VariantSpec) -> Path:
    if not variant.source_study or not variant.source_root_dir or not variant.source_variant:
        raise ValueError(f"Variant {variant.name} has no reusable source checkpoint")
    if dataset.name == "NYCTaxi":
        return source_root / variant.source_root_dir / variant.source_metrics_filename
    return (
        source_root
        / "final_all_dataset_ablations"
        / dataset.name
        / variant.source_study
        / variant.source_metrics_filename
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required source checkpoint CSV is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"dataset", "variant", "seed", "flow", "checkpoint"}
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Source checkpoint CSV is missing columns {sorted(missing)}: {path}")
        return list(reader)


def resolve_source_checkpoint(
    source_root: str | Path,
    dataset: str | DatasetSpec,
    variant: str | VariantSpec,
    seed: int,
) -> SourceCheckpoint:
    """Resolve one complete, existing source checkpoint from legacy metric CSVs."""
    dataset_spec = _dataset_spec(dataset)
    variant_spec = _variant_spec(variant)
    csv_path = _source_csv_path(Path(source_root), dataset_spec, variant_spec)
    selected = [
        row
        for row in _read_csv_rows(csv_path)
        if row["dataset"].strip() == dataset_spec.name
        and row["variant"].strip() == variant_spec.source_variant
        and int(float(row["seed"])) == int(seed)
    ]
    flows = {row["flow"].strip() for row in selected}
    if len(selected) != len(REQUIRED_FLOWS) or flows != set(REQUIRED_FLOWS):
        raise ValueError(
            "Source checkpoint rows must contain exactly inflow, outflow, mean "
            f"for dataset={dataset_spec.name}, variant={variant_spec.source_variant}, seed={seed}"
        )

    checkpoint_values = {row["checkpoint"].strip() for row in selected}
    if len(checkpoint_values) != 1 or not next(iter(checkpoint_values)):
        raise ValueError(
            "Source checkpoint rows must agree on one nonempty checkpoint for "
            f"dataset={dataset_spec.name}, variant={variant_spec.source_variant}, seed={seed}"
        )
    checkpoint = Path(next(iter(checkpoint_values))).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = csv_path.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resolved source checkpoint is missing: {checkpoint}")

    return SourceCheckpoint(
        dataset=dataset_spec.name,
        variant=variant_spec.name,
        seed=int(seed),
        csv_path=csv_path.resolve(),
        csv_sha256=file_digest(csv_path),
        checkpoint=checkpoint,
        checkpoint_sha256=file_digest(checkpoint),
    )


def _hash_existing_paths(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    result = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest input is missing: {path}")
        result.append({"path": str(path), "sha256": file_digest(path)})
    return sorted(result, key=lambda entry: entry["path"])


def _default_code_files() -> tuple[Path, ...]:
    return (
        Path(__file__),
        ROOT / "main.py",
        ROOT / "model" / "models.py",
        ROOT / "model" / "layers.py",
        ROOT / "model" / "trainer.py",
        ROOT / "lib" / "dataloader.py",
        ROOT / "lib" / "metrics.py",
        ROOT / "lib" / "event_masks.py",
        ROOT / "lib" / "utils.py",
        ROOT / "lib" / "logger.py",
    )


def _resolve_data_root(data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        candidate = Path(data_dir).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_dir():
            raise FileNotFoundError(f"Requested data directory is missing: {candidate}")
        return candidate.resolve()

    for candidate in (ROOT / "preprocessed_data", ROOT / "external" / "ST-SSL_Dataset"):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "No default data directory found; pass --data-dir with dataset split files"
    )


def _dataset_data_files(data_root: Path) -> tuple[Path, ...]:
    return tuple(
        data_root / dataset.name / filename
        for dataset in DATASET_SPECS
        for filename in REQUIRED_DATA_FILES
    )


def _protocol_fingerprint() -> str:
    return sha256_text(canonical_json(PROTOCOL))


def build_run_manifest(
    seeds: Sequence[int] = (1, 2, 3),
    smoke_test: bool = False,
    event_loss_weight: float = 0.25,
    *,
    source_checkpoints: Iterable[SourceCheckpoint] = (),
    code_files: Iterable[str | Path] = (),
    config_files: Iterable[str | Path] = (),
    data_files: Iterable[str | Path] = (),
    python_environment: str = "metadata",
    device: str = "unspecified",
) -> dict:
    """Build a canonical, protocol-v2 manifest and attach its stable fingerprint."""
    normalized_seeds = _normalize_seeds(seeds)
    if smoke_test and normalized_seeds != (1,):
        raise ValueError("smoke manifests must use exactly seed 1")
    if float(event_loss_weight) != 0.25:
        raise ValueError("corrected paper results require event_loss_weight=0.25")

    supplied_code = tuple(code_files)
    supplied_configs = tuple(config_files)
    default_code = _default_code_files() if not supplied_code else supplied_code
    default_configs = (
        tuple(ROOT / dataset.config_filename for dataset in DATASET_SPECS)
        if not supplied_configs
        else supplied_configs
    )
    checkpoint_entries = sorted(
        (
            {
                "dataset": source.dataset,
                "variant": source.variant,
                "seed": source.seed,
                "csv_path": str(source.csv_path),
                "csv_sha256": source.csv_sha256,
                "checkpoint": str(source.checkpoint),
                "checkpoint_sha256": source.checkpoint_sha256,
            }
            for source in source_checkpoints
        ),
        key=lambda entry: (entry["dataset"], entry["variant"], entry["seed"]),
    )
    source_manifests = sorted(
        {
            (entry["csv_path"], entry["csv_sha256"])
            for entry in checkpoint_entries
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": dict(PROTOCOL),
        "protocol_fingerprint": _protocol_fingerprint(),
        "dataset_order": [dataset.name for dataset in DATASET_SPECS],
        "variants": [asdict(variant) for variant in VARIANT_SPECS],
        "seeds": list(normalized_seeds),
        "event_loss_weight": float(event_loss_weight),
        "scaler_policy": "train_only",
        "mode": "smoke" if smoke_test else "full",
        "python_environment": python_environment,
        "device": str(device),
        "code_hashes": _hash_existing_paths(default_code),
        "config_hashes": _hash_existing_paths(default_configs),
        "data_hashes": _hash_existing_paths(data_files),
        "source_manifest_hashes": [
            {"path": path, "sha256": digest} for path, digest in source_manifests
        ],
        "source_checkpoint_hashes": checkpoint_entries,
    }
    manifest["fingerprint"] = sha256_text(canonical_json(manifest))
    return manifest


def validate_resume_manifest(output_dir: str | Path, expected_manifest: Mapping[str, object]) -> dict | None:
    """Fail closed when a nonempty output directory lacks a matching manifest."""
    root = Path(output_dir)
    if not root.exists():
        return None
    if not root.is_dir():
        raise ValueError(f"Correction output path is not a directory: {root}")
    contents = list(root.iterdir())
    if not contents:
        return None
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Nonempty correction output lacks run_manifest.json: {root}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read correction run_manifest.json: {manifest_path}") from error
    stored_fingerprint = actual.get("fingerprint")
    stored_payload = dict(actual)
    stored_payload.pop("fingerprint", None)
    if not isinstance(stored_fingerprint, str) or sha256_text(canonical_json(stored_payload)) != stored_fingerprint:
        raise ValueError("Stored correction manifest fingerprint does not match its payload")
    if actual.get("fingerprint") != expected_manifest.get("fingerprint"):
        raise ValueError("Correction output manifest is incompatible with the requested run")
    if actual.get("protocol_fingerprint") != expected_manifest.get("protocol_fingerprint"):
        raise ValueError("Correction output manifest protocol is incompatible with the requested run")
    return actual


def _required_source_variant(unit: RunUnit) -> str | None:
    if unit.action == "reuse_eval":
        return unit.variant
    return unit.load_source_variant


def dry_run(
    *,
    source_root: str | Path = ROOT,
    data_dir: str | Path | None = None,
    output_dir: str | Path = ROOT / "corrected_valid_target_p90_results",
    seeds: Sequence[int] = (1, 2, 3),
    smoke_test: bool = False,
    event_loss_weight: float = 0.25,
    python_environment: str = "metadata",
    device: str = "unspecified",
) -> DryRunResult:
    """Validate every planned source dependency without creating the output root."""
    selected_seeds = (1,) if smoke_test else tuple(int(seed) for seed in seeds)
    plan = build_execution_plan(selected_seeds)
    data_root = _resolve_data_root(data_dir)
    source_keys: set[tuple[str, str, int]] = set()
    resolved: list[SourceCheckpoint] = []
    for unit in plan:
        source_variant = _required_source_variant(unit)
        if source_variant is None:
            continue
        key = (unit.dataset, source_variant, unit.seed)
        if key in source_keys:
            continue
        source_keys.add(key)
        resolved.append(
            resolve_source_checkpoint(source_root, unit.dataset, source_variant, unit.seed)
        )
    manifest = build_run_manifest(
        seeds=selected_seeds,
        smoke_test=smoke_test,
        event_loss_weight=event_loss_weight,
        source_checkpoints=resolved,
        data_files=_dataset_data_files(data_root),
        python_environment=python_environment,
        device=device,
    )
    validate_resume_manifest(_smoke_output_dir(output_dir, smoke_test), manifest)
    return DryRunResult(tuple(plan), tuple(resolved), manifest)


run_dry_run = dry_run


def _json_safe(value):
    """Convert runtime values to strict-JSON values (nonfinite numbers -> null)."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_safe(payload), handle, sort_keys=True, indent=2,
                ensure_ascii=True, allow_nan=False,
            )
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(_json_safe(dict(row)))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read authoritative unit JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Authoritative unit JSON must contain an object: {path}")
    return value


def _array_digest(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _semantic_fingerprint(payload: Mapping[str, object]) -> str:
    return sha256_text(canonical_json(_json_safe(payload)))


def _safe_restart_root(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == Path(resolved.anchor) or resolved == ROOT.resolve():
        raise ValueError(f"Refusing to restart unsafe correction output root: {resolved}")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"Correction output path is not a directory: {resolved}")
        archive = resolved.with_name(f"{resolved.name}.restart-{uuid4().hex[:12]}")
        resolved.replace(archive)


def _smoke_output_dir(output_dir: str | Path, smoke_test: bool) -> Path:
    output = Path(output_dir).expanduser()
    if smoke_test and not output.name.endswith("_smoke"):
        output = output.with_name(output.name + "_smoke")
    return output.resolve()


def _copy_checkpoint(source: Path, destination: Path) -> str:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {source}")
    source_hash = file_digest(source)
    if destination.is_file() and file_digest(destination) == source_hash:
        return source_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        if file_digest(temporary) != source_hash:
            raise ValueError(f"Checkpoint copy hash mismatch: {source} -> {destination}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return source_hash


def _source_map(source_checkpoints: Iterable[SourceCheckpoint]) -> dict[tuple[str, str, int], SourceCheckpoint]:
    result = {}
    for source in source_checkpoints:
        key = (source.dataset, source.variant, int(source.seed))
        if key in result:
            raise ValueError(f"Duplicate source checkpoint dependency: {key}")
        if not source.checkpoint.is_file() or file_digest(source.checkpoint) != source.checkpoint_sha256:
            raise ValueError(f"Source checkpoint hash no longer matches manifest: {source.checkpoint}")
        result[key] = source
    return result


def _unit_record_path(output_root: Path, unit: RunUnit) -> Path:
    return output_root / unit.dataset / "units" / unit.variant / f"seed_{unit.seed}.json"


def _local_checkpoint_path(output_root: Path, unit: RunUnit) -> Path:
    return output_root / unit.dataset / "checkpoints" / unit.variant / f"seed_{unit.seed}.pth"


def _prediction_path(output_root: Path, unit: RunUnit) -> Path:
    return output_root / unit.dataset / "predictions" / unit.variant / f"seed_{unit.seed}.npz"


def _load_config(dataset: str) -> dict:
    config_path = ROOT / _dataset_spec(dataset).config_filename
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config must contain a mapping: {config_path}")
    merged = dict(MODEL_DEFAULTS)
    merged.update(config)
    return merged


def _build_supervisor_args(
    unit: RunUnit,
    *,
    mode: str,
    data_root: Path,
    device: str,
    best_path: Path | None,
    load_path: Path | None,
    smoke_test: bool,
    max_train_batches: int | None,
    max_eval_batches: int | None,
) -> Namespace:
    """Load architecture settings, then explicitly replace every scientific run knob."""
    config = _load_config(unit.dataset)
    config.update(
        {
            "mode": mode,
            "best_path": str(best_path.resolve()) if best_path is not None else None,
            "load_path": str(load_path.resolve()) if load_path is not None else None,
            "data_dir": str(data_root.resolve()),
            "dataset": unit.dataset,
            "graph_file": str((data_root / unit.dataset / "adj_mx.npz").resolve()),
            "scaler_fit": "train",
            "event_percentile": None,
            "event_mask_protocol": PROTOCOL["protocol_id"],
            "event_label_source": "file_verified",
            "device": str(device),
            "ablation_mode": unit.ablation_mode or "original",
            "bias_param_scope": unit.bias_param_scope or "head_only",
            "phase3_mode": "original",
            "start_phase": unit.start_phase,
            "stop_after_phase": unit.stop_after_phase,
            "evaluation_phase": unit.evaluation_phase,
            "event_loss_weight": float(unit.event_loss_weight),
            "capture_predictions": mode == "test",
            "raise_exceptions": True,
            "max_train_batches": max_train_batches,
            "max_eval_batches": max_eval_batches,
            "variant": None,
            "debug": False,
            "experimentName": (
                f"corrected_valid_p90_{unit.dataset}_{unit.variant}_seed={unit.seed}"
                + ("_smoke" if smoke_test else "")
            ),
            "comment": "corrected_valid_target_p90_results",
            "seed": int(unit.seed),
        }
    )
    if smoke_test:
        config.update({"epochs": 1, "num_epochs": 1, "early_stop": False})
    return Namespace(**config)


def _load_split(data_root: Path, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = data_root / dataset / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "y" not in archive or "evs_90" not in archive:
            raise ValueError(f"Dataset split must contain y and evs_90: {path}")
        return np.asarray(archive["y"]), np.asarray(archive["evs_90"])


def _prepare_dataset_thresholds(data_root: Path, output_root: Path, dataset: str) -> dict:
    """Fit once from full training targets and verify all saved raw p90 labels."""
    spec = corrected_p90_spec()
    train_y, _ = _load_split(data_root, dataset, "train")
    thresholds = fit_event_thresholds(train_y, spec)
    mask_fingerprints = {}
    split_shapes = {}
    test_bundle = None
    test_target = None
    for split in ("train", "val", "test"):
        target, raw = _load_split(data_root, dataset, split)
        bundle = build_event_masks(target, thresholds, spec, raw_labels=raw)
        mask_fingerprints[split] = bundle.fingerprint
        split_shapes[split] = list(target.shape)
        if split == "test":
            test_bundle = bundle
            test_target = target
    assert test_bundle is not None and test_target is not None

    threshold_path = output_root / dataset / "thresholds.npz"
    expected_threshold_fingerprint = _semantic_fingerprint(
        {
            "protocol_fingerprint": spec.fingerprint(),
            "values": _array_digest(np.asarray(thresholds.values)),
        }
    )
    if threshold_path.is_file():
        with np.load(threshold_path, allow_pickle=False) as archive:
            stored_values = np.asarray(archive["values"])
            stored_protocol = str(np.asarray(archive["protocol_fingerprint"]).item())
            stored_fingerprint = str(np.asarray(archive["threshold_fingerprint"]).item())
        if (
            stored_protocol != spec.fingerprint()
            or stored_fingerprint != expected_threshold_fingerprint
            or not np.array_equal(stored_values, thresholds.values)
        ):
            raise ValueError(f"Stored thresholds are incompatible for {dataset}")
    else:
        _atomic_npz(
            threshold_path,
            values=np.asarray(thresholds.values),
            protocol_fingerprint=np.asarray(spec.fingerprint()),
            threshold_fingerprint=np.asarray(expected_threshold_fingerprint),
        )

    return {
        "spec": spec,
        "thresholds": thresholds,
        "threshold_path": threshold_path,
        "threshold_sha256": file_digest(threshold_path),
        "threshold_fingerprint": expected_threshold_fingerprint,
        "mask_fingerprints": mask_fingerprints,
        "split_shapes": split_shapes,
        "test_target": np.asarray(test_target),
        "test_event": np.asarray(test_bundle.event, dtype=bool),
        "test_valid": np.asarray(test_bundle.valid, dtype=bool),
    }


def _context_fingerprint(dataset: str, target: np.ndarray, event: np.ndarray, valid: np.ndarray) -> str:
    return _semantic_fingerprint(
        {
            "dataset": dataset,
            "protocol_fingerprint": _protocol_fingerprint(),
            "target": _array_digest(np.asarray(target, dtype=np.float32)),
            "event": _array_digest(np.asarray(event, dtype=bool)),
            "valid": _array_digest(np.asarray(valid, dtype=bool)),
        }
    )


def _normalize_captured_artifacts(result: Mapping[str, object] | None) -> dict[str, np.ndarray]:
    if not isinstance(result, Mapping):
        raise ValueError("Evaluation supervisor returned no result mapping")
    artifacts = result.get("test_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Captured evaluation did not return test_artifacts")
    required = {"prediction", "target", "event", "valid"}
    missing = required - set(artifacts)
    if missing:
        raise ValueError(f"Captured evaluation artifacts are missing {sorted(missing)}")
    normalized = {}
    for name in required:
        value = artifacts[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        normalized[name] = np.asarray(value)
    shapes = {array.shape for array in normalized.values()}
    if len(shapes) != 1 or not next(iter(shapes)):
        raise ValueError("Captured prediction, target, event, and valid must share a nonempty shape")
    if normalized["prediction"].shape[-1] != len(FLOW_NAMES):
        raise ValueError("Corrected paper evaluation requires exactly inflow and outflow")
    if not np.all(np.isfinite(normalized["prediction"])):
        raise ValueError("Captured predictions must be finite")
    if not np.all(np.isfinite(normalized["target"])):
        raise ValueError("Captured targets must be finite")
    for name in ("event", "valid"):
        values = normalized[name]
        if not (
            np.issubdtype(values.dtype, np.bool_)
            or np.all((values == 0) | (values == 1))
        ):
            raise ValueError(f"Captured {name} mask must be bool or exact binary")
        normalized[name] = values.astype(bool)
    normalized["prediction"] = normalized["prediction"].astype(np.float32, copy=False)
    normalized["target"] = normalized["target"].astype(np.float32, copy=False)
    return normalized


def _validate_captured_context(dataset_info: Mapping[str, object], captured: Mapping[str, np.ndarray]) -> None:
    target = captured["target"]
    sample_count = target.shape[0]
    expected_target = np.asarray(dataset_info["test_target"])[:sample_count]
    expected_event = np.asarray(dataset_info["test_event"])[:sample_count]
    expected_valid = np.asarray(dataset_info["test_valid"])[:sample_count]
    if target.shape != expected_target.shape:
        raise ValueError("Captured target shape is not a leading test-set slice")
    if not np.allclose(target, expected_target, rtol=1e-5, atol=1e-4):
        raise ValueError("Captured inverse-scaled targets do not match the local test split")
    if not np.array_equal(captured["event"], expected_event):
        raise ValueError("Captured event mask does not match corrected v2 test events")
    if not np.array_equal(captured["valid"], expected_valid):
        raise ValueError("Captured valid mask does not match Y > 5")
    event = captured["event"]
    valid = captured["valid"]
    normal = valid & ~event
    if np.any(event & ~valid) or np.any(event & normal) or not np.array_equal(event | normal, valid):
        raise ValueError("Captured event and normal masks do not partition valid targets")


def _ensure_evaluation_context(
    output_root: Path,
    dataset: str,
    dataset_info: Mapping[str, object],
    captured: Mapping[str, np.ndarray],
    *,
    allow_partial: bool = False,
) -> dict:
    if not allow_partial and captured["target"].shape[0] != np.asarray(dataset_info["test_target"]).shape[0]:
        raise ValueError("Full corrected runs must capture the complete test target")
    _validate_captured_context(dataset_info, captured)
    target = np.asarray(captured["target"], dtype=np.float32)
    event = np.asarray(captured["event"], dtype=bool)
    valid = np.asarray(captured["valid"], dtype=bool)
    fingerprint = _context_fingerprint(dataset, target, event, valid)
    context_path = output_root / dataset / "evaluation_context.npz"
    if context_path.is_file():
        with np.load(context_path, allow_pickle=False) as archive:
            stored = {
                "target": np.asarray(archive["target"]),
                "event": np.asarray(archive["event"]).astype(bool),
                "valid": np.asarray(archive["valid"]).astype(bool),
                "evaluation_fingerprint": str(np.asarray(archive["evaluation_fingerprint"]).item()),
                "protocol_fingerprint": str(np.asarray(archive["protocol_fingerprint"]).item()),
            }
        if (
            stored["evaluation_fingerprint"] != fingerprint
            or stored["protocol_fingerprint"] != _protocol_fingerprint()
            or not np.allclose(stored["target"], target, rtol=1e-6, atol=1e-6)
            or not np.array_equal(stored["event"], event)
            or not np.array_equal(stored["valid"], valid)
        ):
            raise ValueError(f"Captured evaluation context is inconsistent within {dataset}")
    else:
        _atomic_npz(
            context_path,
            target=target,
            event=event,
            valid=valid,
            evaluation_fingerprint=np.asarray(fingerprint),
            protocol_fingerprint=np.asarray(_protocol_fingerprint()),
        )
    return {
        "path": context_path,
        "sha256": file_digest(context_path),
        "fingerprint": fingerprint,
        "target": target,
        "event": event,
        "valid": valid,
    }


def _load_existing_context(output_root: Path, dataset: str) -> dict | None:
    path = output_root / dataset / "evaluation_context.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            target = np.asarray(archive["target"], dtype=np.float32)
            event = np.asarray(archive["event"]).astype(bool)
            valid = np.asarray(archive["valid"]).astype(bool)
            fingerprint = str(np.asarray(archive["evaluation_fingerprint"]).item())
            protocol = str(np.asarray(archive["protocol_fingerprint"]).item())
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"Cannot read shared evaluation context: {path}") from error
    expected = _context_fingerprint(dataset, target, event, valid)
    if protocol != _protocol_fingerprint() or fingerprint != expected:
        raise ValueError(f"Shared evaluation context fingerprint mismatch: {path}")
    return {
        "path": path,
        "sha256": file_digest(path),
        "fingerprint": fingerprint,
        "target": target,
        "event": event,
        "valid": valid,
    }


def _assert_partition_and_summarize(
    prediction: np.ndarray,
    context: Mapping[str, object],
    unit: RunUnit,
    *,
    active_parameters: object = None,
) -> list[dict]:
    target = np.asarray(context["target"], dtype=np.float32)
    event = np.asarray(context["event"], dtype=bool)
    valid = np.asarray(context["valid"], dtype=bool)
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError("Prediction shape does not match shared evaluation target")
    if not np.array_equal(valid, target > PROTOCOL["valid_min"]):
        raise ValueError("Shared valid mask does not equal strict Y > 5")

    rows = []
    for flow_index, flow_name in enumerate(FLOW_NAMES):
        detail = corrected_event_metrics(
            prediction[..., flow_index],
            target[..., flow_index],
            event[..., flow_index],
            valid_min=PROTOCOL["valid_min"],
        )
        if detail["valid_count"] != detail["event_count"] + detail["normal_count"]:
            raise ValueError(f"Event/normal count partition failed for {unit.dataset} {flow_name}")
        if not np.isclose(
            detail["valid_abs_error_sum"],
            detail["event_abs_error_sum"] + detail["normal_abs_error_sum"],
            rtol=1e-6,
            atol=1e-5,
        ):
            raise ValueError(f"Absolute-error partition failed for {unit.dataset} {flow_name}")
        rows.append(
            {
                "dataset": unit.dataset,
                "variant": unit.variant,
                "seed": unit.seed,
                "flow": flow_name,
                "mae": detail["valid_mae"],
                "eee": detail["event_mae"],
                "normal_mae": detail["normal_mae"],
                "event_signed_error": detail["event_signed_error"],
                "valid_count": detail["valid_count"],
                "event_count": detail["event_count"],
                "normal_count": detail["normal_count"],
                "valid_abs_error_sum": detail["valid_abs_error_sum"],
                "event_abs_error_sum": detail["event_abs_error_sum"],
                "normal_abs_error_sum": detail["normal_abs_error_sum"],
                "event_signed_error_sum": detail["event_signed_error_sum"],
                "active_parameters": active_parameters,
            }
        )

    mean_row = {
        "dataset": unit.dataset,
        "variant": unit.variant,
        "seed": unit.seed,
        "flow": "mean",
        "active_parameters": active_parameters,
    }
    for metric in ("mae", "eee", "normal_mae", "event_signed_error"):
        values = [row[metric] for row in rows]
        mean_row[metric] = float(np.mean(values)) if all(math.isfinite(value) for value in values) else float("nan")
    for metric in (
        "valid_count", "event_count", "normal_count", "valid_abs_error_sum",
        "event_abs_error_sum", "normal_abs_error_sum", "event_signed_error_sum",
    ):
        mean_row[metric] = sum(row[metric] for row in rows)
    rows.append(mean_row)
    return rows


def _training_protocol(unit: RunUnit) -> tuple[str, str | None]:
    if unit.action == "reuse_eval":
        return "legacy_event_independent", None
    return PROTOCOL["protocol_id"], _protocol_fingerprint()


def _training_fingerprint(
    unit: RunUnit,
    source: SourceCheckpoint | None,
    manifest: Mapping[str, object],
    *,
    device: str,
    max_train_batches: int | None,
    max_eval_batches: int | None,
) -> str:
    protocol_name, protocol_fingerprint = _training_protocol(unit)
    return _semantic_fingerprint(
        {
            "unit": asdict(unit),
            "train_protocol": protocol_name,
            "training_protocol_fingerprint": protocol_fingerprint,
            "validation_protocol_fingerprint": protocol_fingerprint,
            "source_checkpoint_sha256": source.checkpoint_sha256 if source else None,
            "manifest_mode": manifest.get("mode"),
            "scaler_policy": "train_only",
            "device": str(device),
            "max_train_batches": max_train_batches,
            "max_eval_batches": max_eval_batches,
        }
    )


def _record_checkpoint_is_valid(record: Mapping[str, object] | None, path: Path) -> bool:
    if not record or record.get("status") not in {"trained", "complete"}:
        return False
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not path.is_file():
        return False
    expected = checkpoint.get("sha256")
    return isinstance(expected, str) and file_digest(path) == expected


def _prediction_is_valid(record: Mapping[str, object] | None, path: Path, context: Mapping[str, object] | None) -> bool:
    if not record or record.get("status") != "complete" or context is None or not path.is_file():
        return False
    prediction = record.get("prediction")
    if not isinstance(prediction, Mapping):
        return False
    if prediction.get("sha256") != file_digest(path):
        return False
    if prediction.get("evaluation_context_sha256") != context.get("sha256"):
        return False
    if record.get("evaluation_fingerprint") != context.get("fingerprint"):
        return False
    if record.get("evaluation_protocol_fingerprint") != _protocol_fingerprint():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"prediction", "evaluation_context_sha256"}:
                return False
            stored_context = str(np.asarray(archive["evaluation_context_sha256"]).item())
            prediction_array = np.asarray(archive["prediction"])
    except (OSError, KeyError, ValueError):
        return False
    return (
        stored_context == context.get("sha256")
        and prediction_array.shape == np.asarray(context["target"]).shape
        and np.all(np.isfinite(prediction_array))
    )


def _unit_identity_matches(record: Mapping[str, object] | None, unit: RunUnit) -> bool:
    return bool(
        record
        and record.get("dataset") == unit.dataset
        and record.get("variant") == unit.variant
        and int(record.get("seed", -1)) == unit.seed
        and record.get("action") == unit.action
    )


def _select_trained_checkpoint(
    train_args: Namespace,
    train_result: Mapping[str, object] | None,
    evaluation_phase: str,
) -> tuple[Path, str]:
    log_dir = getattr(train_args, "log_dir", None)
    if log_dir:
        phase_checkpoint = Path(log_dir) / f"best_model_{evaluation_phase}.pth"
        if phase_checkpoint.is_file():
            return phase_checkpoint.resolve(), "phase_checkpoint"
    if isinstance(train_result, Mapping):
        for key in ("selected_checkpoint", "checkpoint", "best_path"):
            value = train_result.get(key)
            if value and Path(value).is_file():
                return Path(value).resolve(), "supervisor_result"
    load_path = getattr(train_args, "load_path", None)
    if load_path and Path(load_path).is_file():
        return Path(load_path).resolve(), "load_path_fallback"
    raise FileNotFoundError(
        f"Training produced no best_model_{evaluation_phase}.pth and has no valid fallback"
    )


def _extract_active_parameters(*results: Mapping[str, object] | None):
    for result in results:
        if not isinstance(result, Mapping):
            continue
        counts = result.get("parameter_counts")
        if isinstance(counts, Mapping) and counts.get("active") is not None:
            value = counts["active"]
            return int(value) if isinstance(value, (int, np.integer)) else float(value)
    return None


def _write_source_checkpoint_csv(output_root: Path, dataset: str, sources: Iterable[SourceCheckpoint]) -> None:
    rows = [
        {
            "dataset": source.dataset,
            "variant": source.variant,
            "seed": source.seed,
            "source_csv": str(source.csv_path),
            "source_csv_sha256": source.csv_sha256,
            "source_checkpoint": str(source.checkpoint),
            "source_checkpoint_sha256": source.checkpoint_sha256,
        }
        for source in sources
        if source.dataset == dataset
    ]
    rows.sort(key=lambda row: (row["seed"], row["variant"]))
    _atomic_csv(
        output_root / dataset / "source_checkpoints.csv",
        (
            "dataset", "variant", "seed", "source_csv", "source_csv_sha256",
            "source_checkpoint", "source_checkpoint_sha256",
        ),
        rows,
    )


def _write_dataset_manifest(
    output_root: Path,
    dataset: str,
    dataset_info: Mapping[str, object],
    run_manifest: Mapping[str, object],
    context: Mapping[str, object] | None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "run_manifest_fingerprint": run_manifest.get("fingerprint"),
        "protocol": dict(PROTOCOL),
        "protocol_fingerprint": _protocol_fingerprint(),
        "thresholds": {
            "path": str(dataset_info["threshold_path"]),
            "sha256": dataset_info["threshold_sha256"],
            "fingerprint": dataset_info["threshold_fingerprint"],
        },
        "mask_fingerprints": dataset_info["mask_fingerprints"],
        "split_shapes": dataset_info["split_shapes"],
        "evaluation_context": None if context is None else {
            "path": str(context["path"]),
            "sha256": context["sha256"],
            "fingerprint": context["fingerprint"],
        },
    }
    payload["fingerprint"] = _semantic_fingerprint(payload)
    _atomic_json(output_root / dataset / "dataset_manifest.json", payload)


def _default_supervisor(args: Namespace):
    # Lazy import keeps dry-run free of torch/device initialization.
    from main import model_supervisor

    return model_supervisor(args)


def execute_plan(
    plan: Sequence[RunUnit],
    *,
    source_checkpoints: Iterable[SourceCheckpoint],
    manifest: Mapping[str, object],
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    train_supervisor: Callable[[Namespace], Mapping[str, object] | None] | None = None,
    evaluate_supervisor: Callable[[Namespace], Mapping[str, object] | None] | None = None,
    restart: bool = False,
    smoke_test: bool = False,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> ExecutionResult:
    """Execute or resume a manifest-locked list of corrected paper units.

    Training and evaluation callables are separate and injectable so state
    transitions can be tested without starting a model process.
    """
    units = tuple(plan)
    if not units:
        raise ValueError("execute_plan requires at least one run unit")
    if manifest.get("protocol_fingerprint") != _protocol_fingerprint():
        raise ValueError("Execution manifest does not use the corrected event protocol")
    expected_mode = "smoke" if smoke_test else "full"
    if manifest.get("mode") != expected_mode:
        raise ValueError("Execution smoke/full mode does not match the run manifest")
    if manifest.get("device", "unspecified") not in {str(device), "unspecified"}:
        raise ValueError("Execution device does not match the run manifest")
    if smoke_test:
        if max_train_batches not in (None, 1) or max_eval_batches not in (None, 1):
            raise ValueError("Smoke runs require max_train_batches=max_eval_batches=1")
        max_train_batches = 1
        max_eval_batches = 1
    elif max_train_batches is not None or max_eval_batches is not None:
        raise ValueError("Batch limits are allowed only in smoke mode")
    data_root = _resolve_data_root(data_dir)
    output_root = _smoke_output_dir(output_dir, smoke_test)
    if restart:
        _safe_restart_root(output_root)
    validate_resume_manifest(output_root, manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.exists():
        _atomic_json(manifest_path, manifest)

    sources = tuple(source_checkpoints)
    sources_by_key = _source_map(sources)
    train_call = train_supervisor or _default_supervisor
    evaluate_call = evaluate_supervisor or _default_supervisor
    unit_record_paths = []
    trained_units = 0
    evaluated_units = 0
    skipped_units = 0
    dataset_cache: dict[str, dict] = {}
    context_cache: dict[str, dict | None] = {}

    for unit in units:
        if unit.dataset not in {spec.name for spec in DATASET_SPECS}:
            raise ValueError(f"Plan contains unknown dataset: {unit.dataset}")
        if unit.variant not in {spec.name for spec in VARIANT_SPECS}:
            raise ValueError(f"Plan contains unknown variant: {unit.variant}")
        if unit.dataset not in dataset_cache:
            dataset_cache[unit.dataset] = _prepare_dataset_thresholds(
                data_root, output_root, unit.dataset
            )
            context_cache[unit.dataset] = _load_existing_context(output_root, unit.dataset)
            if context_cache[unit.dataset] is not None:
                if (
                    not smoke_test
                    and np.asarray(context_cache[unit.dataset]["target"]).shape[0]
                    != np.asarray(dataset_cache[unit.dataset]["test_target"]).shape[0]
                ):
                    raise ValueError("Full corrected runs require a complete shared test context")
                _validate_captured_context(dataset_cache[unit.dataset], context_cache[unit.dataset])
            _write_source_checkpoint_csv(output_root, unit.dataset, sources)
            _write_dataset_manifest(
                output_root, unit.dataset, dataset_cache[unit.dataset], manifest,
                context_cache[unit.dataset],
            )

        record_path = _unit_record_path(output_root, unit)
        checkpoint_path = _local_checkpoint_path(output_root, unit)
        prediction_path = _prediction_path(output_root, unit)
        old_record = _read_json(record_path)
        if old_record is not None and not _unit_identity_matches(old_record, unit):
            raise ValueError(f"Unit JSON identity does not match its path: {record_path}")

        dependency_source = None
        if unit.action == "reuse_eval":
            dependency_source = sources_by_key.get((unit.dataset, unit.variant, unit.seed))
            if dependency_source is None:
                raise ValueError(f"Missing reusable source checkpoint for {unit}")
        elif unit.load_source_variant:
            dependency_source = sources_by_key.get(
                (unit.dataset, unit.load_source_variant, unit.seed)
            )
            if dependency_source is None:
                raise ValueError(f"Missing base checkpoint dependency for {unit}")

        desired_training_fingerprint = _training_fingerprint(
            unit, dependency_source, manifest,
            device=device,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        )
        protocol_name, training_protocol_fingerprint = _training_protocol(unit)
        training_compatible = bool(
            _unit_identity_matches(old_record, unit)
            and old_record.get("training_fingerprint") == desired_training_fingerprint
            and _record_checkpoint_is_valid(old_record, checkpoint_path)
        )
        if unit.action == "train_eval":
            training_compatible = training_compatible and bool(
                old_record.get("training_protocol_fingerprint") == training_protocol_fingerprint
                and old_record.get("validation_protocol_fingerprint") == training_protocol_fingerprint
            )

        train_result = None
        selection = None
        source_for_selected_checkpoint = None
        if not training_compatible:
            if unit.action == "reuse_eval":
                selected_checkpoint = dependency_source.checkpoint
                selection = "legacy_event_independent_source"
            elif unit.action == "train_eval":
                load_checkpoint = None
                if unit.load_source_variant:
                    base_unit = RunUnit(
                        dataset=unit.dataset,
                        variant=unit.load_source_variant,
                        seed=unit.seed,
                        action="reuse_eval",
                        ablation_mode=None,
                        start_phase="pred",
                        stop_after_phase="pred",
                        evaluation_phase="pred",
                        load_source_variant=None,
                        event_loss_weight=0.0,
                        bias_param_scope="head_only",
                    )
                    local_base = _local_checkpoint_path(output_root, base_unit)
                    local_base_record = _read_json(_unit_record_path(output_root, base_unit))
                    if _record_checkpoint_is_valid(local_base_record, local_base):
                        load_checkpoint = local_base
                    else:
                        load_checkpoint = dependency_source.checkpoint
                train_args = _build_supervisor_args(
                    unit,
                    mode="train",
                    data_root=data_root,
                    device=device,
                    best_path=None,
                    load_path=load_checkpoint,
                    smoke_test=smoke_test,
                    max_train_batches=max_train_batches,
                    max_eval_batches=max_eval_batches,
                )
                train_result = train_call(train_args)
                selected_checkpoint, selection = _select_trained_checkpoint(
                    train_args, train_result, unit.evaluation_phase
                )
                trained_units += 1
            else:
                raise ValueError(f"Unknown plan action: {unit.action}")

            source_for_selected_checkpoint = selected_checkpoint.resolve()
            selected_hash = _copy_checkpoint(selected_checkpoint, checkpoint_path)
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "trained",
                "dataset": unit.dataset,
                "variant": unit.variant,
                "seed": unit.seed,
                "action": unit.action,
                "training_fingerprint": desired_training_fingerprint,
                "train_protocol": protocol_name,
                "training_protocol_fingerprint": training_protocol_fingerprint,
                "validation_protocol_fingerprint": training_protocol_fingerprint,
                "evaluation_protocol": PROTOCOL["protocol_id"],
                "evaluation_protocol_fingerprint": _protocol_fingerprint(),
                "source_checkpoint": None if dependency_source is None else {
                    "path": str(dependency_source.checkpoint),
                    "sha256": dependency_source.checkpoint_sha256,
                },
                "checkpoint": {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": selected_hash,
                    "selected_from": str(source_for_selected_checkpoint),
                    "selection": selection,
                    "phase": unit.evaluation_phase,
                },
                "parameter_counts": (
                    train_result.get("parameter_counts")
                    if isinstance(train_result, Mapping) else None
                ),
            }
            _atomic_json(record_path, record)
            old_record = record
        else:
            record = dict(old_record)

        context = context_cache[unit.dataset]
        if _prediction_is_valid(record, prediction_path, context):
            skipped_units += 1
            unit_record_paths.append(record_path)
            continue

        # Make interrupted re-evaluation unambiguously resumable even when a
        # formerly complete prediction has disappeared or failed its hash.
        if record.get("status") == "complete":
            record = dict(record)
            record["status"] = "trained"
            for stale_key in (
                "metrics", "prediction", "evaluation_context",
                "evaluation_fingerprint",
            ):
                record.pop(stale_key, None)
            _atomic_json(record_path, record)

        eval_args = _build_supervisor_args(
            unit,
            mode="test",
            data_root=data_root,
            device=device,
            best_path=checkpoint_path,
            load_path=None,
            smoke_test=smoke_test,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        )
        evaluation_result = evaluate_call(eval_args)
        captured = _normalize_captured_artifacts(evaluation_result)
        context = _ensure_evaluation_context(
            output_root, unit.dataset, dataset_cache[unit.dataset], captured,
            allow_partial=smoke_test,
        )
        context_cache[unit.dataset] = context
        active_parameters = _extract_active_parameters(evaluation_result, train_result, record)
        metrics = _assert_partition_and_summarize(
            captured["prediction"], context, unit, active_parameters=active_parameters
        )
        _atomic_npz(
            prediction_path,
            prediction=np.asarray(captured["prediction"], dtype=np.float32),
            evaluation_context_sha256=np.asarray(context["sha256"]),
        )
        prediction_sha256 = file_digest(prediction_path)
        for row in metrics:
            row.update(
                {
                    "train_protocol": protocol_name,
                    "evaluation_protocol": PROTOCOL["protocol_id"],
                    "evaluation_protocol_fingerprint": _protocol_fingerprint(),
                    "evaluation_fingerprint": context["fingerprint"],
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": record["checkpoint"]["sha256"],
                    "prediction": str(prediction_path.resolve()),
                    "prediction_sha256": prediction_sha256,
                }
            )
        record.update(
            {
                "status": "complete",
                "evaluation_protocol": PROTOCOL["protocol_id"],
                "evaluation_protocol_fingerprint": _protocol_fingerprint(),
                "evaluation_fingerprint": context["fingerprint"],
                "evaluation_context": {
                    "path": str(context["path"]),
                    "sha256": context["sha256"],
                },
                "prediction": {
                    "path": str(prediction_path.resolve()),
                    "sha256": prediction_sha256,
                    "evaluation_context_sha256": context["sha256"],
                },
                "parameter_counts": (
                    evaluation_result.get("parameter_counts")
                    if isinstance(evaluation_result, Mapping)
                    else record.get("parameter_counts")
                ),
                "metrics": metrics,
            }
        )
        _atomic_json(record_path, record)
        _write_dataset_manifest(
            output_root, unit.dataset, dataset_cache[unit.dataset], manifest, context
        )
        evaluated_units += 1
        unit_record_paths.append(record_path)

    regenerate_metric_csvs(output_root)
    return ExecutionResult(
        output_dir=output_root,
        unit_records=tuple(unit_record_paths),
        trained_units=trained_units,
        evaluated_units=evaluated_units,
        skipped_units=skipped_units,
    )


def _metric_sort_key(row: Mapping[str, object]) -> tuple[int, int, int, int]:
    dataset_order = {spec.name: index for index, spec in enumerate(DATASET_SPECS)}
    variant_order = {spec.name: index for index, spec in enumerate(VARIANT_SPECS)}
    flow_order = {name: index for index, name in enumerate((*FLOW_NAMES, "mean"))}
    return (
        dataset_order.get(str(row.get("dataset")), len(dataset_order)),
        variant_order.get(str(row.get("variant")), len(variant_order)),
        int(row.get("seed", 0)),
        flow_order.get(str(row.get("flow")), len(flow_order)),
    )


def _summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["variant"]), str(row["flow"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (dataset, variant, flow), group in groups.items():
        summary = {
            "dataset": dataset,
            "variant": variant,
            "flow": flow,
            "seeds": len({int(row["seed"]) for row in group}),
            "active_parameters": group[0].get("active_parameters"),
            "evaluation_protocol": group[0].get("evaluation_protocol"),
            "evaluation_protocol_fingerprint": group[0].get("evaluation_protocol_fingerprint"),
            "evaluation_fingerprint": group[0].get("evaluation_fingerprint"),
        }
        for metric in SUMMARY_METRICS:
            values = [
                float(row[metric]) for row in group
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_std"] = float(np.std(values)) if values else None
        result.append(summary)
    flow_order = {name: index for index, name in enumerate((*FLOW_NAMES, "mean"))}
    dataset_order = {spec.name: index for index, spec in enumerate(DATASET_SPECS)}
    variant_order = {spec.name: index for index, spec in enumerate(VARIANT_SPECS)}
    return sorted(
        result,
        key=lambda row: (
            dataset_order.get(row["dataset"], len(dataset_order)),
            variant_order.get(row["variant"], len(variant_order)),
            flow_order.get(row["flow"], len(flow_order)),
        ),
    )


def regenerate_metric_csvs(
    output_dir: str | Path,
    *,
    unit_record_paths: Iterable[str | Path] | None = None,
) -> tuple[Path, Path]:
    """Regenerate all CSVs exclusively from complete authoritative unit JSON."""
    output_root = Path(output_dir).resolve()
    paths = (
        tuple(Path(path) for path in unit_record_paths)
        if unit_record_paths is not None
        else tuple(output_root.glob("*/units/*/seed_*.json"))
    )
    records = []
    for path in paths:
        record = _read_json(path)
        if record is None or record.get("status") != "complete":
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, list):
            raise ValueError(f"Complete unit record has no metric rows: {path}")
        records.append(record)

    protocol_fingerprints = {
        record.get("evaluation_protocol_fingerprint")
        for record in records
        if record.get("evaluation_protocol_fingerprint") is not None
    }
    if len(protocol_fingerprints) > 1:
        raise ValueError("Cannot aggregate mixed evaluation fingerprints")
    by_dataset: dict[str, set] = {}
    for record in records:
        by_dataset.setdefault(str(record.get("dataset")), set()).add(
            record.get("evaluation_fingerprint")
        )
    if any(len(fingerprints) > 1 for fingerprints in by_dataset.values()):
        raise ValueError("Cannot aggregate mixed evaluation fingerprints within a dataset")

    rows = []
    for record in records:
        for raw_row in record["metrics"]:
            if not isinstance(raw_row, Mapping):
                raise ValueError("Metric rows in unit JSON must be objects")
            row = dict(raw_row)
            row.setdefault("evaluation_protocol_fingerprint", record.get("evaluation_protocol_fingerprint"))
            row.setdefault("evaluation_fingerprint", record.get("evaluation_fingerprint"))
            rows.append(row)
    rows.sort(key=_metric_sort_key)
    summaries = _summary_rows(rows)
    summary_fields = (
        "dataset", "variant", "flow", "seeds", "active_parameters",
        "evaluation_protocol", "evaluation_protocol_fingerprint", "evaluation_fingerprint",
        *(field for metric in SUMMARY_METRICS for field in (f"{metric}_mean", f"{metric}_std")),
    )

    combined_per_seed = output_root / "combined_per_seed_metrics.csv"
    combined_summary = output_root / "combined_summary_metrics.csv"
    _atomic_csv(combined_per_seed, PER_SEED_FIELDS, rows)
    _atomic_csv(combined_summary, summary_fields, summaries)
    for dataset in {str(row["dataset"]) for row in rows}:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_summaries = [row for row in summaries if row["dataset"] == dataset]
        _atomic_csv(output_root / dataset / "per_seed_metrics.csv", PER_SEED_FIELDS, dataset_rows)
        _atomic_csv(output_root / dataset / "summary_metrics.csv", summary_fields, dataset_summaries)
    return combined_per_seed, combined_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected paper-result evaluation safely.")
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "corrected_valid_target_p90_results"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--python-environment", default="metadata")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args(argv)


def run_corrected_results(
    *,
    source_root: str | Path = ROOT,
    data_dir: str | Path | None = None,
    output_dir: str | Path = ROOT / "corrected_valid_target_p90_results",
    device: str = "cpu",
    seeds: Sequence[int] = (1, 2, 3),
    event_loss_weight: float = 0.25,
    python_environment: str = "metadata",
    smoke_test: bool = False,
    restart: bool = False,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    train_supervisor: Callable[[Namespace], Mapping[str, object] | None] | None = None,
    evaluate_supervisor: Callable[[Namespace], Mapping[str, object] | None] | None = None,
) -> ExecutionResult:
    selected_seeds = (1,) if smoke_test else tuple(seeds)
    effective_output = _smoke_output_dir(output_dir, smoke_test)
    if restart:
        _safe_restart_root(effective_output)
    planned = dry_run(
        source_root=source_root,
        data_dir=data_dir,
        output_dir=output_dir,
        seeds=selected_seeds,
        smoke_test=smoke_test,
        event_loss_weight=event_loss_weight,
        python_environment=python_environment,
        device=device,
    )
    return execute_plan(
        planned.plan,
        source_checkpoints=planned.source_checkpoints,
        manifest=planned.manifest,
        data_dir=_resolve_data_root(data_dir),
        output_dir=output_dir,
        device=device,
        train_supervisor=train_supervisor,
        evaluate_supervisor=evaluate_supervisor,
        restart=False,
        smoke_test=smoke_test,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device or "cpu"
    if args.dry_run:
        result = dry_run(
            source_root=args.source_root,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seeds=args.seeds,
            smoke_test=args.smoke_test,
            event_loss_weight=args.event_loss_weight,
            python_environment=args.python_environment,
            device=device,
        )
        actions = [unit.action for unit in result.plan]
        print(
            "dry-run "
            f"datasets={len(DATASET_SPECS)} "
            f"reusable_checkpoints={actions.count('reuse_eval')} "
            f"train_units={actions.count('train_eval')} "
            f"evaluation_units={len(result.plan)} "
            f"output={_smoke_output_dir(args.output_dir, args.smoke_test)}"
        )
        return 0

    result = run_corrected_results(
        source_root=args.source_root,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        seeds=args.seeds,
        event_loss_weight=args.event_loss_weight,
        python_environment=args.python_environment,
        smoke_test=args.smoke_test,
        restart=args.restart,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    print(
        "corrected-run "
        f"output={result.output_dir} trained={result.trained_units} "
        f"evaluated={result.evaluated_units} resumed={result.skipped_units}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_corrected_paper_results import (
    DATASET_SPECS,
    MODEL_DEFAULTS,
    VARIANT_SPECS,
    _build_supervisor_args,
    build_execution_plan,
    build_run_manifest,
    dry_run,
    execute_plan,
    parse_args,
    regenerate_metric_csvs,
    resolve_source_checkpoint,
    validate_resume_manifest,
)


def load_tests(loader, tests, pattern):
    no_path_tests = (
        test_fixed_dataset_variant_order_and_action_counts,
        test_manifest_signature_is_stable_and_smoke_is_distinct,
        test_manifest_rejects_empty_duplicate_or_nonpositive_seeds,
        test_default_manifest_hashes_all_scientific_runtime_files,
    )
    for test in no_path_tests:
        tests.addTest(unittest.FunctionTestCase(test))
    for test in (
        test_resolves_root_and_nested_source_checkpoint_layouts,
        test_source_resolution_rejects_incomplete_flow_rows,
        test_resume_manifest_fails_closed_for_missing_or_mismatched_manifest,
        test_resume_manifest_rejects_tampered_payload_with_stale_fingerprint,
        test_dry_run_validates_all_units_without_creating_output,
    ):
        def with_temporary_path(test=test):
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))

        tests.addTest(unittest.FunctionTestCase(with_temporary_path))
    return tests


def _write_source_rows(path: Path, dataset: str, variant: str, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = path.parent / f"{dataset}_{variant}_{seed}.pth"
    checkpoint.write_bytes(b"checkpoint")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "variant", "seed", "flow", "checkpoint"))
        writer.writeheader()
        for flow in ("inflow", "outflow", "mean"):
            writer.writerow(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "flow": flow,
                    "checkpoint": str(checkpoint),
                }
            )
    return checkpoint


def _write_all_reusable_sources(root: Path, seed: int = 1) -> None:
    for dataset in DATASET_SPECS:
        directory = root if dataset.name == "NYCTaxi" else root / "final_all_dataset_ablations" / dataset.name
        for variant in (VARIANT_SPECS[0], VARIANT_SPECS[2]):
            source_dir = variant.source_root_dir if dataset.name == "NYCTaxi" else variant.source_study
            _write_source_rows(
                directory / source_dir / variant.source_metrics_filename,
                dataset.name,
                variant.source_variant,
                seed,
            )


def _write_dataset_inputs(root: Path) -> None:
    for dataset in DATASET_SPECS:
        dataset_dir = root / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("train.npz", "val.npz", "test.npz", "adj_mx.npz"):
            (dataset_dir / filename).write_bytes(f"{dataset.name}:{filename}".encode("utf-8"))


def _write_synthetic_dataset(root: Path, dataset: str = "NYCTaxi") -> None:
    """Write a tiny internally consistent v2 dataset for execution tests."""
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_y = np.arange(1, 21, dtype=np.float32).reshape(10, 1, 1, 2)
    # q90 is [17.2, 18.2].  This supplies a normal point and an event per flow.
    test_y = np.array(
        [
            [[[[6.0, 7.0]]]],
            [[[[19.0, 20.0]]]],
        ],
        dtype=np.float32,
    ).reshape(2, 1, 1, 2)
    val_y = test_y.copy()
    thresholds = np.percentile(train_y, 90.0, axis=0)
    for split, y in (("train", train_y), ("val", val_y), ("test", test_y)):
        x = np.zeros_like(y)
        np.savez(
            dataset_dir / f"{split}.npz",
            x=x,
            y=y,
            evs_90=(y > thresholds).astype(np.float32),
        )
    np.savez(dataset_dir / "adj_mx.npz", adj_mx=np.eye(1, dtype=np.float32))


def _one_seed_sources(root: Path) -> tuple:
    _write_all_reusable_sources(root, seed=1)
    return tuple(
        resolve_source_checkpoint(root, dataset.name, variant.name, 1)
        for dataset in DATASET_SPECS
        for variant in (VARIANT_SPECS[0], VARIANT_SPECS[2])
    )


def _manifest_for_execution(source_root: Path, data_root: Path) -> tuple[dict, tuple]:
    sources = _one_seed_sources(source_root)
    # Execution tests use one dataset, but the canonical manifest deliberately
    # retains the complete fixed paper scope.
    _write_dataset_inputs(data_root)
    _write_synthetic_dataset(data_root, "NYCTaxi")
    manifest = build_run_manifest(
        seeds=(1,),
        source_checkpoints=sources,
        data_files=(
            data_root / dataset.name / filename
            for dataset in DATASET_SPECS
            for filename in ("train.npz", "val.npz", "test.npz", "adj_mx.npz")
        ),
    )
    return manifest, sources


def _captured_evaluation(args: Namespace, calls: list[Namespace]):
    calls.append(args)
    target = np.array(
        [
            [[[[6.0, 7.0]]]],
            [[[[19.0, 20.0]]]],
        ],
        dtype=np.float32,
    ).reshape(2, 1, 1, 2)
    prediction = target + np.array([1.0, 2.0], dtype=np.float32)
    event = np.array(
        [
            [[[[0.0, 0.0]]]],
            [[[[1.0, 1.0]]]],
        ],
        dtype=np.float32,
    ).reshape(2, 1, 1, 2)
    return {
        "test_artifacts": {
            "prediction": prediction,
            "target": target,
            "event": event,
            "valid": target > 5.0,
        },
        "parameter_counts": {"active": 123},
        # A deliberately bogus legacy-like metric must never reach fresh rows.
        "test_details": [{"valid_mae": 999.0}, {"valid_mae": 999.0}],
    }


def test_fixed_dataset_variant_order_and_action_counts():
    assert [spec.name for spec in DATASET_SPECS] == ["NYCTaxi", "NYCBike1", "NYCBike2", "BJTaxi"]
    assert [spec.name for spec in VARIANT_SPECS] == [
        "base_mae",
        "event_weighted_base",
        "frozen_residual_mae",
        "joint_event_weighted_residual",
        "frozen_event_weighted_residual",
    ]
    assert [spec.event_loss_weight for spec in VARIANT_SPECS] == [0.0, 0.25, 0.0, 0.25, 0.25]
    assert {spec.bias_param_scope for spec in VARIANT_SPECS} == {"head_only"}

    plan = build_execution_plan(seeds=(1, 2, 3))
    actions = [unit.action for unit in plan]
    assert actions.count("reuse_eval") == 24
    assert actions.count("train_eval") == 36

    joint = next(unit for unit in plan if unit.variant == "joint_event_weighted_residual")
    frozen = next(unit for unit in plan if unit.variant == "frozen_event_weighted_residual")
    assert (joint.start_phase, joint.stop_after_phase, joint.evaluation_phase) == ("bias", "bias", "bias")
    assert (frozen.start_phase, frozen.stop_after_phase, frozen.evaluation_phase) == ("pred_2", "pred_2", "pred_2")
    assert all("three_stage" not in str(spec) for spec in VARIANT_SPECS)


def test_resolves_root_and_nested_source_checkpoint_layouts(tmp_path):
    source_root = tmp_path / "sources"
    _write_all_reusable_sources(source_root)

    root_level = resolve_source_checkpoint(source_root, "NYCTaxi", "base_mae", 1)
    nested = resolve_source_checkpoint(source_root, "NYCBike1", "frozen_residual_mae", 1)

    assert root_level.checkpoint.is_file()
    assert nested.checkpoint.is_file()
    assert root_level.csv_path == source_root / "paper_rescue_ablation_results" / "shared_phase1_metrics.csv"
    assert nested.csv_path == source_root / "final_all_dataset_ablations" / "NYCBike1" / "residual_schedule" / "per_seed_metrics.csv"
    assert len(root_level.checkpoint_sha256) == 64


def test_source_resolution_rejects_incomplete_flow_rows(tmp_path):
    csv_path = tmp_path / "paper_rescue_ablation_results" / "shared_phase1_metrics.csv"
    checkpoint = _write_source_rows(csv_path, "NYCTaxi", "shared_phase1", 1)
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:2]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "variant", "seed", "flow", "checkpoint"))
        writer.writeheader()
        writer.writerows(rows)

    with unittest.TestCase().assertRaisesRegex(ValueError, "inflow, outflow, mean"):
        resolve_source_checkpoint(tmp_path, "NYCTaxi", "base_mae", 1)
    assert checkpoint.is_file()


def test_manifest_signature_is_stable_and_smoke_is_distinct():
    full = build_run_manifest(seeds=(1, 2, 3), smoke_test=False, device="cpu")
    same_full = build_run_manifest(seeds=(1, 2, 3), smoke_test=False, device="cpu")
    gpu_full = build_run_manifest(seeds=(1, 2, 3), smoke_test=False, device="cuda")
    smoke = build_run_manifest(seeds=(1,), smoke_test=True)

    assert full["fingerprint"] == same_full["fingerprint"]
    assert full["fingerprint"] != gpu_full["fingerprint"]
    assert full["fingerprint"] != smoke["fingerprint"]
    assert full["device"] == "cpu"
    assert full["protocol"]["protocol_id"] == "train_all_node_flow_p90_valid_v2"


def test_manifest_rejects_empty_duplicate_or_nonpositive_seeds():
    case = unittest.TestCase()
    for seeds in ((), (1, 1), (0,), (-1,)):
        with case.assertRaisesRegex(ValueError, "seeds"):
            build_run_manifest(seeds=seeds)


def test_default_manifest_hashes_all_scientific_runtime_files():
    manifest = build_run_manifest(seeds=(1,))
    names = {Path(entry["path"]).name for entry in manifest["code_hashes"]}
    assert names == {
        "run_corrected_paper_results.py",
        "main.py",
        "models.py",
        "layers.py",
        "trainer.py",
        "dataloader.py",
        "metrics.py",
        "event_masks.py",
        "utils.py",
        "logger.py",
    }


def test_resume_manifest_fails_closed_for_missing_or_mismatched_manifest(tmp_path):
    expected = build_run_manifest(seeds=(1,), smoke_test=False)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "partial.json").write_text("{}", encoding="utf-8")
    with unittest.TestCase().assertRaisesRegex(ValueError, "run_manifest.json"):
        validate_resume_manifest(output_dir, expected)

    incompatible = build_run_manifest(seeds=(1,), smoke_test=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(incompatible), encoding="utf-8")
    with unittest.TestCase().assertRaisesRegex(ValueError, "incompatible"):
        validate_resume_manifest(output_dir, expected)


def test_resume_manifest_rejects_tampered_payload_with_stale_fingerprint(tmp_path):
    expected = build_run_manifest(seeds=(1,), smoke_test=False)
    stored = dict(expected)
    stored["scaler_policy"] = "train_val"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text(
        json.dumps(stored), encoding="utf-8"
    )

    with unittest.TestCase().assertRaisesRegex(ValueError, "fingerprint"):
        validate_resume_manifest(output_dir, expected)


def test_dry_run_validates_all_units_without_creating_output(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    _write_all_reusable_sources(source_root)
    _write_dataset_inputs(data_root)
    output_dir = tmp_path / "corrected_results"

    result = dry_run(
        source_root=source_root,
        data_dir=data_root,
        output_dir=output_dir,
        seeds=(1,),
    )

    assert len(result.plan) == 20
    assert len(result.source_checkpoints) == 8
    assert len(result.manifest["data_hashes"]) == 16
    assert parse_args(["--data-dir", str(data_root)]).data_dir == str(data_root)
    assert not output_dir.exists()


def test_execute_plan_uses_separate_exact_train_and_captured_eval_args(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    output_root = tmp_path / "corrected"
    manifest, sources = _manifest_for_execution(source_root, data_root)
    unit = next(
        unit
        for unit in build_execution_plan((1,))
        if unit.dataset == "NYCTaxi" and unit.variant == "event_weighted_base"
    )
    train_calls = []
    eval_calls = []

    def train_supervisor(args):
        train_calls.append(args)
        checkpoint = tmp_path / "trained" / "best_model_pred.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fresh event weighted checkpoint")
        return {"selected_checkpoint": str(checkpoint), "parameter_counts": {"active": 123}}

    def eval_supervisor(args):
        return _captured_evaluation(args, eval_calls)

    execute_plan(
        (unit,),
        source_checkpoints=sources,
        manifest=manifest,
        data_dir=data_root,
        output_dir=output_root,
        device="cpu",
        train_supervisor=train_supervisor,
        evaluate_supervisor=eval_supervisor,
    )

    assert len(train_calls) == len(eval_calls) == 1
    train_args = train_calls[0]
    assert train_args.mode == "train"
    assert train_args.best_path is None and train_args.load_path is None
    assert train_args.data_dir == str(data_root.resolve())
    assert train_args.graph_file == str((data_root / "NYCTaxi" / "adj_mx.npz").resolve())
    assert train_args.scaler_fit == "train"
    assert train_args.event_mask_protocol == "train_all_node_flow_p90_valid_v2"
    assert train_args.event_label_source == "file_verified"
    assert train_args.device == "cpu"
    assert train_args.ablation_mode == "event_weighted_base"
    assert (train_args.start_phase, train_args.stop_after_phase, train_args.evaluation_phase) == (
        "pred", "pred", "pred"
    )
    assert train_args.event_loss_weight == 0.25
    assert train_args.bias_param_scope == "head_only"
    assert train_args.capture_predictions is False
    assert train_args.raise_exceptions is True
    assert train_args.event_percentile is None

    eval_args = eval_calls[0]
    local_checkpoint = output_root / "NYCTaxi" / "checkpoints" / unit.variant / "seed_1.pth"
    assert eval_args.mode == "test"
    assert eval_args.best_path == str(local_checkpoint.resolve())
    assert eval_args.load_path is None
    assert eval_args.capture_predictions is True
    assert eval_args.evaluation_phase == "pred"
    assert local_checkpoint.read_bytes() == b"fresh event weighted checkpoint"

    unit_json = output_root / "NYCTaxi" / "units" / unit.variant / "seed_1.json"
    record = json.loads(unit_json.read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["metrics"][0]["mae"] != 999.0
    prediction_npz = np.load(output_root / "NYCTaxi" / "predictions" / unit.variant / "seed_1.npz")
    assert set(prediction_npz.files) == {"prediction", "evaluation_context_sha256"}


def test_reuse_eval_never_trains_and_copies_the_source_checkpoint(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    output_root = tmp_path / "corrected"
    manifest, sources = _manifest_for_execution(source_root, data_root)
    unit = next(
        unit for unit in build_execution_plan((1,))
        if unit.dataset == "NYCTaxi" and unit.variant == "base_mae"
    )
    train_calls = []
    eval_calls = []

    execute_plan(
        (unit,),
        source_checkpoints=sources,
        manifest=manifest,
        data_dir=data_root,
        output_dir=output_root,
        device="cpu",
        train_supervisor=lambda args: train_calls.append(args),
        evaluate_supervisor=lambda args: _captured_evaluation(args, eval_calls),
    )

    assert train_calls == []
    source = next(source for source in sources if source.dataset == "NYCTaxi" and source.variant == "base_mae")
    local = output_root / "NYCTaxi" / "checkpoints" / "base_mae" / "seed_1.pth"
    assert local.read_bytes() == source.checkpoint.read_bytes()
    assert eval_calls[0].event_loss_weight == 0.0


def test_trained_unit_with_failed_or_hash_invalid_evaluation_resumes_eval_only(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    output_root = tmp_path / "corrected"
    manifest, sources = _manifest_for_execution(source_root, data_root)
    unit = next(
        unit for unit in build_execution_plan((1,))
        if unit.dataset == "NYCTaxi" and unit.variant == "event_weighted_base"
    )
    train_calls = []
    eval_calls = []

    def train_supervisor(args):
        train_calls.append(args)
        checkpoint = tmp_path / "trained.pth"
        checkpoint.write_bytes(b"trained")
        return {"selected_checkpoint": str(checkpoint)}

    def failing_eval(args):
        eval_calls.append(args)
        raise RuntimeError("intentional evaluation interruption")

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "interruption"):
        execute_plan(
            (unit,), source_checkpoints=sources, manifest=manifest,
            data_dir=data_root, output_dir=output_root, device="cpu",
            train_supervisor=train_supervisor, evaluate_supervisor=failing_eval,
        )
    record_path = output_root / "NYCTaxi" / "units" / unit.variant / "seed_1.json"
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "trained"

    execute_plan(
        (unit,), source_checkpoints=sources, manifest=manifest,
        data_dir=data_root, output_dir=output_root, device="cpu",
        train_supervisor=train_supervisor,
        evaluate_supervisor=lambda args: _captured_evaluation(args, eval_calls),
    )
    assert len(train_calls) == 1
    assert len(eval_calls) == 2

    prediction_path = output_root / "NYCTaxi" / "predictions" / unit.variant / "seed_1.npz"
    prediction_path.write_bytes(prediction_path.read_bytes() + b"tamper")
    execute_plan(
        (unit,), source_checkpoints=sources, manifest=manifest,
        data_dir=data_root, output_dir=output_root, device="cpu",
        train_supervisor=train_supervisor,
        evaluate_supervisor=lambda args: _captured_evaluation(args, eval_calls),
    )
    assert len(train_calls) == 1
    assert len(eval_calls) == 3


def test_incompatible_event_dependent_training_fingerprint_forces_retrain(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    output_root = tmp_path / "corrected"
    manifest, sources = _manifest_for_execution(source_root, data_root)
    unit = next(
        unit for unit in build_execution_plan((1,))
        if unit.dataset == "NYCTaxi" and unit.variant == "event_weighted_base"
    )
    train_calls = []
    eval_calls = []

    def train_supervisor(args):
        train_calls.append(args)
        checkpoint = tmp_path / f"trained-{len(train_calls)}.pth"
        checkpoint.write_bytes(f"trained-{len(train_calls)}".encode())
        return {"selected_checkpoint": str(checkpoint)}

    kwargs = dict(
        source_checkpoints=sources, manifest=manifest, data_dir=data_root,
        output_dir=output_root, device="cpu", train_supervisor=train_supervisor,
        evaluate_supervisor=lambda args: _captured_evaluation(args, eval_calls),
    )
    execute_plan((unit,), **kwargs)
    record_path = output_root / "NYCTaxi" / "units" / unit.variant / "seed_1.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["training_fingerprint"] = "legacy-incompatible-fingerprint"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    execute_plan((unit,), **kwargs)
    assert len(train_calls) == 2
    assert len(eval_calls) == 2


def test_phase_checkpoint_selection_and_base_fallback_are_copied_locally(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    manifest, sources = _manifest_for_execution(source_root, data_root)
    joint = next(
        unit for unit in build_execution_plan((1,))
        if unit.dataset == "NYCTaxi" and unit.variant == "joint_event_weighted_residual"
    )

    def run_case(output_root, create_phase_checkpoint):
        train_args_seen = []

        def train_supervisor(args):
            train_args_seen.append(args)
            args.log_dir = str(tmp_path / output_root.name / "log")
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)
            if create_phase_checkpoint:
                (Path(args.log_dir) / "best_model_bias.pth").write_bytes(b"bias checkpoint")
            return {}

        execute_plan(
            (joint,), source_checkpoints=sources, manifest=manifest,
            data_dir=data_root, output_dir=output_root, device="cpu",
            train_supervisor=train_supervisor,
            evaluate_supervisor=lambda args: _captured_evaluation(args, []),
        )
        return train_args_seen[0], output_root / "NYCTaxi" / "checkpoints" / joint.variant / "seed_1.pth"

    phase_args, phase_copy = run_case(tmp_path / "with-phase", True)
    base_source = next(
        source for source in sources
        if source.dataset == "NYCTaxi" and source.variant == "base_mae"
    )
    assert Path(phase_args.load_path).read_bytes() == base_source.checkpoint.read_bytes()
    assert phase_copy.read_bytes() == b"bias checkpoint"

    fallback_args, fallback_copy = run_case(tmp_path / "fallback", False)
    assert fallback_copy.read_bytes() == Path(fallback_args.load_path).read_bytes()


def test_aggregation_rejects_mixed_evaluation_fingerprints(tmp_path):
    output_root = tmp_path / "corrected"
    records = []
    for seed, fingerprint in ((1, "eval-a"), (2, "eval-b")):
        path = output_root / "NYCTaxi" / "units" / "base_mae" / f"seed_{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "complete",
            "dataset": "NYCTaxi",
            "variant": "base_mae",
            "seed": seed,
            "evaluation_fingerprint": fingerprint,
            "metrics": [{"dataset": "NYCTaxi", "variant": "base_mae", "seed": seed,
                         "flow": "inflow", "mae": 1.0}],
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        records.append(path)

    with unittest.TestCase().assertRaisesRegex(ValueError, "mixed evaluation fingerprints"):
        regenerate_metric_csvs(output_root, unit_record_paths=records)


def test_smoke_dry_run_uses_separate_suffix_and_never_writes(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    _write_all_reusable_sources(source_root)
    _write_dataset_inputs(data_root)
    requested = tmp_path / "paper-results"

    result = dry_run(
        source_root=source_root,
        data_dir=data_root,
        output_dir=requested,
        seeds=(1, 2, 3),
        smoke_test=True,
        device="cpu",
    )

    assert result.manifest["mode"] == "smoke"
    assert result.manifest["seeds"] == [1]
    assert not requested.exists()
    assert not requested.with_name("paper-results_smoke").exists()


def test_all_dataset_configs_receive_complete_safe_model_args(tmp_path):
    data_root = tmp_path / "data"
    for dataset in DATASET_SPECS:
        (data_root / dataset.name).mkdir(parents=True)
        (data_root / dataset.name / "adj_mx.npz").write_bytes(b"graph")
        unit = next(
            unit for unit in build_execution_plan((1,))
            if unit.dataset == dataset.name and unit.variant == "base_mae"
        )
        args = _build_supervisor_args(
            unit,
            mode="test",
            data_root=data_root,
            device="cpu",
            best_path=tmp_path / "model.pth",
            load_path=None,
            smoke_test=False,
            max_train_batches=None,
            max_eval_batches=None,
        )
        for key, expected in MODEL_DEFAULTS.items():
            assert hasattr(args, key), f"{dataset.name} missing {key}"
        assert args.scaler_fit == "train"
        assert args.event_loss_weight == 0.0


def test_batch_limits_are_smoke_only_and_forced_to_one(tmp_path):
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    output_root = tmp_path / "corrected"
    manifest, sources = _manifest_for_execution(source_root, data_root)
    unit = next(
        unit for unit in build_execution_plan((1,))
        if unit.dataset == "NYCTaxi" and unit.variant == "base_mae"
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "only in smoke"):
        execute_plan(
            (unit,), source_checkpoints=sources, manifest=manifest,
            data_dir=data_root, output_dir=output_root, device="cpu",
            max_eval_batches=1,
            train_supervisor=lambda args: None,
            evaluate_supervisor=lambda args: _captured_evaluation(args, []),
        )

    smoke_manifest = build_run_manifest(
        seeds=(1,), smoke_test=True, device="cpu",
        source_checkpoints=sources,
        data_files=(
            data_root / dataset.name / filename
            for dataset in DATASET_SPECS
            for filename in ("train.npz", "val.npz", "test.npz", "adj_mx.npz")
        ),
    )
    eval_calls = []
    result = execute_plan(
        (unit,), source_checkpoints=sources, manifest=smoke_manifest,
        data_dir=data_root, output_dir=tmp_path / "smoke-output", device="cpu",
        smoke_test=True,
        train_supervisor=lambda args: None,
        evaluate_supervisor=lambda args: _captured_evaluation(args, eval_calls),
    )
    assert result.output_dir.name == "smoke-output_smoke"
    assert eval_calls[0].max_train_batches == eval_calls[0].max_eval_batches == 1

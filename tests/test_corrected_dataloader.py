from pathlib import Path

import numpy as np
import pytest
import torch

from lib.dataloader import get_dataloader


def _write_split(path: Path, y: np.ndarray, evs_90: np.ndarray) -> None:
    x = np.arange(y.size, dtype=np.float32).reshape(y.shape) + 1.0
    np.savez(path, x=x, y=y.astype(np.float32), evs_90=evs_90.astype(np.float32))


def _write_dataset(tmp_path: Path, *, mismatch: bool = False) -> tuple[Path, str, dict]:
    dataset = "fixture"
    dataset_dir = tmp_path / dataset
    dataset_dir.mkdir()
    y_train = np.array(
        [
            [[[1.0, 1.0]]],
            [[[2.0, 2.0]]],
            [[[3.0, 3.0]]],
            [[[4.0, 4.0]]],
            [[[5.0, 5.0]]],
            [[[6.0, 6.0]]],
            [[[7.0, 7.0]]],
            [[[8.0, 8.0]]],
            [[[9.0, 9.0]]],
            [[[10.0, 10.0]]],
        ]
    )
    y_val = np.array([[[[5.0, 9.0]]], [[[6.0, 10.0]]]])
    y_test = np.array([[[[4.0, 11.0]]], [[[7.0, 8.0]]]])
    thresholds = np.percentile(y_train, 90.0, axis=0)
    expected = {}
    for category, values in (("train", y_train), ("val", y_val), ("test", y_test)):
        raw = values > thresholds
        event = (values > 5.0) & raw
        if mismatch and category == "val":
            raw = raw.copy()
            raw[0, 0, 0, 0] = ~raw[0, 0, 0, 0]
        _write_split(dataset_dir / f"{category}.npz", values, raw)
        expected[category] = event.astype(np.float32)
    return tmp_path, dataset, expected


def test_v2_file_verified_loader_uses_corrected_masks_and_cpu(tmp_path):
    data_dir, dataset, expected = _write_dataset(tmp_path)

    loaders = get_dataloader(
        data_dir=str(data_dir),
        dataset=dataset,
        batch_size=32,
        test_batch_size=32,
        event_mask_protocol="train_all_node_flow_p90_valid_v2",
        event_label_source="file_verified",
        device="cpu",
    )

    for category in ("train", "val", "test"):
        _, _, event, bias = loaders[category].dataset.tensors
        assert event.device.type == "cpu"
        assert bias.device.type == "cpu"
        np.testing.assert_array_equal(event.numpy(), expected[category])
        np.testing.assert_array_equal(bias.numpy(), expected[category])
    assert loaders["event_mask_protocol"] == "train_all_node_flow_p90_valid_v2"
    assert loaders["event_label_source"] == "file_verified"
    np.testing.assert_allclose(loaders["event_thresholds"], np.array([[[9.1, 9.1]]]))


def test_v2_file_verified_loader_rejects_a_saved_label_mismatch(tmp_path):
    data_dir, dataset, _ = _write_dataset(tmp_path, mismatch=True)

    with pytest.raises(ValueError, match="raw_labels do not match recomputed raw exceedance"):
        get_dataloader(
            data_dir=str(data_dir),
            dataset=dataset,
            batch_size=32,
            test_batch_size=32,
            event_mask_protocol="train_all_node_flow_p90_valid_v2",
            event_label_source="file_verified",
            device="cpu",
        )


def test_loader_explicit_cpu_device_does_not_query_cuda(monkeypatch, tmp_path):
    data_dir, dataset, _ = _write_dataset(tmp_path)

    def fail_if_queried():
        raise AssertionError("explicit CPU execution must not query CUDA availability")

    monkeypatch.setattr(torch.cuda, "is_available", fail_if_queried)
    loaders = get_dataloader(
        data_dir=str(data_dir),
        dataset=dataset,
        batch_size=32,
        test_batch_size=32,
        event_mask_protocol="train_all_node_flow_p90_valid_v2",
        event_label_source="file_verified",
        device="cpu",
    )

    assert loaders["train"].dataset.tensors[0].device.type == "cpu"


@pytest.mark.parametrize(
    ("protocol", "source"),
    [
        ("legacy_raw_p90_v1", "file_verified"),
        ("legacy_raw_p90_v1", "generated"),
        ("train_all_node_flow_p90_valid_v2", "legacy_file_unverified"),
    ],
)
def test_loader_rejects_ambiguous_protocol_and_label_source_pairs(tmp_path, protocol, source):
    data_dir, dataset, _ = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="event_label_source"):
        get_dataloader(
            data_dir=str(data_dir),
            dataset=dataset,
            batch_size=32,
            test_batch_size=32,
            event_mask_protocol=protocol,
            event_label_source=source,
            device="cpu",
        )


def test_v2_generated_source_does_not_verify_saved_labels(tmp_path):
    data_dir, dataset, expected = _write_dataset(tmp_path, mismatch=True)

    loaders = get_dataloader(
        data_dir=str(data_dir),
        dataset=dataset,
        batch_size=32,
        test_batch_size=32,
        event_mask_protocol="train_all_node_flow_p90_valid_v2",
        event_label_source="generated",
        device="cpu",
    )

    np.testing.assert_array_equal(loaders["val"].dataset.tensors[2].numpy(), expected["val"])


def test_legacy_defaults_continue_to_use_saved_labels(tmp_path):
    data_dir, dataset, _ = _write_dataset(tmp_path)
    train_file = data_dir / dataset / "train.npz"
    with np.load(train_file) as saved:
        x, y = saved["x"], saved["y"]
    raw_labels = np.zeros_like(y, dtype=np.float32)
    np.savez(train_file, x=x, y=y, evs_90=raw_labels)

    loaders = get_dataloader(
        data_dir=str(data_dir), dataset=dataset, batch_size=32, test_batch_size=32, device="cpu"
    )

    np.testing.assert_array_equal(loaders["train"].dataset.tensors[2].numpy(), raw_labels)
    assert loaders["event_mask_protocol"] == "legacy_raw_p90_v1"
    assert loaders["event_label_source"] == "legacy_file_unverified"

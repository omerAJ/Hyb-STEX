import numpy as np
import pytest

from lib.metrics import corrected_event_metrics


def test_corrected_metrics_reconstruct_valid_error_totals_with_unequal_flows():
    true = np.array(
        [
            [[[4.0, 6.0], [7.0, 8.0]]],
            [[[6.0, 9.0], [10.0, 3.0]]],
        ]
    )
    pred = np.array(
        [
            [[[3.0, 7.0], [5.0, 10.0]]],
            [[[8.0, 6.0], [13.0, 2.0]]],
        ]
    )
    event = np.array(
        [
            [[[0, 0], [1, 1]]],
            [[[0, 0], [1, 0]]],
        ],
        dtype=np.float32,
    )

    detail = corrected_event_metrics(pred, true, event, valid_min=5.0)

    assert detail["valid_count"] == 6
    assert detail["event_count"] == 3
    assert detail["normal_count"] == 3
    assert np.isclose(
        detail["valid_abs_error_sum"],
        detail["event_abs_error_sum"] + detail["normal_abs_error_sum"],
    )
    assert np.isclose(detail["valid_abs_error_sum"], 13.0)
    assert np.isclose(detail["event_signed_error"], 1.0)


def test_corrected_metrics_reports_nan_event_means_when_no_events_exist():
    true = np.array([[[[4.0, 6.0], [7.0, 5.0]]]])
    pred = np.array([[[[3.0, 8.0], [5.0, 10.0]]]])
    event = np.zeros_like(true, dtype=np.float32)

    detail = corrected_event_metrics(pred, true, event, valid_min=5.0)

    assert detail["valid_count"] == detail["event_count"] + detail["normal_count"]
    assert detail["event_count"] == 0
    assert detail["event_abs_error_sum"] == 0.0
    assert np.isnan(detail["event_mae"])
    assert np.isnan(detail["event_signed_error"])


def test_corrected_metrics_rejects_an_event_outside_the_valid_population():
    true = np.array([[[[4.0, 6.0]]]])
    pred = true.copy()
    event = np.array([[[[1.0, 0.0]]]])

    with pytest.raises(ValueError, match="subset of valid"):
        corrected_event_metrics(pred, true, event, valid_min=5.0)


@pytest.mark.parametrize("event", [np.array([[[[0.5, 1.0]]]]), np.array([[[[0, 2]]]])])
def test_corrected_metrics_rejects_nonbinary_numpy_event_labels(event):
    true = np.array([[[[6.0, 7.0]]]])

    with pytest.raises(ValueError, match="bool or exact binary 0/1"):
        corrected_event_metrics(true, true, event, valid_min=5.0)


def test_corrected_metrics_rejects_nonbinary_torch_event_labels():
    torch = pytest.importorskip("torch")
    true = torch.tensor([[[[6.0, 7.0], [8.0, 9.0]]]])
    event = torch.tensor([[[[0.0, 1.0], [0.5, 0.0]]]])

    with pytest.raises(ValueError, match="bool or exact binary 0/1"):
        corrected_event_metrics(true, true, event, valid_min=5.0)


def test_corrected_metrics_rejects_mixed_torch_devices_before_mask_conversion():
    torch = pytest.importorskip("torch")
    true = torch.tensor([[[[6.0, 7.0], [8.0, 9.0]]]])
    event = torch.zeros_like(true, device="meta")

    with pytest.raises(ValueError, match="same device"):
        corrected_event_metrics(true, true, event, valid_min=5.0)

import numpy as np
import pytest

from lib.event_masks import (
    EventMaskSpec,
    build_event_masks,
    corrected_p90_spec,
    fit_event_thresholds,
    validate_event_partition,
)


def _training_targets() -> np.ndarray:
    """Three channels whose p90 thresholds are below, equal to, and above 5."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [4.0, 4.0, 5.0],
            [4.0, 4.0, 6.0],
            [4.0, 4.0, 7.0],
            [4.0, 5.0, 10.0],
            [4.0, 5.0, 10.0],
        ]
    )


def _test_targets() -> np.ndarray:
    return np.array(
        [
            [5.0, 5.0, 10.0],
            [6.0, 6.0, 11.0],
            [4.0, 6.0, 10.0],
        ]
    )


def test_corrected_p90_masks_partition_valid_targets_at_strict_boundaries():
    spec = corrected_p90_spec()
    thresholds = fit_event_thresholds(_training_targets(), spec)
    masks = build_event_masks(_test_targets(), thresholds, spec)

    np.testing.assert_allclose(thresholds.values, [4.0, 5.0, 10.0])
    assert not np.any(masks.event & ~masks.valid)
    assert not np.any(masks.event & masks.normal)
    np.testing.assert_array_equal(masks.event | masks.normal, masks.valid)
    validate_event_partition(masks)

    # Values exactly at the valid cutoff and fitted quantile are never events.
    assert not masks.valid[0, 0]  # Y == 5, despite q90 < 5
    assert not masks.event[0, 1]  # Y == 5 == q90
    assert masks.valid[0, 2] and not masks.event[0, 2]  # Y == q90 == 10
    np.testing.assert_array_equal(
        masks.event,
        np.array(
            [
                [False, False, False],
                [True, True, True],
                [False, True, False],
            ]
        ),
    )


def test_file_verified_labels_match_recomputed_raw_exceedance():
    spec = corrected_p90_spec()
    thresholds = fit_event_thresholds(_training_targets(), spec)
    generated = build_event_masks(_test_targets(), thresholds, spec)
    verified = build_event_masks(
        _test_targets(),
        thresholds,
        spec,
        raw_labels=generated.raw_exceedance.astype(np.float32),
    )

    np.testing.assert_array_equal(verified.raw_exceedance, generated.raw_exceedance)
    np.testing.assert_array_equal(verified.event, generated.event)
    assert verified.fingerprint == generated.fingerprint


def test_file_verified_labels_reject_a_mismatch():
    spec = corrected_p90_spec()
    thresholds = fit_event_thresholds(_training_targets(), spec)
    wrong_labels = np.zeros_like(_test_targets(), dtype=bool)

    with pytest.raises(ValueError, match="raw_labels do not match recomputed raw exceedance"):
        build_event_masks(_test_targets(), thresholds, spec, raw_labels=wrong_labels)


def test_file_verified_labels_reject_non_binary_numeric_values():
    spec = corrected_p90_spec()
    thresholds = fit_event_thresholds(_training_targets(), spec)
    raw_labels = np.zeros_like(_test_targets(), dtype=np.float32)
    raw_labels[0, 0] = 2.0

    with pytest.raises(ValueError, match="raw_labels must contain only boolean or exact 0/1"):
        build_event_masks(_test_targets(), thresholds, spec, raw_labels=raw_labels)


def test_protocol_and_bundle_fingerprints_are_stable():
    spec = corrected_p90_spec()
    thresholds = fit_event_thresholds(_training_targets(), spec)
    first = build_event_masks(_test_targets(), thresholds, spec)
    second = build_event_masks(_test_targets().copy(), thresholds, spec)

    assert spec.protocol_id == "train_all_node_flow_p90_valid_v2"
    assert spec.fingerprint() == "e73cb3038733ca0a8a258f3bfaf0147ebc5c3122c53ed427a60ca71e4bfb0ad3"
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    ("protocol_id", "percentile", "valid_min", "message"),
    [
        ("another_protocol", 90.0, 5.0, "protocol_id"),
        ("train_all_node_flow_p90_valid_v2", 89.0, 5.0, "percentile"),
        ("train_all_node_flow_p90_valid_v2", 90.0, 4.0, "valid_min"),
    ],
)
def test_builders_reject_noncanonical_protocol_values(
    protocol_id, percentile, valid_min, message
):
    spec = EventMaskSpec(
        protocol_id=protocol_id,
        percentile=percentile,
        valid_min=valid_min,
    )

    with pytest.raises(ValueError, match=message):
        fit_event_thresholds(_training_targets(), spec)


def test_builders_reject_incompatible_shapes_and_threshold_protocols():
    spec = corrected_p90_spec()
    thresholds = fit_event_thresholds(_training_targets(), spec)

    with pytest.raises(ValueError, match="trailing shape"):
        build_event_masks(np.ones((3, 2)), thresholds, spec)
    with pytest.raises(ValueError, match="spec fingerprint"):
        build_event_masks(
            _test_targets(),
            type(thresholds)(thresholds.values, "different-protocol"),
            spec,
        )

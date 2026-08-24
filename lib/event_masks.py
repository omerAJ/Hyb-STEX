"""Canonical train_all_node_flow_p90_valid_v2 event-mask protocol.

Masks are built on unscaled target arrays.  The first array axis is the fit
population; every remaining axis identifies an independent target channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class EventMaskSpec:
    protocol_id: str
    percentile: float
    valid_min: float
    comparison: str = "gt"
    fit_population: str = "all_targets"

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EventThresholds:
    values: np.ndarray
    spec_fingerprint: str


@dataclass(frozen=True)
class EventMaskBundle:
    valid: np.ndarray
    raw_exceedance: np.ndarray
    event: np.ndarray
    normal: np.ndarray
    protocol_id: str
    fingerprint: str


def corrected_p90_spec() -> EventMaskSpec:
    """Return the sole protocol used for corrected paper-result masks."""
    return EventMaskSpec(
        protocol_id="train_all_node_flow_p90_valid_v2",
        percentile=90.0,
        valid_min=5.0,
    )


def fit_event_thresholds(y_train: np.ndarray, spec: EventMaskSpec) -> EventThresholds:
    """Fit one percentile threshold per target channel from training targets."""
    _validate_spec(spec)
    targets = _as_nonempty_numeric_array(y_train, "y_train")
    if targets.ndim < 1:
        raise ValueError("y_train must have a leading fit-population axis")
    if targets.shape[0] == 0:
        raise ValueError("y_train must contain at least one fit-population value")

    values = np.percentile(targets, spec.percentile, axis=0)
    return EventThresholds(
        values=np.asarray(values),
        spec_fingerprint=spec.fingerprint(),
    )


def build_event_masks(
    y: np.ndarray,
    thresholds: EventThresholds,
    spec: EventMaskSpec,
    raw_labels: np.ndarray | None = None,
) -> EventMaskBundle:
    """Build valid, raw-exceedance, event, and normal masks for ``y``.

    If ``raw_labels`` is supplied (the file-verified path), it must agree
    exactly with the raw percentile exceedance before corrected masks are
    emitted.
    """
    _validate_spec(spec)
    targets = _as_nonempty_numeric_array(y, "y")
    values = _as_nonempty_numeric_array(thresholds.values, "thresholds.values")

    if thresholds.spec_fingerprint != spec.fingerprint():
        raise ValueError("thresholds spec fingerprint does not match spec")
    if targets.ndim < 1:
        raise ValueError("y must have a leading sample axis")
    if targets.shape[1:] != values.shape:
        raise ValueError(
            "y trailing shape must match thresholds.values shape; "
            f"got {targets.shape[1:]!r} and {values.shape!r}"
        )

    valid = np.greater(targets, spec.valid_min)
    raw_exceedance = np.greater(targets, values)
    if raw_labels is not None:
        supplied = np.asarray(raw_labels)
        if supplied.shape != targets.shape:
            raise ValueError(
                "raw_labels shape must match y shape; "
                f"got {supplied.shape!r} and {targets.shape!r}"
            )
        supplied_mask = _as_binary_raw_labels(supplied)
        if not np.array_equal(supplied_mask, raw_exceedance):
            raise ValueError("raw_labels do not match recomputed raw exceedance")

    event = valid & raw_exceedance
    normal = valid & ~event
    fingerprint = _bundle_fingerprint(
        valid=valid,
        raw_exceedance=raw_exceedance,
        event=event,
        normal=normal,
        protocol_id=spec.protocol_id,
    )
    bundle = EventMaskBundle(
        valid=valid,
        raw_exceedance=raw_exceedance,
        event=event,
        normal=normal,
        protocol_id=spec.protocol_id,
        fingerprint=fingerprint,
    )
    validate_event_partition(bundle)
    return bundle


def validate_event_partition(masks: EventMaskBundle) -> None:
    """Raise ``ValueError`` unless event and normal exactly partition valid."""
    valid = np.asarray(masks.valid, dtype=bool)
    raw_exceedance = np.asarray(masks.raw_exceedance, dtype=bool)
    event = np.asarray(masks.event, dtype=bool)
    normal = np.asarray(masks.normal, dtype=bool)
    shapes = {valid.shape, raw_exceedance.shape, event.shape, normal.shape}
    if len(shapes) != 1:
        raise ValueError("valid, raw_exceedance, event, and normal must share a shape")
    if np.any(event & ~valid):
        raise ValueError("event mask must be a subset of valid mask")
    if np.any(event & normal):
        raise ValueError("event and normal masks must not overlap")
    if not np.array_equal(event | normal, valid):
        raise ValueError("event and normal masks must partition valid mask")


def _validate_spec(spec: EventMaskSpec) -> None:
    if not isinstance(spec, EventMaskSpec):
        raise ValueError("spec must be an EventMaskSpec")
    if spec.comparison != "gt":
        raise ValueError("only strict 'gt' comparison is supported")
    if spec.fit_population != "all_targets":
        raise ValueError("only 'all_targets' fit_population is supported")
    if spec.protocol_id != "train_all_node_flow_p90_valid_v2":
        raise ValueError("protocol_id must be 'train_all_node_flow_p90_valid_v2'")
    if spec.percentile != 90.0:
        raise ValueError("percentile must be exactly 90.0")
    if spec.valid_min != 5.0:
        raise ValueError("valid_min must be exactly 5.0")


def _as_nonempty_numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_binary_raw_labels(raw_labels: np.ndarray) -> np.ndarray:
    if np.issubdtype(raw_labels.dtype, np.bool_):
        return raw_labels
    is_real_number = np.issubdtype(raw_labels.dtype, np.number) and not np.issubdtype(
        raw_labels.dtype, np.complexfloating
    )
    if not is_real_number or not np.all((raw_labels == 0) | (raw_labels == 1)):
        raise ValueError("raw_labels must contain only boolean or exact 0/1 values")
    return raw_labels.astype(bool)


def _bundle_fingerprint(
    *,
    valid: np.ndarray,
    raw_exceedance: np.ndarray,
    event: np.ndarray,
    normal: np.ndarray,
    protocol_id: str,
) -> str:
    payload = {
        "event": _mask_metadata(event),
        "normal": _mask_metadata(normal),
        "protocol_id": protocol_id,
        "raw_exceedance": _mask_metadata(raw_exceedance),
        "valid": _mask_metadata(valid),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mask_metadata(mask: np.ndarray) -> dict:
    binary = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    return {
        "dtype": binary.dtype.str,
        "sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
        "shape": list(binary.shape),
    }

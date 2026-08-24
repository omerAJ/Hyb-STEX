import numpy as np
import torch


def corrected_event_metrics(pred, true, event, valid_min=5.0):
    """Summarize corrected event-mask errors over valid target values.

    The returned values are Python scalars so they can be safely serialized
    alongside training results.  Signed error uses ``pred - true``: positive
    values therefore represent over-prediction.
    """
    _validate_corrected_metric_inputs(pred, true, event)
    if isinstance(pred, np.ndarray):
        valid = true > valid_min
        event_mask = event.astype(bool)
        abs_error = np.abs(pred - true)
        signed_error = pred - true
        any_invalid_event = np.any(event_mask & ~valid)
    else:
        valid = true > valid_min
        event_mask = event.to(dtype=torch.bool)
        abs_error = torch.abs(pred - true)
        signed_error = pred - true
        any_invalid_event = bool(torch.any(event_mask & ~valid).item())
    if any_invalid_event:
        raise ValueError("event mask must be a subset of valid targets")

    normal = valid & ~event_mask
    detail = {}
    _add_corrected_metric_group(detail, "valid", valid, abs_error, signed_error)
    _add_corrected_metric_group(detail, "event", event_mask, abs_error, signed_error)
    _add_corrected_metric_group(detail, "normal", normal, abs_error, signed_error)
    return detail


def _validate_corrected_metric_inputs(pred, true, event):
    supported = (np.ndarray, torch.Tensor)
    if not isinstance(pred, supported) or not isinstance(true, supported) or not isinstance(event, supported):
        raise TypeError("pred, true, and event must all be numpy arrays or torch tensors")
    if type(pred) is not type(true) or type(pred) is not type(event):
        raise TypeError("pred, true, and event must have the same array type")
    if pred.shape != true.shape or pred.shape != event.shape:
        raise ValueError("pred, true, and event must share a shape")
    if isinstance(pred, torch.Tensor) and (
        pred.device != true.device or pred.device != event.device
    ):
        raise ValueError("pred, true, and event must be on the same device")
    if isinstance(event, np.ndarray):
        if np.issubdtype(event.dtype, np.bool_):
            return
        is_real_numeric = np.issubdtype(event.dtype, np.number) and not np.issubdtype(
            event.dtype, np.complexfloating
        )
        is_binary = is_real_numeric and np.all((event == 0) | (event == 1))
    else:
        if event.dtype == torch.bool:
            return
        is_real_numeric = not event.is_complex() and (
            event.is_floating_point() or event.dtype in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        )
        is_binary = is_real_numeric and bool(torch.all((event == 0) | (event == 1)).item())
    if not is_binary:
        raise ValueError("event must contain only bool or exact binary 0/1 values")


def _add_corrected_metric_group(detail, name, mask, abs_error, signed_error):
    if isinstance(abs_error, np.ndarray):
        count = int(np.count_nonzero(mask))
        abs_error_sum = float(abs_error[mask].sum())
        signed_error_sum = float(signed_error[mask].sum())
    else:
        count = int(mask.sum().item())
        abs_error_sum = float(abs_error[mask].sum().item())
        signed_error_sum = float(signed_error[mask].sum().item())
    mae = float("nan") if count == 0 else abs_error_sum / count
    signed_error_mean = float("nan") if count == 0 else signed_error_sum / count
    detail[f"{name}_count"] = count
    detail[f"{name}_abs_error_sum"] = abs_error_sum
    detail[f"{name}_signed_error_sum"] = signed_error_sum
    detail[f"{name}_mae"] = mae
    detail[f"{name}_mean_abs_error"] = mae
    detail[f"{name}_signed_error"] = signed_error_mean

"""
def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)    ## selects strictly greater than
        # nodesMasked=mask[mask==True].shape[0]
        # print("total nodes masked", nodesMasked, "/4096",  "nodes on average masked in each sample: ", nodesMasked/true.shape[0])
        inv_mask = ~mask

        masked_count = mask.sum().item()
        unmasked_count = inv_mask.sum().item()
        # print(f"mask.shape: {mask.shape}, pred.shape: {pred.shape}, true.shape: {true.shape}. ~mask.shape: {inv_mask.shape}")
        unmasked_pred = torch.masked_select(pred, ~mask)
        unmasked_true = torch.masked_select(true, ~mask)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        
    return torch.mean(torch.abs(true-pred)), torch.mean(torch.abs(unmasked_true-unmasked_pred)), masked_count, unmasked_count
"""

def mae_torch_evalLosses(pred, true, mask_value1=None, mask_value2=None):
    if mask_value1 is not None and mask_value2 is not None:
        # Masks for different ranges
        mask1 = true <= mask_value1  # Values less than mask_value1
        mask2 = (true > mask_value1) & (true < mask_value2)  # Values between mask_value1 and mask_value2
        mask3 = true >= mask_value2  # Values greater than mask_value2
        
        # Calculating MAE for each range
        mae1 = torch.mean(torch.abs(pred[mask1] - true[mask1])) if mask1.any() else torch.tensor(float('nan'))
        mae2 = torch.mean(torch.abs(pred[mask2] - true[mask2])) if mask2.any() else torch.tensor(float('nan'))
        mae3 = torch.mean(torch.abs(pred[mask3] - true[mask3])) if mask3.any() else torch.tensor(float('nan'))
        mae_original = torch.mean(torch.abs(pred[~mask1] - true[~mask1]))
        return mae1.item(), mae2.item(), mae3.item(), mask1.sum().item(), mask2.sum().item(), mask3.sum().item(), mae_original.item()
    
def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        # print(f"true.device: {true.device}, pred.device: {pred.device}")
        mask = torch.gt(true, mask_value)
        # nodesMasked=mask[mask==True].shape[0]
        # print("total nodes masked", nodesMasked, "/4096",  "nodes on average masked in each sample: ", nodesMasked/true.shape[0])
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def mse_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((true - pred) ** 2)

def mape_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def mae_np(pred, true, mask_value=None):
    if mask_value != None:
        
        mask = np.where(true > (mask_value), True, False)  ## True where condition met false elsewhere
        # print("mask.shape: ", mask.shape)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred-true))

def mape_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def test_metrics_evalLosses(pred, true, mask1=5, mask2=4000000):
    # mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask1)
        mape = mape_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae_bad, mae_med, mae_good, count_bad, count_med, count_good, mae_original  = mae_torch_evalLosses(pred, true, mask1, mask2)
        mape = mape_torch(pred, true, mask2).item()
    else:
        raise TypeError
    return mae_bad, mae_med, mae_good, mape, count_bad, count_med, count_good, mae_original



def test_metrics(pred, true, evs=None, mask1=5):
    """
    Returns:
        mae: Mean Absolute Error (all points)
        eee: Extreme Event Error (MAE on extreme points, if evs is provided)
    """
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask1)
        eee = np.nan
        if evs is not None:
            mask = evs == 1
            if np.any(mask):
                eee = np.mean(np.abs(pred[mask] - true[mask]))
    elif type(pred) == torch.Tensor:
        mae  = mae_torch(pred, true, mask1).item()
        eee = float('nan')
        if evs is not None:
            eee = eee_torch(pred, true, evs)
    else:
        raise TypeError
    return mae, eee

def eee_torch(pred, true, evs):
    """
    Calculate Extreme Event Error (EEE) as MAE on datapoints where evs == 1.
    Args:
        pred (torch.Tensor): Predictions.
        true (torch.Tensor): Ground truth.
        evs (torch.Tensor): Binary extreme event indicator tensor (same shape as pred/true).
    Returns:
        float: MAE on extreme events, or np.nan if no extreme events.
    """
    mask = evs == 1
    if mask.sum() == 0:
        return float('nan')
    pred_extreme = pred[mask]
    true_extreme = true[mask]
    return torch.mean(torch.abs(true_extreme - pred_extreme)).item()




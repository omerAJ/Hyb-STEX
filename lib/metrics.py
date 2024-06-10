import numpy as np
import torch

def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
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

def test_metrics(pred, true, mask1=5, mask2=5):
    # mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask1)
        mape = mape_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae, masked_mae, masked_count, unmasked_count  = mae_torch(pred, true, mask1)
        mae = mae.item()
        masked_mae = masked_mae.item()
        mape = mape_torch(pred, true, mask2).item()
    else:
        raise TypeError
    return mae, mape, masked_mae, masked_count, unmasked_count



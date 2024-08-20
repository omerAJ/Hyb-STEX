import numpy as np
import torch

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


def gumbell_torch(pred, true, mask_value=None):
    if mask_value != None:
        # print(f"true.device: {true.device}, pred.device: {pred.device}")
        mask = torch.gt(true, mask_value)
        # nodesMasked=mask[mask==True].shape[0]
        # print("total nodes masked", nodesMasked, "/4096",  "nodes on average masked in each sample: ", nodesMasked/true.shape[0])
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    delta = pred - true
    gamma = 1.0
    e = 2.71828
    lg = ((1-e**(-delta**2))**gamma)*delta**2
    return torch.mean(lg)

def frechet_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    
    delta = pred - true
    mask = torch.ge(delta, 0)
    delta_pos = torch.masked_select(pred, mask)
    delta_neg = torch.masked_select(pred, ~mask)
    
    if delta_pos.numel() > 0:  
        s, a = 1.7, 10
        fl_pos = (-1 - a) * (-(delta_pos + s * ((a) / (1 + a)) ** (1 / a)) / s) ** (-a) + torch.log((delta_pos + s * (a / (1 + a)) ** (1 / a)) / s)
        fl_pos = torch.mean(fl_pos)
    else:
        fl_pos = 0
    
    if delta_neg.numel() > 0:
        mae_neg = torch.mean(torch.abs(delta_neg))
    else:
        mae_neg = 0
    return fl_pos + mae_neg



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



def test_metrics(pred, true, mask1=5, mask2=5):
    # mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = mae_np(pred, true, mask1)
        mape = mape_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = mae_torch(pred, true, mask1).item()
        mape = mape_torch(pred, true, mask2).item()
    else:
        raise TypeError
    return mae, mape




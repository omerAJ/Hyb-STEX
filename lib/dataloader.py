import os
import time
import torch 
import numpy as np 

from lib.event_masks import build_event_masks, corrected_p90_spec, fit_event_thresholds

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

def STDataloader(X, Y, evs, bias, batch_size, shuffle=True, drop_last=True, device=None):
    ## Note: bias is only used when we use the fixed bias. A tensor for the fixed bias is passed to the model.
    if device is None:
        cuda = True if torch.cuda.is_available() else False
        TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        X, Y, evs, bias = TensorFloat(X), TensorFloat(Y), TensorFloat(evs), TensorFloat(bias)
    else:
        target_device = torch.device(device)
        X = torch.as_tensor(X, dtype=torch.float32, device=target_device)
        Y = torch.as_tensor(Y, dtype=torch.float32, device=target_device)
        evs = torch.as_tensor(evs, dtype=torch.float32, device=target_device)
        bias = torch.as_tensor(bias, dtype=torch.float32, device=target_device)
    data = torch.utils.data.TensorDataset(X, Y, evs, bias)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last,
    )
    return dataloader

def normalize_data(data, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        scalar = MinMax01Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    # print('{} scalar is used!!!'.format(scalar_type))
    # time.sleep(3)
    return scalar

def get_dataloader(
    data_dir,
    dataset,
    batch_size,
    test_batch_size,
    scalar_type='Standard',
    scaler_fit='train_val',
    event_percentile=None,
    event_mask_protocol='legacy_raw_p90_v1',
    event_label_source='legacy_file_unverified',
    device=None,
):
    data = {}
    _validate_event_options(event_mask_protocol, event_label_source, event_percentile)
    
    # print("input_dataset_context: ", input_dataset_context, input_sequence_type)
    # if input_dataset_context == 19:
    #     print("\n\n in first if\n\n")
    #     input_sequence_dict = {"A":[-4, 19], "B":[-9, -4], "C":[-14, -9], "D":[-19, -14]}
    #     input_sequence = input_sequence_dict[input_sequence_type]
    # elif input_dataset_context == 35:
    #     input_sequence_dict = {"A":[-8, 35], "B":[-17, -8], "C":[-26, -17], "D":[-35, -26]}
    #     input_sequence = input_sequence_dict[input_sequence_type]

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'))
        # skip = cat_data['x'].shape[1] - input_length
        # print(f"cat_data['x'].shape: {cat_data['x'].shape}, cat_data['y'].shape: {cat_data['y'].shape}, cat_data['evs_90'].shape: {cat_data['evs_90'].shape}")
        
        
        # if dataset == 'NYCBike1':
        #     data['x_' + category] = cat_data['x'][:, -9:19, :, :]  # cat_data['x'].shape: (1912, 35, 200, 2)
        # else:
        #     data['x_' + category] = cat_data['x'][:, -17:35, :, :]  # cat_data['x'].shape: (1912, 35, 200, 2)
        # print("indexing")

        # print("not indexing")
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['evs_' + category] = cat_data['evs_90']
        data['bias_' + category] = cat_data['evs_90']  ## This is a placeholder for the bias, which is not used in the current implementation.
        # print("using 90percent evs")
    event_metadata = None
    if event_mask_protocol == 'train_all_node_flow_p90_valid_v2':
        spec = corrected_p90_spec()
        thresholds = fit_event_thresholds(data['y_train'], spec)
        fingerprints = {}
        for category in ['train', 'val', 'test']:
            raw_labels = data['evs_' + category] if event_label_source == 'file_verified' else None
            masks = build_event_masks(
                data['y_' + category], thresholds, spec, raw_labels=raw_labels
            )
            labels = masks.event.astype(np.float32)
            data['evs_' + category] = labels
            data['bias_' + category] = labels
            fingerprints[category] = masks.fingerprint
        event_metadata = {
            'protocol': spec.to_dict(),
            'protocol_fingerprint': spec.fingerprint(),
            'thresholds': thresholds.values.copy(),
            'mask_fingerprints': fingerprints,
        }
    elif event_percentile is not None:
        if not 0.0 < float(event_percentile) < 100.0:
            raise ValueError("event_percentile must be between 0 and 100")
        thresholds = np.percentile(
            data['y_train'], float(event_percentile), axis=0
        )
        for category in ['train', 'val', 'test']:
            labels = (data['y_' + category] > thresholds).astype(np.float32)
            data['evs_' + category] = labels
            data['bias_' + category] = labels
    if scaler_fit == 'train':
        scaler_data = data['x_train']
    elif scaler_fit == 'train_val':
        scaler_data = np.concatenate([data['x_train'], data['x_val']], axis=0)
    else:
        raise ValueError("scaler_fit must be 'train' or 'train_val'")
    scaler = normalize_data(scaler_data, scalar_type)
    # print("skip: ", skip)
    # Data format
    # print("\n\n!!Scaling is NOT off!!\n\n")
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    print("data['x_train'].shape: ", data['x_train'].shape, data['y_train'].shape)
    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'], 
        data['y_train'], 
        data['evs_train'], 
        data['bias_train'], 
        batch_size, 
        shuffle=True,
        device=device,
    )
    dataloader['val'] = STDataloader(
        data['x_val'], 
        data['y_val'], 
        data['evs_val'], 
        data['bias_val'], 
        test_batch_size, 
        shuffle=False,
        device=device,
    )
    dataloader['test'] = STDataloader(
        data['x_test'], 
        data['y_test'], 
        data['evs_test'], 
        data['bias_test'], 
        test_batch_size, 
        shuffle=False, 
        drop_last=False,
        device=device,
    )
    dataloader['scaler'] = scaler
    dataloader['event_mask_protocol'] = event_mask_protocol
    dataloader['event_label_source'] = event_label_source
    if event_metadata is not None:
        dataloader['event_thresholds'] = event_metadata['thresholds']
        dataloader['event_mask_metadata'] = event_metadata
    return dataloader


def _validate_event_options(event_mask_protocol, event_label_source, event_percentile):
    supported_protocols = {
        'legacy_raw_p90_v1',
        'train_all_node_flow_p90_valid_v2',
    }
    supported_label_sources = {
        'legacy_file_unverified',
        'file_verified',
        'generated',
    }
    if event_mask_protocol not in supported_protocols:
        raise ValueError("event_mask_protocol is not supported")
    if event_label_source not in supported_label_sources:
        raise ValueError("event_label_source is not supported")
    allowed_label_sources = {
        'legacy_raw_p90_v1': {'legacy_file_unverified'},
        'train_all_node_flow_p90_valid_v2': {'file_verified', 'generated'},
    }
    if event_label_source not in allowed_label_sources[event_mask_protocol]:
        raise ValueError(
            "event_label_source is not supported for event_mask_protocol"
        )
    if event_mask_protocol == 'train_all_node_flow_p90_valid_v2' and event_percentile is not None:
        raise ValueError("event_percentile is not supported with train_all_node_flow_p90_valid_v2")

if __name__ == '__main__':
    loader = get_dataloader('../data/', 'NYCBike1', batch_size=64, test_batch_size=64)
    for key in loader.keys():
        print(key)

import os
import time
import torch 
import numpy as np 

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

def STDataloader(X, Y, evs, bias, batch_size, shuffle=True, drop_last=True):
    ## Note: bias is only used when we use the fixed bias. A tensor for the fixed bias is passed to the model.
    cuda = True if torch.cuda.is_available() else False
    # cuda = False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y, evs, bias = TensorFloat(X), TensorFloat(Y), TensorFloat(evs), TensorFloat(bias)
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

def get_dataloader(data_dir, dataset, batch_size, test_batch_size, scalar_type='Standard'):
    data = {}
    
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

        print("not indexing")
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['evs_' + category] = cat_data['evs_90']
        data['bias_' + category] = cat_data['bias']
        print("using 90percent evs")
    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)
    # print("skip: ", skip)
    # Data format
    print("\n\n!!Scaling is NOT off!!\n\n")
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    print("data['x_train'].shape: ", data['x_train'].shape, data['y_train'].shape, "\n\n!!train shuffle is True!!\n\n")
    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader(
        data['x_train'], 
        data['y_train'], 
        data['evs_train'], 
        data['bias_train'], 
        batch_size, 
        shuffle=True
    )
    dataloader['val'] = STDataloader(
        data['x_val'], 
        data['y_val'], 
        data['evs_val'], 
        data['bias_val'], 
        test_batch_size, 
        shuffle=False
    )
    dataloader['test'] = STDataloader(
        data['x_test'], 
        data['y_test'], 
        data['evs_test'], 
        data['bias_test'], 
        test_batch_size, 
        shuffle=False, 
        drop_last=False
    )
    dataloader['scaler'] = scaler
    return dataloader

if __name__ == '__main__':
    loader = get_dataloader('../data/', 'NYCBike1', batch_size=64, test_batch_size=64)
    for key in loader.keys():
        print(key)
import os

import numpy as np
import torch


class StandardScaler:
    """Standardize inputs with a single mean/std pair."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.mean, np.ndarray):
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


class MinMax01Scaler:
    """Scale inputs to [0, 1]."""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.min, np.ndarray):
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min)) + self.min


class MinMax11Scaler:
    """Scale inputs to [-1, 1]."""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2.0 - 1.0

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.min, np.ndarray):
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.0) / 2.0) * (self.max - self.min) + self.min


class PEMS04FlowScaler:
    """Scale only the flow channel; keep time-of-week untouched."""

    def __init__(self, flow_mean, flow_std):
        self.flow_mean = np.asarray(flow_mean, dtype=np.float32)
        self.flow_std = np.asarray(max(float(flow_std), 1.0e-6), dtype=np.float32)

    def _get_stats(self, data):
        if isinstance(data, torch.Tensor):
            flow_mean = torch.as_tensor(self.flow_mean, device=data.device, dtype=data.dtype)
            flow_std = torch.as_tensor(self.flow_std, device=data.device, dtype=data.dtype)
            return flow_mean, flow_std
        return self.flow_mean, self.flow_std

    def transform_inputs(self, data):
        flow_mean, flow_std = self._get_stats(data)
        scaled = data.copy() if isinstance(data, np.ndarray) else data.clone()
        scaled[..., 0] = (scaled[..., 0] - flow_mean) / flow_std
        return scaled

    def transform_targets(self, data):
        flow_mean, flow_std = self._get_stats(data)
        scaled = data.copy() if isinstance(data, np.ndarray) else data.clone()
        scaled[..., 0] = (scaled[..., 0] - flow_mean) / flow_std
        return scaled

    def inverse_transform(self, data):
        flow_mean, flow_std = self._get_stats(data)
        restored = data.clone() if isinstance(data, torch.Tensor) else data.copy()
        if restored.shape[-1] == 1:
            restored = restored * flow_std + flow_mean
        else:
            restored[..., 0] = restored[..., 0] * flow_std + flow_mean
        return restored


def STDataloader(X, Y, evs, bias, batch_size, shuffle=True, drop_last=True):
    """Construct a tensor dataloader on CPU or CUDA tensors."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    evs = torch.as_tensor(evs, dtype=torch.float32, device=device)
    bias = torch.as_tensor(bias, dtype=torch.float32, device=device)
    data = torch.utils.data.TensorDataset(X, Y, evs, bias)
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def normalize_data(data, scalar_type="Standard"):
    scalar = None
    if scalar_type == "MinMax01":
        scalar = MinMax01Scaler(min=data.min(), max=data.max())
    elif scalar_type == "MinMax11":
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == "Standard":
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError("scalar_type is not supported in data_normalization.")
    return scalar


def _get_extreme_value_tensor(cat_data, dataset_path, evs_key):
    if not evs_key:
        raise ValueError("evs_key must be provided in the dataset config.")

    if evs_key not in cat_data.files:
        available_keys = ", ".join(cat_data.files)
        raise KeyError(
            f"Requested EV tensor '{evs_key}' not found in {dataset_path}. "
            f"Available keys: [{available_keys}]"
        )

    return cat_data[evs_key]


def _slice_pems04_samples(data_array, index_array):
    inputs = []
    targets = []
    for start_idx, split_idx, end_idx in index_array:
        inputs.append(data_array[start_idx:split_idx, :, :])
        targets.append(data_array[split_idx:end_idx, :, [0]])
    return np.stack(inputs, axis=0), np.stack(targets, axis=0)


def _load_pems04_dataset(dataset_dir):
    raw_data = np.load(os.path.join(dataset_dir, "data.npz"))["data"].astype(np.float32)
    index_data = np.load(os.path.join(dataset_dir, "index.npz"))

    split_payload = {}
    for category in ["train", "val", "test"]:
        x_split, y_split = _slice_pems04_samples(raw_data, index_data[category])
        split_payload[f"x_{category}"] = x_split.astype(np.float32)
        split_payload[f"y_{category}"] = y_split.astype(np.float32)
        split_payload[f"evs_{category}"] = np.zeros_like(y_split, dtype=np.float32)
        split_payload[f"bias_{category}"] = np.zeros_like(y_split, dtype=np.float32)
    return split_payload


def get_dataloader(
    data_dir,
    dataset,
    batch_size,
    test_batch_size,
    evs_key,
    scalar_type="Standard",
    pems04_evs_quantile=0.95,
):
    del pems04_evs_quantile
    data = {}
    dataset_dir = os.path.join(data_dir, dataset)

    if dataset == "PEMS04":
        data.update(_load_pems04_dataset(dataset_dir))
        flow_mean = float(data["x_train"][..., 0].mean())
        flow_std = float(data["x_train"][..., 0].std())
        scaler = PEMS04FlowScaler(flow_mean=flow_mean, flow_std=flow_std)
        for category in ["train", "val", "test"]:
            data[f"x_{category}"] = scaler.transform_inputs(data[f"x_{category}"])
            data[f"y_{category}"] = scaler.transform_targets(data[f"y_{category}"])
        print(
            "Loaded PEMS04 indexed dataset from {} with full 12-step targets. "
            "Flow-only normalization mean={:.4f}, std={:.4f}".format(
                dataset_dir,
                flow_mean,
                flow_std,
            )
        )
    else:
        for category in ["train", "val", "test"]:
            dataset_path = os.path.join(dataset_dir, category + ".npz")
            cat_data = np.load(dataset_path)
            evs_tensor = _get_extreme_value_tensor(cat_data, dataset_path, evs_key)
            data["x_" + category] = cat_data["x"]
            data["y_" + category] = cat_data["y"]
            data["evs_" + category] = evs_tensor
            data["bias_" + category] = evs_tensor
            print(f"Loaded {category} EV labels from {dataset_path} using key '{evs_key}'")
        scaler = normalize_data(np.concatenate([data["x_train"], data["x_val"]], axis=0), scalar_type)
        for category in ["train", "val", "test"]:
            data["x_" + category] = scaler.transform(data["x_" + category])
            data["y_" + category] = scaler.transform(data["y_" + category])

    print("data['x_train'].shape: ", data["x_train"].shape, data["y_train"].shape)
    dataloader = {}
    dataloader["train"] = STDataloader(
        data["x_train"],
        data["y_train"],
        data["evs_train"],
        data["bias_train"],
        batch_size,
        shuffle=True,
    )
    dataloader["val"] = STDataloader(
        data["x_val"],
        data["y_val"],
        data["evs_val"],
        data["bias_val"],
        test_batch_size,
        shuffle=False,
    )
    dataloader["test"] = STDataloader(
        data["x_test"],
        data["y_test"],
        data["evs_test"],
        data["bias_test"],
        test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    dataloader["scaler"] = scaler
    return dataloader


if __name__ == "__main__":
    loader = get_dataloader("../data/", "NYCBike1", batch_size=64, test_batch_size=64, evs_key="evs_90")
    for key in loader.keys():
        print(key)

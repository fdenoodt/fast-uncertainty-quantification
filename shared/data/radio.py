from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data_and_label(path, batch_size, data_type, num_workers: int) -> DataLoader:
    print(f"info: start loading the {data_type} data")
    X_data = torch.load(f"{path}X_{data_type}.pth")  # (b, c, t)
    Y_data = torch.load(f"{path}y_{data_type}.pth")  # (b, 1)

    # Flatten the channel and time dimensions for min and max calculation
    flat_X_data = X_data.view(X_data.shape[0], -1)  # Flattening to (b, c*t)

    # Calculate global min and max across both channels
    min_vals = flat_X_data.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    max_vals = flat_X_data.max(dim=1, keepdim=True)[0].view(-1, 1, 1)

    # Apply normalization using global min and max
    X_data_scaled = -1 + 2 * (X_data - min_vals) / (max_vals - min_vals)
    X_data_scaled = X_data_scaled.float()
    print(f"info: loaded the {data_type} signals")

    data_dataset = TensorDataset(X_data_scaled, Y_data)
    shuffle = True if data_type == 'train' else False
    data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                             num_workers=num_workers,
                             pin_memory=True, persistent_workers=True)
    return data_loader


def get_radio_data_loaders(all_datasets_path: str, batch_size: int, num_workers: int) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    path = f"{all_datasets_path}/RadioIdentification/"
    train_loader: DataLoader = load_data_and_label(path, batch_size, 'train', num_workers=num_workers)

    # create a sub_train_loader which is the subset of the train_loader of percentage 10%
    # subset_percentage = config.train_subset_percentage
    # if subset_percentage < 1:
    #     subset_size = int(subset_percentage * len(train_loader.dataset))
    #     indices = torch.randperm(len(train_loader.dataset))[:subset_size]
    #     train_subset = torch.utils.data.Subset(train_loader.dataset, indices)
    #     train_loader = DataLoader(train_subset, batch_size=config.batch_size_multiGPU, shuffle=True, drop_last=True)
    # print(f"Info: Train loader shape: {len(train_loader.dataset)}, subset percentage: {subset_percentage}")

    val_loader = load_data_and_label(path, batch_size, 'val', num_workers=num_workers)
    test_loader = load_data_and_label(path, batch_size, 'test', num_workers=num_workers)
    return train_loader, val_loader, test_loader

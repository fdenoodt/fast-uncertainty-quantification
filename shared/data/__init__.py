from torch.utils.data import DataLoader


def filter_iterator(data_loader: DataLoader, limit_batches: float):
    total_batches = len(data_loader)
    num_batches_to_process = int(total_batches * limit_batches)

    for i, batch in enumerate(data_loader):
        if i >= num_batches_to_process:
            break
        yield i, batch

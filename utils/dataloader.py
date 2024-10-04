
import torch
from torch.utils.data.distributed import DistributedSampler


def get_sample_data_loader(model, total_samples, input_dim, device, batch_size):
    train_data = torch.randn(total_samples, input_dim, device=device, dtype=torch.half)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(input_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    #train_sampler, val_sampler = DistributedSampler(train_dataset), DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader, val_loader
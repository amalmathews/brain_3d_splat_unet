from torch.utils.data import DataLoader, random_split
from src.data.dataset import BraTSDataset
import torch

def get_brats_dataloader(batch_size=1, data_path="data/BraTS2021/BraTS2021_Training_Data", val_split=0.2):
    dataset = BraTSDataset(data_path)
    
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader, random_split


class MusicDataset(Dataset):
    def __init__(self, features, attributes, rel_paths):
        self.features = features
        self.attributes = attributes
        self.rel_paths = rel_paths

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.attributes[idx], self.rel_paths[idx]

def create_data_loaders(data_vector, attribute_tensor, rel_paths, batch_size):
    dataset_with_paths = MusicDataset(data_vector, attribute_tensor, rel_paths)
    
    train_size = int(0.8 * len(dataset_with_paths))
    val_size = int(0.1 * len(dataset_with_paths))
    test_size = len(dataset_with_paths) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset_with_paths, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

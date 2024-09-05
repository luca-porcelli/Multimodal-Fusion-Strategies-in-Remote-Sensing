import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RandomDataset(Dataset):
    def __init__(self, num_samples, image_size=(11, 256, 256), num_classes=15):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = np.random.rand(*self.image_size).astype(np.float32)
        
        mask = np.random.randint(0, self.num_classes, self.image_size[1:]).astype(np.int64)
        
        return torch.tensor(image), torch.tensor(mask)
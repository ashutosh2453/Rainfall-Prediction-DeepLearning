import torch
from torch.utils.data import Dataset


class RainfallDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]

        # Add channel dimension
        # (7,129,135) -> (7,1,129,135)
        x = x.unsqueeze(1)

        y = self.y[idx]

        # Add channel dimension
        # (129,135) -> (1,129,135)
        y = y.unsqueeze(0)

        return x, y
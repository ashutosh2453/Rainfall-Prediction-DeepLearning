import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    CNN Baseline Model
    Input : (Batch, 7, 1, 129, 135)
    Output: (Batch, 1, 129, 135)
    """

    def __init__(self):

        super().__init__()

        # Spatial Feature Extractor
        self.cnn = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=1)

        )

    def forward(self, x):

        # x shape:
        # (Batch, Time, Channel, Height, Width)

        # Use only the latest rainfall map
        x = x[:, -1]

        out = self.cnn(x)

        return out
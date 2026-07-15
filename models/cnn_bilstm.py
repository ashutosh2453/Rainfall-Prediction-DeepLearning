import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    CNN + BiLSTM for Rainfall Prediction

    Input:
        (Batch, 7, 1, 129, 135)

    Output:
        (Batch, 1, 129, 135)
    """

    def __init__(self):
        super(CNNBiLSTM, self).__init__()

        # =====================================================
        # CNN Feature Extractor
        # =====================================================

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # =====================================================
        # BiLSTM
        # =====================================================

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # =====================================================
        # Fully Connected Regression Head
        # =====================================================

        self.regressor = nn.Sequential(

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(
                512,
                129 * 135
            )
        )

    def forward(self, x):
        """
        Input:
            x -> (Batch, 7, 1, 129, 135)
        """

        batch_size = x.size(0)
        time_steps = x.size(1)

        features = []

        # -----------------------------------------
        # CNN Feature Extraction
        # -----------------------------------------

        for t in range(time_steps):

            frame = x[:, t, :, :, :]      # (Batch,1,129,135)

            feature = self.feature_extractor(frame)

            feature = feature.view(batch_size, -1)

            features.append(feature)

        # -----------------------------------------
        # Stack Features
        # (Batch,7,64)
        # -----------------------------------------

        features = torch.stack(features, dim=1)

        # -----------------------------------------
        # BiLSTM
        # -----------------------------------------

        lstm_out, _ = self.bilstm(features)

        last_output = lstm_out[:, -1, :]

        prediction = self.regressor(last_output)

        prediction = prediction.view(
            batch_size,
            1,
            129,
            135
        )

        return prediction
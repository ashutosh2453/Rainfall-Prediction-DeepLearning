import torch
import torch.nn as nn


# ==========================================================
# Attention Module
# ==========================================================

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):

        # lstm_output
        # (Batch, Time, Hidden*2)

        weights = torch.softmax(
            self.attention(lstm_output),
            dim=1
        )

        context = torch.sum(
            weights * lstm_output,
            dim=1
        )

        return context


# ==========================================================
# CNN + BiLSTM + Attention
# ==========================================================

class CNNBiLSTMAttention(nn.Module):

    """
    Input
        (Batch, 7, 1, 129, 135)

    Output
        (Batch, 1, 129, 135)
    """

    def __init__(self):

        super().__init__()

        # ---------------------------------------------------
        # CNN Encoder
        # ---------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))

        )

        # ---------------------------------------------------
        # BiLSTM
        # ---------------------------------------------------

        self.lstm = nn.LSTM(

            input_size=64,

            hidden_size=128,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.3

        )

        # ---------------------------------------------------
        # Attention
        # ---------------------------------------------------

        self.attention = Attention(128)

        # ---------------------------------------------------
        # Regression Head
        # ---------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                256,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                129 * 135
            )

        )

    def forward(self, x):

        """
        x

        (Batch,7,1,129,135)
        """

        batch_size = x.size(0)

        sequence = []

        # -----------------------------------------------
        # CNN Feature Extraction
        # -----------------------------------------------

        for t in range(x.size(1)):

            frame = x[:, t]          # (B,1,129,135)

            feature = self.cnn(frame)

            feature = feature.view(
                batch_size,
                -1
            )

            sequence.append(feature)

        sequence = torch.stack(
            sequence,
            dim=1
        )

        # -----------------------------------------------
        # BiLSTM
        # -----------------------------------------------

        lstm_output, _ = self.lstm(sequence)

        # -----------------------------------------------
        # Attention
        # -----------------------------------------------

        context = self.attention(lstm_output)

        # -----------------------------------------------
        # Regression
        # -----------------------------------------------

        prediction = self.fc(context)

        prediction = prediction.view(

            batch_size,

            1,

            129,

            135

        )

        return prediction
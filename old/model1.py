import torch
import torch.nn as nn


class HybridECGModel(nn.Module):
    """
    Two branches:
      1. CNN+BiLSTM  -> learns temporal patterns from raw ECG
      2. MLP         -> learns from HRV engineered features
    Combined -> final emotion classifier
    """
    def __init__(self, hrv_dim=6, num_classes=4):
        super().__init__()

        # Branch 1: Raw ECG
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)

        # Branch 2: HRV features
        self.hrv_net = nn.Sequential(
            nn.Linear(hrv_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),     nn.ReLU(),
        )

        # Combined classifier
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, ecg, hrv):
        x = self.cnn(ecg)              # (B, 128, 32)
        x = x.permute(0, 2, 1)        # (B, 32, 128)
        x, _ = self.lstm(x)           # (B, 32, 128)
        x = x[:, -1, :]               # (B, 128)
        h = self.hrv_net(hrv)         # (B, 32)
        out = torch.cat([x, h], dim=1)
        return self.head(out)

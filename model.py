import torch
import torch.nn as nn


class HybridECGModel(nn.Module):
    def __init__(self, hrv_dim=10, num_classes=4):  # hrv_dim updated: 6 -> 10
        super().__init__()

        # Branch 1: Raw ECG (CNN + BiLSTM)
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

        # Branch 2: 10 HRV features (MLP)
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
        x = self.cnn(ecg)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        h = self.hrv_net(hrv)
        out = torch.cat([x, h], dim=1)
        return self.head(out)

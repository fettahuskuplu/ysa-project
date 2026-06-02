import torch
import torch.nn as nn


class GRUDQN(nn.Module):
    """
    Zaman serisi verilerindeki ardışık ilişkileri (trend, momentum vb.)
    yakalamak için tasarlanmış Gated Recurrent Unit (GRU) tabanlı Q-Ağı.
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_layers: int = 1):
        super(GRUDQN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # batch_first=True, tensörlerin (batch_size, seq_len, features) formatında gelmesini sağlar
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Karar aşaması için tam bağlı (Fully Connected) katmanlar
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x boyutu: (batch_size, window_size, input_dim)
        gru_out, _ = self.gru(x)

        # Sadece son zaman adımının (pencerenin en güncel anı) çıktısını alıyoruz
        last_step_out = gru_out[:, -1, :]

        out = self.fc1(last_step_out)
        out = self.relu(out)
        q_values = self.fc2(out)

        return q_values
import torch
import torch.nn as nn

from config import DuelingDQNConfig

class DuelingDQNNetwork(nn.Module):

    def __init__(self, input_size: int):
        super(DuelingDQNNetwork, self).__init__()

        hidden_size = DuelingDQNConfig.HIDDEN_SIZE
        action_size = DuelingDQNConfig.ACTION_SIZE

        # Ortak katman
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3D gelirse düzleştir: (batch, window, features) → (batch, window×features)
        x = x.view(x.size(0), -1)

        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
 
        return q_values
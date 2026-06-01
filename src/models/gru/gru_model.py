import torch
import torch.nn as nn

class GRUDQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_layers: int = 1):
        super(GRUDQN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_step_out = gru_out[:, -1, :]

        out = self.fc1(last_step_out)
        out = self.relu(out)
        q_values = self.fc2(out)

        return q_values
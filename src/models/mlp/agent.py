import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import deque
from .model import MLPQNetwork

class MLPDQNAgent:
    def __init__(self, window_size: int, num_features: int, action_size: int = 3, lr=0.0005, gamma=0.98, tau=0.005):
        self.window_size = window_size
        self.num_features = num_features
        self.action_size = action_size
        self.gamma = gamma 
        self.tau = tau     
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.batch_size = 64 
        self.memory = deque(maxlen=10000) 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = MLPQNetwork(window_size, num_features, action_size).to(self.device)
        self.target_net = MLPQNetwork(window_size, num_features, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) 
        self.criterion = nn.MSELoss()

    def get_action(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size) 
        
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(torch.argmax(q_values).item()) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))
            
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(loss.item())   

    def save_as_json_for_dashboard(self, filepath: str, metrics: dict):
        """
        Backend ve Frontend şemasına tam uyumlu JSON çıktısı üretir.
        """
        output_data = {
            "metrics": {
                "cumulative_return_pct": float(metrics.get("return_pct", 0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0)),
                "total_trades": int(metrics.get("trade_count", 0))
            },
            "portfolio_history": [float(v) for v in metrics.get("portfolio_history", [])]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print(f"--> 🎉 [JSON EXPORT] Dashboard uyumlu veri '{filepath}' konumuna yazıldı.")
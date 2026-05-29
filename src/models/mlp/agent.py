import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from .model import MLPQNetwork

class MLPDQNAgent:
    def __init__(self, window_size: int, num_features: int, action_size: int = 3, lr=0.0005, gamma=0.98, tau=0.005):
        self.window_size = window_size
        self.num_features = num_features
        self.action_size = action_size
        self.gamma = gamma # Künyedeki İndirim faktörü
        self.tau = tau     # Soft Update (Yumuşak Güncelleme) katsayısı
        
        # Epsilon-Greedy Parametreleri
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.batch_size = 64 # Künyedeki Batch size aralığına uygun
        self.memory = deque(maxlen=10000) # Replay Memory (Deneyim Hafızası)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Kararlı bir DQN için Policy ve Target ağları
        self.policy_net = MLPQNetwork(window_size, num_features, action_size).to(self.device)
        self.target_net = MLPQNetwork(window_size, num_features, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) # Künyedeki Öğrenme Oranı
        self.criterion = nn.MSELoss()

    def get_action(self, state: np.ndarray) -> int:
        """Ortamdan (environment.py) aldığı state'e göre aksiyon seçer."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size) # Keşif (Rastgele işlem)
        
        # model.py içindeki forward fonksiyonu tensör dönüşümünü ve boyut kontrolünü güvenle yapar
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(torch.argmax(q_values).item()) # En yüksek Q değerine sahip aksiyon

    def remember(self, state, action, reward, next_state, done):
        """Ajanın yaşadığı deneyimi hafızaya kaydeder."""
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        """Target network ağırlıklarını pürüzsüz (Soft Update) olarak günceller."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self) -> float:
        """Hafızadan batch çekerek ağın ağırlıklarını günceller."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Mevcut durumdaki tahmin edilen Q değerleri
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Bellman denklemine göre hedef Q değerleri
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))
            
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # REVİZE GİDERİLDİ: Gradyan patlamasını engellemek için kısıtlama (Clipping)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Epsilon sönümleme
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(loss.item())   
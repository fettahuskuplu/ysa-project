import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import DuelingDQNConfig
from network import DuelingDQNNetwork
from replay_buffer import ReplayBuffer

class DuelingDQNAgent:

    def __init__(self, input_size: int):
        self.epsilon       = DuelingDQNConfig.EPSILON_START
        self.epsilon_end   = DuelingDQNConfig.EPSILON_END
        self.epsilon_decay = DuelingDQNConfig.EPSILON_DECAY
        self.gamma         = DuelingDQNConfig.GAMMA
        self.batch_size    = DuelingDQNConfig.BATCH_SIZE             
        self.target_update = DuelingDQNConfig.TARGET_UPDATE_FREQUENCY
        self.action_size   = DuelingDQNConfig.ACTION_SIZE

        self.device = torch.device(DuelingDQNConfig.DEVICE)

         # Online network: her adımda güncellenir, aksiyon seçmek için kullanılır
        self.online_net = DuelingDQNNetwork(input_size).to(self.device)
 
        # Target network: sabit tutulur, "doğru cevap" üretmek için kullanılır
        self.target_net = DuelingDQNNetwork(input_size).to(self.device)
 
        # Target network başlangıçta online ile aynı ağırlıkları taşısın
        self.target_net.load_state_dict(self.online_net.state_dict())
 
        # Target network hiç gradient hesaplamaz — sadece forward pass yapar
        self.target_net.eval()

        # Adam optimizer ile ağırlıklar güncelleniyor
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=DuelingDQNConfig.LEARNING_RATE
        )

        # Loss fonksiyonu MSE
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer()

        # Episode sayacı (target network'ü ne zaman güncelleyeceğimi belirlemek için)
        self.episode_count = 0

    # AKSİYON SEÇİMİ
    def select_action(self, state: np.ndarray) -> int:
        """
        Aksiyon seçer.
        int: 0 (HOLD), 1 (BUY) veya 2 (SELL)
        """
        if random.random() < self.epsilon:
            # Keşif: rastgele aksiyon seçiyoruz
            return random.randint(0, self.action_size -1)
        else:
            self.online_net.eval()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_t)
                action = q_values.argmax(dim=1).item()
            self.online_net.train()
            return action
        
    # EĞİTİM ADIMI
    def train_step(self) -> float | None:
        """
        Her environment adımından sonra çağırılır (buffer yeterli olduğu sürece).
        """

        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── HEDEF Q DEĞERİ (Bellman denklemi) ──────────────────
        # Bellman: Q(s,a) = r + γ * max_a'(Q(s',a'))
        # r: alınan ödül
        # γ (gamma): gelecek ödüllerin ne kadar önemsendiği
        # max_a'(...): bir sonraki durumda alabileceğimiz en iyi Q değeri
        # (1 - done): episode bittiyse gelecek yok, sadece r alınır
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # ── TAHMİN (Online network'ün mevcut Q tahmini) ─────────
        current_q_all = self.online_net(states_t)
        current_q = current_q_all.gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)
 
        # ── LOSS & BACKPROP ─────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)
 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        return loss.item()
    
    # EPISODE SONU
    def on_episode_end(self) -> None:
        """
        Her episode sonunda çağrılır.
        Epsilon azalt: Her episode sonunda epsilon × 0.995 ile küçülür.
        Target network güncelle:
           Her 10 episode'da bir online network'ün
           ağırlıklarını target'a kopyalarız.
        """
        self.episode_count += 1
 
        # Epsilon azalt ama minimum değerin altına düşürme
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
 
        # Her 10 episode'da target network'ü güncelle
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            print(f"  [Target network güncellendi] Episode: {self.episode_count} | Epsilon: {self.epsilon:.3f}")

    # KAYDET / YÜKLE
    def save(self, path: str) -> None:
        """Eğitilmiş modeli diske kaydeder."""
        torch.save({
            "online_net":    self.online_net.state_dict(),
            "target_net":    self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "episode_count": self.episode_count,
        }, path)
        print(f"[Model kaydedildi] {path}")
 
    def load(self, path: str) -> None:
        """Kaydedilmiş modeli yükler."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon       = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        print(f"[Model yüklendi] {path}")
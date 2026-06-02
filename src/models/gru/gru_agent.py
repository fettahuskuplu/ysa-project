import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import GRUDQNConfig
from gru_model import GRUDQN
from replay_buffer import ReplayBuffer


class GRUDQNAgent:
    """
    Borsa İstanbul (BIST) verileri üzerinde GRU (Gated Recurrent Unit) tabanlı
    Derin Pekiştirmeli Öğrenme (DQN) ajanı.
    Zaman serilerindeki ardışık ilişkileri yakalamak için tasarlanmıştır.
    """

    def __init__(self, input_size: int):
        # --- Hiperparametreler ---
        self.epsilon = GRUDQNConfig.EPSILON_START
        self.epsilon_end = GRUDQNConfig.EPSILON_END
        self.epsilon_decay = GRUDQNConfig.EPSILON_DECAY
        self.gamma = GRUDQNConfig.GAMMA
        self.batch_size = GRUDQNConfig.BATCH_SIZE
        self.target_update = GRUDQNConfig.TARGET_UPDATE_FREQUENCY
        self.action_size = GRUDQNConfig.ACTION_SIZE

        # Donanım hızlandırıcısı (Mac için genelde 'mps', aksi halde 'cuda' veya 'cpu')
        self.device = torch.device(GRUDQNConfig.DEVICE)

        # --- Sinir Ağları (Neural Networks) ---
        # Online network: Her adımda güncellenir, aksiyon seçmek için kullanılır
        self.online_net = GRUDQN(
            input_size,
            GRUDQNConfig.HIDDEN_SIZE,
            self.action_size,
            GRUDQNConfig.NUM_LAYERS
        ).to(self.device)

        # Target network: Sabit tutulur, Q-öğrenme hedefini stabil kılmak için kullanılır
        self.target_net = GRUDQN(
            input_size,
            GRUDQNConfig.HIDDEN_SIZE,
            self.action_size,
            GRUDQNConfig.NUM_LAYERS
        ).to(self.device)

        # Başlangıçta iki ağın ağırlıklarını eşitliyoruz
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network sadece tahmin yapar, gradyan hesaplamaz

        # --- Optimizasyon ve Kayıp Fonksiyonu ---
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=GRUDQNConfig.LEARNING_RATE
        )

        # GRU için aykırı değerlere karşı dirençli olan Huber Loss (SmoothL1Loss) tercih edilmiştir
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer()

        # Episode sayacı (Epsilon azaltma ve target network güncelleme takibi için)
        self.episode_count = 0

    # AKSİYON SEÇİMİ
    def select_action(self, state: np.ndarray) -> int:
        """
        Mevcut duruma göre Epsilon-Greedy politikasıyla aksiyon seçer.
        Returns: 0 (HOLD), 1 (BUY) veya 2 (SELL)
        """
        if random.random() < self.epsilon:
            # Keşif (Exploration): Rastgele aksiyon
            return random.randint(0, self.action_size - 1)
        else:
            # Sömürü (Exploitation): Ağın tahmini
            self.online_net.eval()
            with torch.no_grad():
                # GRU'nun beklediği boyutlara getirme işlemi
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_t)
                action = q_values.argmax(dim=1).item()
            self.online_net.train()
            return action

    # EĞİTİM ADIMI
    def train_step(self) -> float | None:
        """
        Replay Buffer'dan alınan mini-batch ile Online Network'ü eğitir.
        Ortamdan her adım (step) atıldığında çağrılır.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None  # Yeterli veri yoksa eğitimi atla

        # Buffer'dan rastgele deneyim örnekle
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # ── HEDEF Q DEĞERİ (Bellman Denklemi) ──────────────────
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            max_next_q = next_q.max(dim=1)[0]
            # Sadece done=False olan durumlar için gelecekteki ödülleri hesaba kat
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # ── TAHMİN (Online Network) ─────────
        # Ağın mevcut durum için ürettiği tüm Q değerlerinden, gerçekten alınan aksiyonun Q değerini seç
        current_q = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        # ── LOSS & BACKPROP ─────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # GRU / RNN'lerde gradient patlamasını önlemek için kırpma işlemi (Kritik!)
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 100)

        self.optimizer.step()

        return loss.item()

    # EPISODE SONU
    def on_episode_end(self) -> None:
        """
        Episode bittiğinde hiperparametre güncellemelerini yapar.
        """
        self.episode_count += 1

        # Epsilon'u kademeli olarak azalt
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

        # Belirli periyotlarda hedef ağı güncelle
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            print(f"  [GRU Target Güncellendi] Episode: {self.episode_count} | Epsilon: {self.epsilon:.3f}")

    # KAYDET / YÜKLE
    def save(self, path: str) -> None:
        """Eğitilmiş modeli ve eğitim durumunu diske kaydeder."""
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
        }, path)
        print(f"[GRU Modeli Kaydedildi] {path}")

    def load(self, path: str) -> None:
        """Kaydedilmiş modeli tüm optimizasyon süreçleriyle birlikte yükler."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        print(f"[GRU Modeli Yüklendi] {path}")
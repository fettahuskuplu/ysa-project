"""
Borsa İstanbul (BIST) 1D-CNN Double DQN Ajanı
Yazar: Fettah & AI Pair Programmer

Bu modül; Double Deep Q-Network (DDQN) algoritmasını 1D-CNN mimarisi üzerinde 
uygulayan pekiştirmeli öğrenme ajanını tanımlar.
"""

import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cnn.config import CNNDQNConfig
from src.models.cnn.architecture import CNNDQNNetwork

# --- Deneyim Veri Yapısı ---
Experience = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """
    Ajanın geçmiş deneyimlerini (Transitions) saklayan ve rastgele mini-batch 
    örnekleme yapmasını sağlayan Experience Replay yapısı.
    """

    def __init__(self, capacity: int) -> None:
        """
        Replay Buffer'ı başlatır.

        Args:
            capacity (int): Maksimum deneyim saklama kapasitesi.
        """
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Yeni bir deneyimi hafızaya ekler.

        Args:
            state (np.ndarray): Mevcut durum.
            action (int): Alınan aksiyon.
            reward (float): Elde edilen ödül.
            next_state (np.ndarray): Sonraki durum.
            done (bool): Bölümün bitip bitmediği.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Hafızadan rastgele bir mini-batch örnekler ve bunları Numpy dizileri olarak döndürür.

        Args:
            batch_size (int): Mini-batch boyutu.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (states, actions, rewards, next_states, dones) dizileri.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Mevcut doluluk oranını döndürür."""
        return len(self.buffer)


class CNNDQNAgent:
    """
    1D-CNN tabanlı Double DQN ajanı.
    BIST ortamında en yüksek portföy büyümesini elde etmek amacıyla aksiyonları optimize eder.
    """

    def __init__(self, num_features: int, window_size: int) -> None:
        """
        Ajan parametrelerini, ağları, optimizer ve bellek yapılarını kurar.

        Args:
            num_features (int): Girdi öznitelik sayısı (~11).
            window_size (int): Kayan pencere boyutu (30).
        """
        # Konfigürasyon parametreleri
        self.epsilon: float = CNNDQNConfig.EPSILON_START
        self.epsilon_end: float = CNNDQNConfig.EPSILON_END
        self.epsilon_decay: float = CNNDQNConfig.EPSILON_DECAY
        self.gamma: float = CNNDQNConfig.GAMMA
        self.batch_size: int = CNNDQNConfig.BATCH_SIZE
        self.tau: float = CNNDQNConfig.TAU
        self.use_soft_update: bool = CNNDQNConfig.USE_SOFT_UPDATE
        self.target_update_freq: int = CNNDQNConfig.TARGET_UPDATE_FREQUENCY
        self.action_size: int = CNNDQNConfig.ACTION_SIZE
        self.device = torch.device(CNNDQNConfig.DEVICE)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=CNNDQNConfig.MEMORY_SIZE)

        # Dual Network Yapısı (Double DQN)
        # Online Ağ: Karar alan ve sürekli güncellenen ana ağ
        self.online_net = CNNDQNNetwork(
            num_features=num_features,
            window_size=window_size,
            action_size=self.action_size
        ).to(self.device)

        # Target Ağ: Karşılaştırma hedeflerini (Bellman Targets) üreten stabil ağ
        self.target_net = CNNDQNNetwork(
            num_features=num_features,
            window_size=window_size,
            action_size=self.action_size
        ).to(self.device)

        # Başlangıçta Online ağ ağırlıklarını Target ağa kopyala
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target ağ sadece forward pass yapar, eğitim/dropout kapalı

        # Optimizer ve Loss Fonksiyonu (Stabilite için Smooth L1 Loss)
        self.optimizer = optim.Adam(
            params=self.online_net.parameters(),
            lr=CNNDQNConfig.LEARNING_RATE
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.episode_count: int = 0

    # ------------------------------------------------------------------
    # Aksiyon Seçimi
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-Greedy politikasıyla aksiyon seçer.

        Args:
            state (np.ndarray): Mevcut ortam durumu (window_size, num_features).

        Returns:
            int: Seçilen aksiyon (0: HOLD, 1: BUY, 2: SELL).
        """
        # Keşif (Exploration): Rastgele aksiyon seç
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Sömürü (Exploitation): En yüksek Q değerine sahip aksiyonu seç
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, window_size, features)
            q_values = self.online_net(state_t)
            action: int = q_values.argmax(dim=1).item()
        self.online_net.train()
        return action

    def select_action_greedy(self, state: np.ndarray) -> int:
        """
        Tamamen greedy aksiyon seçimi — Değerlendirme (Evaluation) adımlarında kullanılır.

        Args:
            state (np.ndarray): Mevcut ortam durumu.

        Returns:
            int: En iyi aksiyon.
        """
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            action: int = q_values.argmax(dim=1).item()
        return action

    # ------------------------------------------------------------------
    # Eğitim Adımı (Replay)
    # ------------------------------------------------------------------
    def train_step(self) -> Optional[float]:
        """
        Experience Replay'den bir mini-batch çekip Double DQN Bellman güncellemesini yapar.

        Returns:
            Optional[float]: Bu adımdaki eğitim kaybı (loss) değeri. Bellek yetersizse None.
        """
        # Bellekte yeterli deneyim yoksa öğrenmeye başlama
        if len(self.replay_buffer) < self.batch_size:
            return None

        # --- 1. Mini-Batch Örnekleme ve Tensör Dönüşümleri ---
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # --- 2. Double DQN Bellman Güncellemesi ---
        # Online ağ, bir sonraki durum s' için en iyi aksiyonu SEÇER (argmax_a' Q_online(s', a'))
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            best_next_actions = next_q_online.argmax(dim=1).unsqueeze(1)  # (batch, 1)

            # Target ağ, online ağın seçtiği aksiyonun Q-değerini HESAPLAR (Q_target(s', best_action))
            next_q_target = self.target_net(next_states_t)
            q_next_eval = next_q_target.gather(1, best_next_actions).squeeze(1)  # (batch,)

            # Bellman Hedefi: y_i = r_i + gamma * Q_target(s_next, argmax Q_online) * (1 - done)
            target_q = rewards_t + self.gamma * q_next_eval * (1.0 - dones_t)

        # --- 3. Mevcut Q-Değeri Tahmini ---
        current_q = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        # --- 4. Loss Hesaplama ve Backpropagation ---
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradyan Patlamasını Önlemek İçin Gradyan Kırpma (Norm-based clipping)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Her adımdan sonra eğer soft update aktifse target ağı güncelle
        if self.use_soft_update:
            self.soft_update_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Target Network Güncellemeleri
    # ------------------------------------------------------------------
    def soft_update_target(self) -> None:
        """
        Target Network ağırlıklarını Online Network ağırlıklarına doğru kademeli olarak kaydırır.
        Formül: θ_target = τ * θ_online + (1 - τ) * θ_target
        """
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update_target(self) -> None:
        """Target network'ü online network ağırlıklarıyla tamamen eşitler (Hard Copy)."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Episode Sonu Yönetimi
    # ------------------------------------------------------------------
    def on_episode_end(self) -> None:
        """
        Her episode bitiminde keşif oranını (epsilon) düşürür ve target ağı (hard ise) günceller.
        """
        self.episode_count += 1

        # Epsilon'u kademeli azalt
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Eğer hard update aktifse periyodik olarak güncelle
        if not self.use_soft_update and self.episode_count % self.target_update_freq == 0:
            self.hard_update_target()

    # ------------------------------------------------------------------
    # Kaydetme / Yükleme
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Modelin online ağ ağırlıklarını ve parametrelerini diske yazar."""
        torch.save(
            {
                "online_net_state": self.online_net.state_dict(),
                "target_net_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Kaydedilmiş kontrol noktasından (checkpoint) modeli geri yükler."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]

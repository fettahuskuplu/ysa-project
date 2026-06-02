import numpy as np
import random
from collections import deque
from typing import Tuple
from config import GRUDQNConfig

class ReplayBuffer:
    """
    Ajanın geçmiş tecrübelerini (deneyimlerini) hafızada tutan ve eğitim sırasında
    rastgele örneklemeler (mini-batch) yaparak öğrenmeyi sağlayan bellek sınıfı.
    (Experience Replay mekanizması)
    """

    def __init__(self):
        # Hafıza kapasitesini config dosyasından alıyoruz
        capacity = GRUDQNConfig.MEMORY_SIZE
        # deque, maxlen aşıldığında en eski verileri otomatik olarak siler (First-In, First-Out)
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Ortamda atılan her bir adımın (step) sonucunu (transition) hafızaya ekler.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Eğitim (train_step) için hafızadan rastgele 'batch_size' kadar deneyim çeker.
        Rastgele çekim yapılması, zaman serilerindeki ardışık korelasyonu kırarak
        sinir ağının (GRU) daha stabil ve genellenebilir öğrenmesini sağlar.
        """
        # Buffer'dan rastgele mini-batch seçimi
        batch = random.sample(self.buffer, batch_size)

        # [(s1, a1, r1...), (s2, a2, r2...)] şeklindeki listeyi ayrıştır (Unzip)
        states, actions, rewards, next_states, dones = zip(*batch)

        # PyTorch tensörlerine dönüştürülmeye hazır numpy dizilerini döndür
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """
        Buffer'da anlık olarak kaç adet deneyim bulunduğunu döndürür.
        """
        return len(self.buffer)
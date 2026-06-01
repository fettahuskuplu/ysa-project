"""
Borsa İstanbul (BIST) 1D-CNN Double DQN — Konfigürasyon Modülü
Yazar: Fettah & AI Pair Programmer

Bu modül; 1D-CNN modelinin ve DQN ajanının hiperparametrelerini merkezi olarak yönetir.
"""

import torch


class CNNDQNConfig:
    # --- Veriseti ve Ortam Parametreleri ---
    STATE_WINDOW_SIZE: int = 30
    ACTION_SIZE: int = 3  # HOLD (0), BUY (1), SELL (2)

    # --- Model Mimarisi Parametreleri ---
    HIDDEN_SIZE: int = 64  # Karar (Dense) katmanlarındaki nöron sayısı

    # --- Eğitim Parametreleri ---
    LEARNING_RATE: float = 0.0007  # Adam optimizer öğrenme hızı (En iyi doğrulanan)
    GAMMA: float = 0.95            # İndirgeme faktörü (Bellman denklemindeki gelecek ödül ağırlığı)
    BATCH_SIZE: int = 64           # Deneyim havuzundan çekilecek mini-batch boyutu
    EPISODES: int = 100            # Toplam eğitim episode (bölüm) sayısı

    # --- Epsilon-Greedy Politikası (Keşif vs. Sömürü) ---
    EPSILON_START: float = 1.0     # Başlangıç keşif olasılığı (Tamamen rastgele aksiyonlar)
    EPSILON_END: float = 0.01      # Ulaşılacak minimum keşif olasılığı
    EPSILON_DECAY: float = 0.995   # Her episode sonundaki epsilon azalma çarpanı

    # --- Experience Replay Buffer ---
    MEMORY_SIZE: int = 5000        # Replay Buffer'ın maksimum saklayabileceği deneyim sayısı

    # --- Target Network Güncelleme Mekanizmaları ---
    # Soft Update (Polyak Averaging) kullanılıyorsa:
    TAU: float = 0.005             # target_net = tau * online_net + (1 - tau) * target_net
    USE_SOFT_UPDATE: bool = True   # Çok daha stabil bir eğitim için Soft Update'i aktif ediyoruz

    # Hard Update kullanılıyorsa (USE_SOFT_UPDATE = False ise devreye girer):
    TARGET_UPDATE_FREQUENCY: int = 10  # Kaç episode'da bir online ağırlıkları target'a kopyalansın

    # --- Donanım Cihaz Yapılandırması ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

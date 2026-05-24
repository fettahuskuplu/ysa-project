"""
Borsa İstanbul (BIST) LSTM Double DQN Ajanı
Yazar: Senior AI Engineer & RL Takımı

Bu modül; Double Deep Q-Network (DDQN) algoritmasını LSTM mimarisi üzerinde
uygulayan RL ajanını tanımlar.

Temel Mekanizmalar:
- Double DQN: Overestimation bias'ı azaltmak için aksiyon seçimi (online) ve
  değer tahmini (target) iki ayrı ağ ile yapılır.
- Soft Update (Polyak Averaging): Target ağ, her replay adımında τ=0.005 ile
  kademeli güncellenir → Hard copy'ye göre çok daha stabil öğrenme sağlar.
- Experience Replay: Geçmiş deneyimleri rastgele örnekleyerek temporal
  korelasyonu kırar ve veri verimliliğini artırır.
- Epsilon-Greedy: Keşif (exploration) ve sömürü (exploitation) dengesini
  kademeli olarak sömürü lehine kaydırır.
"""

import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.models.lstm.architecture import build_lstm_model

# --- Tip Tanımları ---
# Replay Buffer'da saklanan tek bir deneyim kaydı
Experience = Tuple[np.ndarray, int, float, np.ndarray, bool]


class LSTMDQNAgent:
    """
    LSTM tabanlı Double DQN ajanı.

    Bu ajan, BIST piyasa ortamında (TradingEnvironment) alım-satım kararları
    vererek portföy değerini maksimize etmeyi öğrenir.

    Attributes:
        state_shape (Tuple[int, int]): Durum tensörü boyutu (window_size, features).
        action_size (int): Aksiyon uzayı boyutu (HOLD=0, BUY=1, SELL=2).
        gamma (float): İndirgeme faktörü (discount factor) — gelecekteki ödüllerin
                        bugünkü değerini belirler.
        epsilon (float): Mevcut keşif (exploration) olasılığı.
        tau (float): Soft update katsayısı (Polyak averaging).
    """

    def __init__(
        self,
        state_shape: Tuple[int, int],
        action_size: int = 3,
        n_units: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        tau: float = 0.005,
        memory_size: int = 2000,
        dropout_rate: float = 0.25,
    ) -> None:
        """
        LSTM DQN ajanını başlatır.

        Args:
            state_shape: Giriş durumu boyutu (window_size, num_features).
            action_size: Aksiyon uzayı büyüklüğü.
            n_units: LSTM ve Dense katman nöron sayısı.
            learning_rate: Adam optimizer öğrenme hızı.
            gamma: Bellman denklemindeki indirgeme faktörü (0.90 - 0.99).
            epsilon: Başlangıç keşif olasılığı.
            epsilon_min: Minimum keşif olasılığı (alt sınır).
            epsilon_decay: Her replay sonrası epsilon çarpanı.
            tau: Soft update katsayısı (θ_target = τ·θ_online + (1-τ)·θ_target).
            memory_size: Replay Buffer maksimum kapasitesi.
            dropout_rate: Dropout oranı.
        """
        self.state_shape: Tuple[int, int] = state_shape
        self.action_size: int = action_size
        self.gamma: float = gamma
        self.tau: float = tau

        # --- Epsilon-Greedy Parametreleri ---
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay

        # --- Experience Replay Buffer ---
        # deque: FIFO yapısı → Buffer dolduğunda en eski deneyimler otomatik silinir
        self.memory: Deque[Experience] = deque(maxlen=memory_size)

        # --- Dual Network Yapısı (Double DQN) ---
        # Online (Policy) Network: Her adımda güncellenen ana ağ
        self.model: tf.keras.Model = build_lstm_model(
            state_shape=state_shape,
            action_size=action_size,
            n_units=n_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
        )

        # Target Network: Q-değeri hedeflerini hesaplayan stabil referans ağ
        # Not: Target ağ hiçbir zaman .fit() ile eğitilmez (sadece predict + set_weights).
        # Bu yüzden compile adımı atlanarak gereksiz optimizer/loss overhead'i engellenir.
        self.target_model: tf.keras.Model = build_lstm_model(
            state_shape=state_shape,
            action_size=action_size,
            n_units=n_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
        )

        # İlk senkronizasyon: Target ağ, Online ağ ile aynı ağırlıklarla başlar
        self.target_model.set_weights(self.model.get_weights())

    # ------------------------------------------------------------------
    # Aksiyon Seçimi
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        """
        Epsilon-Greedy politikasıyla aksiyon seçer.

        ε olasılıkla rastgele bir aksiyon seçilir (keşif/exploration),
        (1-ε) olasılıkla en yüksek Q-değerine sahip aksiyon seçilir (sömürü/exploitation).

        Args:
            state: Mevcut ortam durumu. Boyut: (window_size, features).

        Returns:
            int: Seçilen aksiyon indeksi (0: HOLD, 1: BUY, 2: SELL).
        """
        # Keşif: Rastgele aksiyon
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Sömürü: Q-değerlerinden en iyisini seç
        # Shape düzeltmesi: (window_size, features) → (1, window_size, features)
        # model() doğrudan çağrısı predict()'e göre ~3x hızlıdır (batch overhead yok).
        state_batch: np.ndarray = np.expand_dims(state, axis=0)
        q_values: np.ndarray = self.model(state_batch, training=False).numpy()

        return int(np.argmax(q_values[0]))

    def act_greedy(self, state: np.ndarray) -> int:
        """
        Tamamen greedy (açgözlü) aksiyon seçimi — değerlendirme (evaluation) için.

        Epsilon kullanmadan doğrudan en yüksek Q-değerli aksiyonu döndürür.

        Args:
            state: Mevcut ortam durumu. Boyut: (window_size, features).

        Returns:
            int: En yüksek Q-değerine sahip aksiyon indeksi.
        """
        state_batch: np.ndarray = np.expand_dims(state, axis=0)
        q_values: np.ndarray = self.model(state_batch, training=False).numpy()

        return int(np.argmax(q_values[0]))

    # ------------------------------------------------------------------
    # Deneyim Yönetimi
    # ------------------------------------------------------------------
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Bir deneyim kaydını (transition) Replay Buffer'a ekler.

        Args:
            state: Mevcut durum (s_t).
            action: Alınan aksiyon (a_t).
            reward: Elde edilen ödül (r_t).
            next_state: Sonraki durum (s_{t+1}).
            done: Episode'un bitip bitmediği.
        """
        self.memory.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    # Öğrenme (Replay)
    # ------------------------------------------------------------------
    def replay(self, batch_size: int = 32) -> Optional[float]:
        """
        Experience Replay ile mini-batch öğrenme adımı gerçekleştirir.

        Double DQN Bellman Güncellemesi:
            Q_target(s, a) = r + γ · Q_target(s', argmax_a' Q_online(s', a'))

        - Online ağ (self.model): En iyi aksiyonu SEÇer (argmax)
        - Target ağ (self.target_model): Bu aksiyonun Q-değerini DEĞERLENDİRİR
        - Bu ayrışma, standart DQN'deki Q-değeri aşırı tahminini (overestimation) engeller.

        Args:
            batch_size: Mini-batch boyutu.

        Returns:
            Optional[float]: Eğitim kaybı (loss) değeri. Buffer yetersizse None.
        """
        # Buffer'da yeterli deneyim yoksa öğrenme yapma
        if len(self.memory) < batch_size:
            return None

        # --- 1. Rastgele Mini-Batch Örnekleme ---
        # Temporal korelasyonu kırmak için rastgele seçim yapılır
        minibatch: List[Experience] = random.sample(self.memory, batch_size)

        # Vektörize işlem için batch tensörleri oluştur
        states: np.ndarray = np.array([exp[0] for exp in minibatch])
        actions: np.ndarray = np.array([exp[1] for exp in minibatch], dtype=np.int32)
        rewards: np.ndarray = np.array([exp[2] for exp in minibatch], dtype=np.float32)
        next_states: np.ndarray = np.array([exp[3] for exp in minibatch])
        dones: np.ndarray = np.array([exp[4] for exp in minibatch], dtype=np.float32)

        # --- 2. Double DQN: İki Aşamalı Q-Değeri Hesaplama ---
        # model() doğrudan çağrısı predict()'e göre ~3x hızlıdır (batch overhead yok).

        # Aşama A: Online ağ ile sonraki durumlarda EN İYİ aksiyonları belirle
        q_next_online: np.ndarray = self.model(next_states, training=False).numpy()
        best_actions: np.ndarray = np.argmax(q_next_online, axis=1)

        # Aşama B: Target ağ ile bu aksiyonların Q-değerlerini hesapla
        q_next_target: np.ndarray = self.target_model(next_states, training=False).numpy()

        # --- 3. Bellman Denklemi (Vektörize) ---
        # done=True ise gelecek ödül sıfırlanır (terminal state)
        # target = r + γ · Q_target(s', argmax Q_online(s', ·))
        batch_indices: np.ndarray = np.arange(batch_size)
        target_q: np.ndarray = (
            rewards
            + (1.0 - dones)
            * self.gamma
            * q_next_target[batch_indices, best_actions]
        )

        # --- 4. Mevcut Q-Değerlerini Güncelle ---
        # Sadece alınan aksiyona ait Q-değeri güncellenir, diğerleri sabit kalır
        q_current: np.ndarray = self.model(states, training=False).numpy()
        q_current[batch_indices, actions] = target_q

        # --- 5. Online Ağı Eğit ---
        # epochs=1: Her replay adımında tek geçiş → Ajanın istikrarını korur
        history: tf.keras.callbacks.History = self.model.fit(
            states, q_current, epochs=1, verbose=0
        )

        # --- 6. Epsilon Decay ---
        # Eğitim ilerledikçe rastgeleliği kademeli azalt
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        loss: float = float(history.history["loss"][0])
        return loss

    # ------------------------------------------------------------------
    # Target Network Güncellemesi
    # ------------------------------------------------------------------
    def soft_update_target(self) -> None:
        """
        Soft Update (Polyak Averaging) ile Target Network'ü günceller.

        Formül:
            θ_target = τ · θ_online + (1 - τ) · θ_target

        τ (tau) genellikle 0.001 - 0.01 arasında küçük bir değerdir.
        Bu yaklaşım, hard copy'ye (tam ağırlık kopyalama) kıyasla çok daha
        pürüzsüz ve stabil bir öğrenme dinamiği sağlar. Target ağ, online ağı
        yavaşça takip eder ve ani parametre değişimlerinden etkilenmez.
        """
        online_weights: list = self.model.get_weights()
        target_weights: list = self.target_model.get_weights()

        updated_weights: list = [
            self.tau * online_w + (1.0 - self.tau) * target_w
            for online_w, target_w in zip(online_weights, target_weights)
        ]

        self.target_model.set_weights(updated_weights)

    # ------------------------------------------------------------------
    # Model Kaydetme / Yükleme
    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        """
        Online ağın ağırlıklarını diske kaydeder.

        Args:
            filepath: Kayıt dosya yolu (Örn: 'saved_models/lstm_best.keras').
        """
        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        """
        Daha önce kaydedilmiş ağırlıkları yükler ve target ağı senkronize eder.

        Args:
            filepath: Yüklenecek model dosyasının yolu.
        """
        self.model = tf.keras.models.load_model(filepath)
        self.target_model.set_weights(self.model.get_weights())

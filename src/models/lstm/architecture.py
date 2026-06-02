"""
Bu modül; Stacked LSTM (çok katmanlı) tabanlı bir Deep Q-Network (DQN) mimarisini tanımlar.
BIST piyasa verilerindeki zamansal bağımlılıkları öğrenmek için tasarlanmıştır.

Mimari Kararları:
- Stacked LSTM: İlk katman return_sequences=True ile zaman bilgisini korur, ikinci katman
  bu bilgiyi sabit boyutlu bir temsile (fixed-length representation) sıkıştırır.
- Huber Loss: BIST'teki ani fiyat hareketlerinin (outlier) gradient patlamasına
  yol açmasını engeller. MSE'ye göre uç değerlere daha dayanıklıdır.
- Dropout: Katmanlar arasında %25 oranında nöron devre dışı bırakılarak
  aşırı öğrenme (overfitting) riski azaltılır.
"""

import tensorflow as tf
from typing import Tuple


def build_lstm_model(
    state_shape: Tuple[int, int],
    action_size: int,
    n_units: int = 32,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.25,
) -> tf.keras.Model:
    """
    LSTM tabanlı Q-Network modelini oluşturur ve derler.

    Mimari:
        Input(batch, window_size, features)
          → LSTM(n_units, return_sequences=True)
          → Dropout(dropout_rate)
          → LSTM(n_units, return_sequences=False)
          → Dropout(dropout_rate)
          → Dense(n_units, relu)
          → Dense(action_size, linear)  ← Q-değerleri

    Args:
        state_shape (Tuple[int, int]): Giriş tensörünün boyutu (window_size, num_features).
                                       Örnek: (30, 12) → 30 günlük pencere, 12 öznitelik.
        action_size (int): Çıktı katmanındaki nöron sayısı (aksiyon uzayı boyutu).
                           BIST ortamı için 3: HOLD, BUY, SELL.
        n_units (int): LSTM ve Dense katmanlarındaki nöron sayısı (En iyi doğrulanan: 32).
        learning_rate (float): Adam optimizer öğrenme hızı (En iyi doğrulanan: 0.0007).
        dropout_rate (float): Dropout oranı (0.20 - 0.30 arası önerilir).

    Returns:
        tf.keras.Model: Derlenmiş (compiled) ve eğitime hazır Keras modeli.
    """
    model = tf.keras.Sequential(
        [
            # --- Giriş Katmanı ---
            tf.keras.layers.Input(shape=state_shape, name="market_state_input"),

            # --- Stacked LSTM Bloku ---
            # Katman 1: return_sequences=True → Zaman boyutunu koruyarak
            # her adımdaki gizli durumu (hidden state) bir sonraki LSTM'e aktarır.
            tf.keras.layers.LSTM(
                units=n_units,
                return_sequences=True,
                name="lstm_temporal_encoder",
            ),
            tf.keras.layers.Dropout(rate=dropout_rate, name="dropout_1"),

            # Katman 2: return_sequences=False → Tüm zaman serisini tek bir
            # sabit boyutlu vektöre sıkıştırır (sequence-to-vector).
            tf.keras.layers.LSTM(
                units=n_units,
                return_sequences=False,
                name="lstm_context_compressor",
            ),
            tf.keras.layers.Dropout(rate=dropout_rate, name="dropout_2"),

            # --- Karar Katmanları (Decision Head) ---
            # Doğrusal olmayan (non-linear) ara katman: LSTM çıktısından
            # aksiyonlara geçişte ek soyutlama kapasitesi sağlar.
            tf.keras.layers.Dense(
                units=n_units,
                activation="relu",
                name="decision_dense",
            ),

            # Çıktı: Her aksiyon için bir Q-değeri (lineer aktivasyon).
            # Q(s, a) ≈ beklenen kümülatif ödül
            tf.keras.layers.Dense(
                units=action_size,
                activation="linear",
                name="q_values_output",
            ),
        ],
        name="LSTM_DQN",
    )

    # --- Derleme (Compilation) ---
    # Huber Loss (delta=1.0): |error| < delta → MSE, |error| >= delta → MAE
    # BIST gibi volatil piyasalarda uç ödül değerlerine karşı gradient stabilitesi sağlar.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
    )

    return model

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics
from config import GRUDQNConfig
from gru_model import GRUDQN

# ---------------- 1. VERİ YÜKLEME ---------------- #
klasor_yolu = "/Users/guzideiremyavuz/Desktop/archive (1)/stocks"
print("Test verileri hazırlanıyor...")

loader = BISTDataLoader(data_dir=klasor_yolu, window_size=GRUDQNConfig.STATE_WINDOW_SIZE, test_split=0.2)
X_train, X_test, prices_train, prices_test = loader.get_pipeline_data('THYAO')

print(f"Test Verisi Boyutu (X_test): {X_test.shape}")

# ---------------- 2. TEST ORTAMI VE MODEL KURULUMU ---------------- #
input_dim = X_test.shape[2]

test_env = TradingEnvironment(
    market_data=X_test,
    real_prices=prices_test,
    initial_balance=10000.0,
    transaction_fee_percent=0.001,
    window_size=GRUDQNConfig.STATE_WINDOW_SIZE
)

model_yolu = os.path.join(os.path.dirname(__file__), "gru_dqn_model.pth")
model = GRUDQN(input_dim, GRUDQNConfig.HIDDEN_SIZE, GRUDQNConfig.ACTION_SIZE, GRUDQNConfig.NUM_LAYERS)

if not os.path.exists(model_yolu):
    raise FileNotFoundError("Eğitilmiş model bulunamadı! Önce gru_env.py dosyasını çalıştırıp modeli eğitin.")

model.load_state_dict(torch.load(model_yolu))
model.eval()

# ---------------- 3. TEST DÖNGÜSÜ ---------------- #
print("\n--- TEST (EVALUATION) BAŞLIYOR... ---")
state = test_env.reset()

while True:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    next_state, reward, done, info = test_env.step(action)
    state = next_state

    if done:
        break

# ---------------- 4. SONUÇ RAPORU ---------------- #
stats = episode_metrics(
    portfolio_history=test_env.portfolio_history,
    reward_history=test_env.reward_history,
    action_history=test_env.action_history,
    initial_balance=test_env.initial_balance
)

print("\n" + "=" * 40)
print("🎯 GRU-DQN TEST PERFORMANS RAPORU 🎯")
print("=" * 40)
print(f"Başlangıç Bakiyesi : {test_env.initial_balance:.2f} TL")
print(f"Final Portföy      : {stats['final_value']:.2f} TL")
print(f"Net Kâr/Zarar      : {stats['final_value'] - test_env.initial_balance:.2f} TL")
print(f"Yüzdesel Getiri    : %{stats['return_pct']:.2f}")
print(f"Sharpe Oranı       : {stats['sharpe_ratio']:.2f} (Risk/Getiri Dengesi)")
print(f"Max Drawdown       : -%{stats['max_drawdown_pct']:.2f} (En büyük düşüş)")
print(f"Toplam İşlem Adedi : {stats['trade_count']}")
print("=" * 40)
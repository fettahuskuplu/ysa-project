import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt  # Grafik kütüphanesi eklendi

# Proje ana dizinini yola ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics
from config import GRUDQNConfig
from gru_model import GRUDQN


def evaluate():
    print("=" * 55)
    print("GRU-DQN - Test (Evaluation) Aşaması")
    print("=" * 55)

    # 1. Veri Yükleme
    klasor_yolu = "/Users/guzideiremyavuz/Desktop/archive (1)/stocks"
    loader = BISTDataLoader(
        data_dir=klasor_yolu,
        window_size=GRUDQNConfig.STATE_WINDOW_SIZE,
        test_split=0.2
    )
    _, X_test, _, prices_test = loader.get_pipeline_data('THYAO')

    print(f"  Test Verisi Boyutu: {X_test.shape}")

    # 2. Test Ortamı ve Model Kurulumu
    input_dim = X_test.shape[2]

    test_env = TradingEnvironment(
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=10000.0,
        transaction_fee_percent=0.001,
        window_size=GRUDQNConfig.STATE_WINDOW_SIZE
    )

    # Cihaz ayarı
    device = torch.device(GRUDQNConfig.DEVICE)
    model = GRUDQN(input_dim, GRUDQNConfig.HIDDEN_SIZE, GRUDQNConfig.ACTION_SIZE, GRUDQNConfig.NUM_LAYERS).to(device)

    model_yolu = os.path.join(os.path.dirname(__file__), "outputs/gru_dqn_best.pth")

    if not os.path.exists(model_yolu):
        raise FileNotFoundError(f"Eğitilmiş model bulunamadı: {model_yolu}\nÖnce train scriptini çalıştırın.")

    checkpoint = torch.load(model_yolu, map_location=device)
    model.load_state_dict(checkpoint["online_net"])
    model.eval()

    # 3. Test Döngüsü
    print("\n  Simülasyon başlatılıyor...")
    state = test_env.reset()

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done, info = test_env.step(action)
        state = next_state

        if done:
            break

    # 4. Sonuç Raporu
    stats = episode_metrics(
        portfolio_history=test_env.portfolio_history,
        reward_history=test_env.reward_history,
        action_history=test_env.action_history,
        initial_balance=test_env.initial_balance
    )

    print("\n" + "=" * 55)
    print("🎯 GRU-DQN TEST PERFORMANS RAPORU 🎯")
    print("=" * 55)
    print(f"  Başlangıç Bakiyesi : {test_env.initial_balance:,.2f} TL")
    print(f"  Final Portföy      : {stats['final_value']:,.2f} TL")
    print(f"  Net Kâr/Zarar      : {stats['final_value'] - test_env.initial_balance:,.2f} TL")
    print(f"  Yüzdesel Getiri    : %{stats['return_pct']:.2f}")
    print(f"  Sharpe Oranı       : {stats['sharpe_ratio']:.2f} (Risk/Getiri Dengesi)")
    print(f"  Max Drawdown       : -%{stats['max_drawdown_pct']:.2f} (En büyük düşüş)")
    print(f"  Toplam İşlem Adedi : {stats['trade_count']}")
    print("=" * 55)

    # ---------------------------------------------------------
    # 5. MATPLOTLIB İLE GRAFİK ÇİZİMİ VE KAYDEDİLMESİ
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))

    # Portföy geçmişi (Yeşil çizgi)
    plt.plot(test_env.portfolio_history, color='forestgreen', linewidth=2, label='GRU-DQN Yapay Zeka Portföyü')

    # Sabit nakit durumu (Kırmızı kesik çizgi)
    plt.axhline(y=test_env.initial_balance, color='crimson', linestyle='--', linewidth=1.5,
                label='Sabit Nakit (Hiçbir Şey Yapmama)')

    # Başlık ve Eksenler
    plt.title(f'GRU-DQN Modelinin Hiç Görmediği Test Verisi Üzerindeki Performansı (Kâr: %{stats["return_pct"]:.2f})',
              fontweight='bold', fontsize=12)
    plt.xlabel('Borsa Günleri (Zaman Adımı)', fontsize=10)
    plt.ylabel('Portföy Toplam Değeri (TL)', fontsize=10)

    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Grafiği proje klasörüne PNG olarak kaydet!
    plt.savefig("bist_gru_test_sonuc.png", dpi=300)

    # Grafiği ekranda göster
    plt.show()


if __name__ == "__main__":
    evaluate()
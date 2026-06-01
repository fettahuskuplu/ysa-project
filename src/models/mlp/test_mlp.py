import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.common.data_loader import BISTDataLoader 
from src.common.environment import TradingEnvironment 
from src.models.mlp.agent import MLPDQNAgent
from src.common.metrics import episode_metrics

def main():
    window_size = 30
    
    # 1. VERİ YÜKLEME VE TEST SETİNİ AYIRMA
    data_loader = BISTDataLoader(data_dir="data", window_size=window_size) 
    # train_mlp'de hangi sembolü seçtiysen (Örn: "ACSEL" veya "THYAO") buraya aynısını yaz
    X_train, X_test, prices_train, prices_test = data_loader.get_pipeline_data(symbol="THYAO")
    
    # KRİTİK: Bu sefer ortamı tamamen HİÇ GÖRMEDİĞİ X_test ve prices_test ile kuruyoruz!
    env = TradingEnvironment(market_data=X_test, real_prices=prices_test, window_size=window_size)
    
    num_features = X_test.shape[2]
    action_size = 3
    
    # 2. AJANI BAŞLATMA VE EN İYİ AĞIRLIKLARI YÜKLEME
    agent = MLPDQNAgent(window_size=window_size, num_features=num_features, action_size=action_size)
    
    # Epsilon'u 0 yapıyoruz çünkü test ortamında rastgele hareket istemiyoruz, tamamen yapay zeka karar verecek!
    agent.epsilon = 0.0 
    
    # Kaydettiğimiz en başarılı 200. episode ağırlık dosyasını yüklüyoruz
    model_path = "mlp_dqn_bist_ep200.pth"
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        print(f"\n🎯 Başarılı model ağırlıkları '{model_path}' dosyası üzerinden başarıyla yüklendi!")
    else:
        print(f"\n❌ HATA: '{model_path}' dosyası ana dizinde bulunamadı! Lütfen dosya adını kontrol et.")
        return

    print("📈 Ajan Hiç Görmediği Canlı Test Ortamında İşlem Yapmaya Başlıyor...")
    print("-" * 75)
    
    # 3. TEST SİMÜLASYONU BAŞLANGICI
    state = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
    # 4. AKADEMİK METRİKLERİN HESAPLANMASI
    metrics = episode_metrics(
        portfolio_history=env.portfolio_history,
        reward_history=env.reward_history,
        action_history=env.action_history,
        initial_balance=env.initial_balance
    )
    
    print("\n📊 --- CANLI TEST ORTAMI SONUÇLARI ---")
    print(f"💰 Başlangıç Sermayesi : {env.initial_balance:,.2f} TL")
    print(f"💵 Final Portföy Değeri: {metrics['final_value']:,.2f} TL")
    print(f"🚀 Net Test Kâr Oranı  : %{metrics['return_pct']:.2f}")
    print(f"📉 Maksimum Düşüş (MaxDD): %{metrics['max_drawdown_pct']:.2f}")
    print(f"⚖️ Sharpe Oranı        : {metrics['sharpe_ratio']:.2f}")
    print(f"🔄 Toplam İşlem Sayısı : {metrics['trade_count']}")
    print("-" * 75)
    
    # 5.PERFORMANS GRAFİĞİ ÇİZDİRME
    plt.figure(figsize=(12, 6))
    plt.plot(env.portfolio_history, label="MLP-DQN Yapay Zeka Portföyü", color="forestgreen", lw=2)
    # Karşılaştırma için başlangıç parasını düz bir çizgi olarak çiziyoruz
    plt.axhline(y=env.initial_balance, color="crimson", linestyle="--", label="Sabit Nakit (Hiçbir Şey Yapmama)")
    
    plt.title(f"MLP-DQN Modelinin Hiç Görmediği Test Verisi Üzerindeki Performansı (Kâr: %{metrics['return_pct']:.2f})", fontsize=12, fontweight='bold')
    plt.xlabel("Borsa Günleri (Zaman Adımı)", fontsize=10)
    plt.ylabel("Portföy Toplam Değeri (TL)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    
    # Grafiği proje klasörüne kaydet
    plt.savefig("bist_mlp_test_sonuc.png", dpi=300, bbox_inches='tight')
    print("📸 Performans grafiği 'bist_mlp_test_sonuc.png' adıyla ana dizine kaydedildi!")
    plt.show()

if __name__ == "__main__":
    main()
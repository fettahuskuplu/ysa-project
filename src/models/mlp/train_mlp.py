import numpy as np
import torch
import random
from collections import deque

# Ortak modüller ve senin yazdığın model/ajan yapısı
from src.common.data_loader import BISTDataLoader 
from src.common.environment import TradingEnvironment 
from src.models.mlp.agent import MLPDQNAgent
# Metrik hesaplama fonksiyonunu ekliyoruz
from src.common.metrics import episode_metrics

def main():
    # ==========================================
    # 1. PARAMETRELERİN TANIMLANMASI
    # ==========================================
    window_size = 30
    num_episodes = 300  
    
    # ==========================================
    # 2. VERİNİN VE ORTAMIN HAZIRLANMASI
    # ==========================================
    # Veri klasörün "data" olarak ayarlandı, pencere boyutu paslandı
    data_loader = BISTDataLoader(data_dir="data", window_size=window_size) 
    
    # THYAO veya hangi hisseyi eğiteceksen sembolü buraya yazabilirsin
    X_train, X_test, prices_train, prices_test = data_loader.get_pipeline_data(symbol="THYAO")
    
    # Ortam (Environment) doğrudan Numpy dizilerini kabul ediyor
    env = TradingEnvironment(market_data=X_train, real_prices=prices_train, window_size=window_size)
    
    # HATAYA KESİN ÇÖZÜM: Öznitelik sayısını doğrudan Numpy dizisinin 3. boyutundan çekiyoruz!
    num_features = X_train.shape[2] 
    action_size = 3  # 0: HOLD, 1: BUY, 2: SELL
    
    # ==========================================
    # 3. AJANIN BAŞLATILMASI
    # ==========================================
    agent = MLPDQNAgent(window_size=window_size, num_features=num_features, action_size=action_size)
    
    print("\n🚀 Eğitim Başlıyor... Toplam Episode:", num_episodes)
    print(f"🎮 Cihaz: {agent.device} | Öznitelik Sayısı: {num_features}")
    print("-" * 75)
    
    # ==========================================
    # 4. ANA EĞİTİM DÖNGÜSÜ (EPISODE DÖNGÜSÜ)
    # ==========================================
    for episode in range(num_episodes):
        state = env.reset() # Her episode başında borsayı sıfırla
        done = False
        ep_losses = []
        
        # Zaman Serisi Adımları (Her bir borsa günü için döngü)
        while not done:
            # Ajan mevcut borsa durumuna bakıp karar verir
            action = agent.get_action(state)
            
            # Kararı borsaya uygular, yeni durumu ve ödülü alır
            next_state, reward, done, info = env.step(action)
            
            # Bu deneyimi hafızasına kaydeder
            agent.remember(state, action, reward, next_state, done)
            
            # Modelini eğitir ve adım adım hatayı (loss) hesaplar
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)
                
            # Target network'ü her adımda pürüzsüzce (Soft Update) güncelliyoruz
            agent.update_target_network()
            
            state = next_state
            
        # ==========================================
        # METRİKLERİN HESAPLANMASI VE RAPORLAMA
        # ==========================================
        # metrics.py fonksiyonunu çağırarak akademik metrikleri üretiyoruz
        metrics = episode_metrics(
            portfolio_history=env.portfolio_history,
            reward_history=env.reward_history,
            action_history=env.action_history,
            initial_balance=env.initial_balance
        )
        
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        
        # Ekran çıktısını tam bir veri bilimci raporuna dönüştürüyoruz
        print(f"Ep: {episode+1:03d}/{num_episodes} | "
              f"Net Kâr: %{metrics['return_pct']:.2f} | "
              f"MaxDD: %{metrics['max_drawdown_pct']:.2f} | "
              f"İşlem: {metrics['trade_count']} | "
              f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
              f"Loss: {avg_loss:.4f}")
        
        # Belirli aralıklarla modeli yedekliyoruz
        if (episode + 1) % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f"mlp_dqn_bist_ep{episode+1}.pth")
            print(f"--> [YEDEK] Model 'mlp_dqn_bist_ep{episode+1}.pth' olarak kaydedildi.")

    print("-" * 75)
    print("🎉 TEBRİKLER! Eğitim Başarıyla Tamamlandı, Modelin Borsayı Öğrendi!")

if __name__ == "__main__":
    main()
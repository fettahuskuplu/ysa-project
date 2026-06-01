import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.common.data_loader import BISTDataLoader 
from src.common.environment import TradingEnvironment 
from src.models.mlp.agent import MLPDQNAgent
from src.common.metrics import episode_metrics

def main():
    window_size = 30  # Optimizasyondan çıkan en iyi window değeri
    symbol = "THYAO"
    
    data_loader = BISTDataLoader(data_dir="data", window_size=window_size) 
    X_train, X_test, prices_train, prices_test = data_loader.get_pipeline_data(symbol=symbol)
    
    env = TradingEnvironment(market_data=X_test, real_prices=prices_test, window_size=window_size)
    num_features = X_test.shape[2]
    action_size = 3
    
    agent = MLPDQNAgent(window_size=window_size, num_features=num_features, action_size=action_size)
    agent.epsilon = 0.0 
    
   # 🎯 train_mlp.py tarafından kaydedilen gerçek model adıyla eşitliyoruz:
    model_path = os.path.join("outputs", "mlp_dqn_best.pth")
    
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        print(f"\n🎯 Başarılı model ağırlıkları JSON tabanlı '{model_path}' üzerinden başarıyla yüklendi!")
    else:
        print(f"\n❌ HATA: '{model_path}' bulunamadı! Lütfen train işlemini tamamla.")
        return

    print("📈 Ajan Canlı Test Ortamında İşlem Yapmaya Başlıyor...")
    print("-" * 75)
    
    state = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
    metrics = episode_metrics(
        portfolio_history=env.portfolio_history,
        reward_history=env.reward_history,
        action_history=env.action_history,
        initial_balance=env.initial_balance
    )
    
    print("\n📊 --- CANLI TEST ORTAMI SONUÇLARI ---")
    print(f"🚀 Net Test Kâr Oranı  : %{metrics['return_pct']:.2f}")
    print(f"📉 Maksimum Düşüş (MaxDD): %{metrics['max_drawdown_pct']:.2f}")
    print(f"⚖️ Sharpe Oranı        : {metrics['sharpe_ratio']:.2f}")
    print(f"🔄 Toplam İşlem Sayısı : {metrics['trade_count']}")
    print("-" * 75)

if __name__ == "__main__":
    main()
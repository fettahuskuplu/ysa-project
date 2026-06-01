import os
import sys
import numpy as np
import torch

# --- Proje Kök Dizinini Python Path'e Ekle ---
# Bu blok `src.common.*` modüllerinin terminalden çağrıldığında doğru import edilmesini sağlar.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── IMPORTLAR ──
from src.common.data_loader import BISTDataLoader 
from src.common.environment import TradingEnvironment 
from src.models.mlp.agent import MLPDQNAgent
from src.common.metrics import episode_metrics

def run_training_session(X_train, prices_train, window_size, num_features, num_episodes, lr, gamma):
    action_size = 3
    env = TradingEnvironment(market_data=X_train, real_prices=prices_train, window_size=window_size)
    agent = MLPDQNAgent(window_size=window_size, num_features=num_features, action_size=action_size, lr=lr, gamma=gamma)
    
    best_sharpe = -np.inf
    best_metrics = {}

    # 🎯 HER EPISODE İÇİN CANLI ÇIKTI MEKANİZMASI
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        ep_losses = []
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)
                
            agent.update_target_network()
            state = next_state
            
        metrics = episode_metrics(
            portfolio_history=env.portfolio_history,
            reward_history=env.reward_history,
            action_history=env.action_history,
            initial_balance=env.initial_balance
        )
        
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        
        # 🎯 Kombinasyonun içindeki her adımı ekrana basıyoruz
        print(f"   ↳ Ep: {episode+1:02d}/{num_episodes} | "
              f"Net Kâr: %{metrics['return_pct']:.2f} | "
              f"MaxDD: %{metrics['max_drawdown_pct']:.2f} | "
              f"İşlem: {metrics['trade_count']} | "
              f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
              f"Loss: {avg_loss:.4f}")
        
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_metrics = metrics

    return agent, best_sharpe, best_metrics

def main():
    # Grid Search Parametreleri
    grid_window_sizes = [20, 30]
    grid_learning_rates = [0.001, 0.0005]
    grid_gammas = [0.95, 0.98]
    
    num_episodes = 50  
    symbol = "THYAO"
    
    global_best_sharpe = -np.inf
    global_best_agent = None
    global_best_metrics = {}

    print("\n🔍 MLP Model için Hiperparametre Optimizasyonu Başlatılıyor...")
    print("-" * 75)

    combination_count = 1
    for w_size in grid_window_sizes:
        data_loader = BISTDataLoader(data_dir="data", window_size=w_size) 
        X_train, X_test, prices_train, prices_test = data_loader.get_pipeline_data(symbol=symbol)
        num_features = X_train.shape[2]

        for lr in grid_learning_rates:
            for gamma in grid_gammas:
                print(f"\n🎲 [{combination_count}/8] DENENİYOR -> Window: {w_size} | LR: {lr} | Gamma: {gamma}")
                print("   " + "-" * 60)
                
                agent, session_best_sharpe, session_metrics = run_training_session(
                    X_train, prices_train, w_size, num_features, num_episodes, lr, gamma
                )
                
                print("   " + "-" * 60)
                print(f"   🌟 Kombinasyon En İyi Sonucu -> Sharpe: {session_best_sharpe:.4f} | En Yüksek Kâr: %{session_metrics['return_pct']:.2f}")
                
                if session_best_sharpe > global_best_sharpe:
                    global_best_sharpe = session_best_sharpe
                    global_best_agent = agent
                    global_best_metrics = session_metrics
                
                combination_count += 1

    print("\n" + "=" * 75)
    print(f"🏆 Tüm Optimizasyon Tamamlandı! En İyi Genel Sharpe: {global_best_sharpe:.4f}")
    print(f"📈 En Başarılı Kombinasyonun Kâr Oranı: %{global_best_metrics['return_pct']:.2f}")
    print("=" * 75)
    
    os.makedirs("outputs", exist_ok=True)
    pt_save_path = "outputs/mlp_dqn_best.pth"
    torch.save(global_best_agent.policy_net.state_dict(), pt_save_path)
    print(f"--> 💾 [LOCAL SAVE] Ağır model dosyası güvende: '{pt_save_path}'")

if __name__ == "__main__":
    main()
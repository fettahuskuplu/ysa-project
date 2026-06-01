import sys 
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from config import DuelingDQNConfig
from agent import DuelingDQNAgent

def train():
    print("=" * 55)
    print("Dueling DQN - Eğitim Başlıyor")
    print("=" * 55)

    # 1. Veriyi yükle
    print("\n[1/3] Veri Yükleniyor...")

    loader = BISTDataLoader(
        window_size=DuelingDQNConfig.STATE_WINDOW_SIZE,
    )
    X_train, X_test, prices_train, prices_test = loader.get_pipeline_data("THYAO")

    input_size = X_train.shape[1] * X_train.shape[2]

    print(f" X_train boyutu : {X_train.shape}")
    print(f" X_test boyutu  : {X_test.shape}")
    print(f" input_size     : {input_size}")

    # 2. Ajanı Oluştur
    print("\n[2/3] Ajan Oluşturuluyor...")

    agent = DuelingDQNAgent(input_size=input_size)

    print(f"  Cihaz    : {agent.device}")
    print(f"  Epsilon  : {agent.epsilon}")
    print(f"  Episodes : {DuelingDQNConfig.EPISODES}")

    # 3. Eğitim Döngüsü
    # Her adımda: aksiyon seç → ortamda uygula → buffer'a ekle → eğit.
    print("\n[3/3] Eğitim Başlıyor...\n")

    best_portfolio = 0.0
    episode_rewards = []

    for episode in range(1, DuelingDQNConfig.EPISODES + 1):
        # Her episode başında ortamı sıfırlıyoruz
        env = TradingEnvironment(
            market_data=X_train,
            real_prices=prices_train,
            initial_balance=10_000.0,
            window_size=DuelingDQNConfig.STATE_WINDOW_SIZE,
        )
        state = env.reset()
        done = False
        total_reward = 0.0

        # Adım döngüsü — episode bitene kadar devam et
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        
        # Episode bitti, epsilon azaltılacak
        agent.on_episode_end()
        episode_rewards.append(total_reward)

        # En iyi model kaydediliyor
        final_portfolio = env.portfolio_history[-1]
        if final_portfolio > best_portfolio:
            best_portfolio = final_portfolio
            os.makedirs("outputs", exist_ok=True)
            agent.save("outputs/dueling_dqn_best.pth")

        if episode % 10 == 0 or episode == 1:
            print(
                f"  Episode {episode:3d}/{DuelingDQNConfig.EPISODES} | "
                f"Ödül: {total_reward:8.2f} | "
                f"Portföy: {final_portfolio:8.0f} TL | "
                f"Epsilon: {agent.epsilon:.3f}"
            ) 
        
    print("\n" + "=" * 55)
    print("  Eğitim tamamlandı.")
    print(f"  En iyi portföy : {best_portfolio:,.0f} TL")
    print(f"  Model kaydı    : outputs/dueling_dqn_best.pth")
    print("=" * 55)

if __name__ == "__main__":
    train()

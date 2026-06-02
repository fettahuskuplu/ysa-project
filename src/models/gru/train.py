import sys
import os

# Proje ana dizinini yola ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics
from config import GRUDQNConfig
from gru_agent import GRUDQNAgent


def train():
    print("=" * 55)
    print("GRU-DQN - Eğitim Başlıyor")
    print("=" * 55)

    # 1. Veri Yükleme
    print("\n[1/3] Veri Yükleniyor...")
    klasor_yolu = "/Users/guzideiremyavuz/Desktop/archive (1)/stocks"

    loader = BISTDataLoader(
        data_dir=klasor_yolu,
        window_size=GRUDQNConfig.STATE_WINDOW_SIZE,
        test_split=0.2
    )
    X_train, X_test, prices_train, prices_test = loader.get_pipeline_data('THYAO')

    # Dueling DQN'in aksine flatten yapmıyoruz, GRU için sadece feature sayısını (input_dim) veriyoruz.
    input_dim = X_train.shape[2]

    print(f" X_train boyutu : {X_train.shape}")
    print(f" input_dim      : {input_dim}")

    # 2. Ajan Oluşturma
    print("\n[2/3] Ajan Oluşturuluyor...")
    agent = GRUDQNAgent(input_size=input_dim)

    print(f"  Cihaz    : {agent.device}")
    print(f"  Epsilon  : {agent.epsilon}")
    print(f"  Episodes : {GRUDQNConfig.EPISODES}")

    # 3. Eğitim Döngüsü
    print("\n[3/3] Eğitim Döngüsü Başlıyor...\n")
    best_portfolio = 0.0

    for episode in range(1, GRUDQNConfig.EPISODES + 1):
        env = TradingEnvironment(
            market_data=X_train,
            real_prices=prices_train,
            initial_balance=10000.0,
            transaction_fee_percent=0.001,
            window_size=GRUDQNConfig.STATE_WINDOW_SIZE
        )

        state = env.reset()
        done = False
        total_loss = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Deneyimi buffer'a ekle ve eğit
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            if loss is not None:
                total_loss += loss
                steps += 1

            state = next_state

        # Episode sonu işlemleri (Epsilon azaltma, target net güncelleme)
        agent.on_episode_end()
        avg_loss = total_loss / steps if steps > 0 else 0

        # Metriklerin hesaplanması
        stats = episode_metrics(
            portfolio_history=env.portfolio_history,
            reward_history=env.reward_history,
            action_history=env.action_history,
            initial_balance=env.initial_balance
        )

        final_portfolio = stats['final_value']

        # En iyi modeli kaydetme
        if final_portfolio > best_portfolio:
            best_portfolio = final_portfolio
            os.makedirs("outputs", exist_ok=True)
            # Ekip standardına uygun isimlendirme
            agent.save("outputs/gru_dqn_best.pth")

        # Loglama
        if episode % 5 == 0 or episode == 1:
            print(
                f"  Episode {episode:3d}/{GRUDQNConfig.EPISODES} | "
                f"Portföy: {final_portfolio:8.2f} TL | "
                f"Sharpe: {stats['sharpe_ratio']:5.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {agent.epsilon:.3f}"
            )

    print("\n" + "=" * 55)
    print("  Eğitim Tamamlandı.")
    print(f"  En İyi Portföy : {best_portfolio:,.2f} TL")
    print(f"  Model Kaydı    : outputs/gru_dqn_best.pth")
    print("=" * 55)


if __name__ == "__main__":
    train()
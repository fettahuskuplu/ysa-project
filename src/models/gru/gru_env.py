import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics
from config import GRUDQNConfig
from gru_agent import GRUDQNAgent


def train():
    klasor_yolu = "/Users/guzideiremyavuz/Desktop/archive (1)/stocks"
    print("Veriler masaüstünden okunuyor ve işleniyor. Lütfen bekleyin...")

    loader = BISTDataLoader(data_dir=klasor_yolu, window_size=GRUDQNConfig.STATE_WINDOW_SIZE, test_split=0.2)
    X_train, X_test, prices_train, prices_test = loader.get_pipeline_data('THYAO')

    input_dim = X_train.shape[2]
    agent = GRUDQNAgent(input_size=input_dim)

    print("\n--- GRU-DQN EĞİTİMİ BAŞLIYOR... ---")
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
        total_loss = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                steps += 1

            state = next_state

        agent.on_episode_end()
        avg_loss = total_loss / steps if steps > 0 else 0

        stats = episode_metrics(
            portfolio_history=env.portfolio_history,
            reward_history=env.reward_history,
            action_history=env.action_history,
            initial_balance=env.initial_balance
        )

        final_portfolio = stats['final_value']

        if final_portfolio > best_portfolio:
            best_portfolio = final_portfolio
            os.makedirs("outputs", exist_ok=True)
            # Modeli kendi istediğin isimle kaydediyoruz
            agent.save("gru_dqn_model.pth")

        print(f"Bölüm {episode}/{GRUDQNConfig.EPISODES} | "
              f"Bakiye: {final_portfolio:.2f} TL | "
              f"Getiri: %{stats['return_pct']:.2f} | "
              f"Sharpe: {stats['sharpe_ratio']:.2f} | "
              f"Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f}")

    print("\nEğitim Tamamlandı! En iyi model 'gru_dqn_model.pth' olarak kaydedildi.")


if __name__ == "__main__":
    train()
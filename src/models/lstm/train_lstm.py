"""
Borsa İstanbul (BIST) LSTM Double DQN — Eğitim Modülü
Yazar: Senior AI Engineer & RL Takımı

Bu modül; LSTM-DQN ajanının episode tabanlı eğitim döngüsünü yönetir.

Eğitim Stratejisi:
- Episode tabanlı eğitim: Her episode'da ajan tüm train verisini baştan sona gezer.
- Replay: Her N adımda mini-batch replay yapılır (epochs=1 → istikrar).
- Soft Update: Her replay sonrası target ağ τ=0.005 ile kademeli güncellenir.
- Anti-Data Leakage: Eğitim sadece train seti üzerinde yapılır. Test seti yalnızca
  final değerlendirmede kullanılır.
"""

import os
import sys
import argparse
import logging
from typing import Any, Dict, Tuple

# TensorFlow uyarılarını bastır (import'tan ÖNCE ayarlanmalı)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# --- Proje Kök Dizinini Python Path'e Ekle ---
# Bu, `src.common.*` modüllerinin doğru şekilde import edilmesini sağlar.
_PROJECT_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics
from src.models.lstm.agent import LSTMDQNAgent

# --- Logging Yapılandırması ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("LSTM-DQN")


# ======================================================================
# Eğitim Döngüsü
# ======================================================================
def train_agent(
    agent: LSTMDQNAgent,
    env: TradingEnvironment,
    n_episodes: int = 50,
    batch_size: int = 32,
    replay_every: int = 4,
) -> Dict[str, Any]:
    """
    LSTM-DQN ajanını episode tabanlı olarak eğitir.

    Her episode'da:
    1. Ortam sıfırlanır (env.reset).
    2. Ajan her adımda epsilon-greedy aksiyon seçer.
    3. Deneyim Replay Buffer'a eklenir.
    4. Mini-batch replay ile online ağ güncellenir (epochs=1).
    5. Soft update ile target ağ kademeli güncellenir.
    6. Episode sonunda performans metrikleri hesaplanır.

    Args:
        agent: Eğitilecek LSTM DQN ajanı.
        env: BIST trading simülasyon ortamı.
        n_episodes: Toplam episode sayısı (50-150 arası önerilir).
        batch_size: Replay mini-batch boyutu.
        replay_every: Kaç adımda bir replay yapılacağı (varsayılan: 4).
                      Her adımda replay yapmak CPU'da çok yavaştır.
                      DQN literatüründe 4 standart değerdir (Mnih et al. 2015).

    Returns:
        Dict[str, Any]: Son episode'un performans metrikleri ve eğitim geçmişi.
            - 'best_sharpe': Eğitim boyunca görülen en iyi Sharpe Ratio.
            - 'final_metrics': Son episode'un metrik sözlüğü.
            - 'sharpe_history': Tüm episode'ların Sharpe Ratio geçmişi.
    """
    best_sharpe: float = -np.inf
    sharpe_history: list[float] = []
    final_metrics: Dict[str, float] = {}

    for episode in range(1, n_episodes + 1):
        state: np.ndarray = env.reset()
        done: bool = False
        step_count: int = 0

        # --- Episode İçi Döngü ---
        while not done:
            # 1. Aksiyon seçimi (epsilon-greedy)
            action: int = agent.act(state)

            # 2. Ortamda bir adım at
            next_state: np.ndarray
            reward: float
            info: Dict[str, Any]
            next_state, reward, done, info = env.step(action)

            # 3. Deneyimi hafızaya kaydet
            agent.remember(state, action, reward, next_state, done)

            # 4. Replay ile öğren + Soft Update
            # Her adımda replay yapmak yerine, her N adımda bir yapılır.
            # Bu, DQN literatüründeki standart yaklaşımdır (replay_every=4).
            # CPU'da LSTM forward pass maliyetini ~%75 azaltır.
            if len(agent.memory) > batch_size and step_count % replay_every == 0:
                agent.replay(batch_size)
                agent.soft_update_target()

            state = next_state
            step_count += 1

        # --- Episode Sonu: Performans Değerlendirmesi ---
        metrics: Dict[str, float] = episode_metrics(
            portfolio_history=env.portfolio_history,
            reward_history=env.reward_history,
            action_history=env.action_history,
            initial_balance=env.initial_balance,
        )

        current_sharpe: float = metrics["sharpe_ratio"]
        sharpe_history.append(current_sharpe)

        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe

        final_metrics = metrics

        # Eğitim durumu logu
        logger.info(
            f"Episode {episode:3d}/{n_episodes} | "
            f"Port: {metrics['final_value']:>10,.2f} TL | "
            f"Getiri: {metrics['return_pct']:>+7.2f}% | "
            f"Sharpe: {current_sharpe:>+6.3f} | "
            f"MDD: {metrics['max_drawdown_pct']:>6.2f}% | "
            f"İşlem: {metrics['trade_count']:>4d} | "
            f"Adım: {step_count:>4d} | "
            f"ε: {agent.epsilon:.4f}"
        )



    return {
        "best_sharpe": best_sharpe,
        "final_metrics": final_metrics,
        "sharpe_history": sharpe_history,
    }


# ======================================================================
# Değerlendirme (Evaluation)
# ======================================================================
def evaluate_agent(
    agent: LSTMDQNAgent,
    market_data: np.ndarray,
    real_prices: np.ndarray,
    initial_balance: float = 10_000.0,
) -> Dict[str, float]:
    """
    Eğitilmiş ajanı greedy politikayla (ε=0) değerlendirir.

    Anti-Data Leakage: Bu fonksiyon sadece test verisi ile çağrılmalıdır.

    Args:
        agent: Eğitilmiş LSTM DQN ajanı.
        market_data: Test seti state matrisi (Samples, Window, Features).
        real_prices: Test seti gerçek kapanış fiyatları.
        initial_balance: Başlangıç bakiyesi.

    Returns:
        Dict[str, float]: Test performans metrikleri.
    """
    env = TradingEnvironment(
        market_data=market_data,
        real_prices=real_prices,
        initial_balance=initial_balance,
    )

    state: np.ndarray = env.reset()
    done: bool = False

    while not done:
        # Greedy aksiyon: Eğitimde öğrenileni uygula, rastgelelik yok
        action: int = agent.act_greedy(state)
        state, _, done, _ = env.step(action)

    metrics: Dict[str, float] = episode_metrics(
        portfolio_history=env.portfolio_history,
        reward_history=env.reward_history,
        action_history=env.action_history,
        initial_balance=initial_balance,
    )

    return metrics



# ======================================================================
# Ana Giriş Noktası
# ======================================================================
def main() -> None:
    """
    LSTM Double DQN eğitim pipeline'ının ana giriş noktası.

    Akış:
    1. CLI argümanlarını oku.
    2. BISTDataLoader ile veri yükle ve hazırla.
    3. Sabit parametrelerle model eğit.
    4. Test seti üzerinde greedy değerlendirme yap.
    5. Modeli diske kaydet.
    """
    # --- CLI Argümanları ---
    parser = argparse.ArgumentParser(
        description="BIST LSTM Double DQN — Eğitim Pipeline'ı"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="THYAO",
        help="Hisse senedi sembolü (Örn: THYAO, GARAN, ASELS)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Toplam episode sayısı (varsayılan: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Replay mini-batch boyutu (varsayılan: 32)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="CSV dosyalarının bulunduğu klasör (varsayılan: data)",
    )
    args: argparse.Namespace = parser.parse_args()

    # --- Sabit Hiperparametreler (Optuna Sonuçlarından) ---
    N_UNITS: int = 32
    LEARNING_RATE: float = 0.0007

    # --- Banner ---
    logger.info("=" * 70)
    logger.info("  BIST LSTM Double DQN — Eğitim Pipeline'ı")
    logger.info(f"  Sembol: {args.symbol} | Episodes: {args.episodes}")
    logger.info(f"  n_units: {N_UNITS} | lr: {LEARNING_RATE}")
    logger.info("=" * 70)

    # =================================================================
    # AŞAMA 1: Veri Yükleme
    # =================================================================
    logger.info("[1/4] Veri yükleniyor ve hazırlanıyor...")

    loader = BISTDataLoader(data_dir=args.data_dir)
    X_train: np.ndarray
    X_test: np.ndarray
    prices_train: np.ndarray
    prices_test: np.ndarray
    X_train, X_test, prices_train, prices_test = loader.get_pipeline_data(args.symbol)

    state_shape: Tuple[int, int] = (X_train.shape[1], X_train.shape[2])

    logger.info(f"  Train seti  : {X_train.shape}  → {len(prices_train)} örnek")
    logger.info(f"  Test seti   : {X_test.shape}  → {len(prices_test)} örnek")
    logger.info(f"  State shape : {state_shape}")

    # =================================================================
    # AŞAMA 2: Model Eğitimi
    # =================================================================
    logger.info("[2/4] Model eğitimi başlatılıyor...")

    agent = LSTMDQNAgent(
        state_shape=state_shape,
        action_size=3,  # HOLD, BUY, SELL
        n_units=N_UNITS,
        learning_rate=LEARNING_RATE,
    )

    env = TradingEnvironment(
        market_data=X_train,
        real_prices=prices_train,
        initial_balance=10_000.0,
    )

    train_result: Dict[str, Any] = train_agent(
        agent=agent,
        env=env,
        n_episodes=args.episodes,
        batch_size=args.batch_size,
    )

    logger.info(f"  ✓ Eğitim tamamlandı! En iyi Sharpe: {train_result['best_sharpe']:+.4f}")

    # =================================================================
    # AŞAMA 3: Test Seti Değerlendirmesi (Anti-Data Leakage)
    # =================================================================
    logger.info("[3/4] Test seti üzerinde değerlendirme yapılıyor...")

    test_metrics: Dict[str, float] = evaluate_agent(
        agent=agent,
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=10_000.0,
    )

    logger.info("=" * 70)
    logger.info("  📊 TEST SETİ SONUÇLARI (Görülmemiş Veri)")
    logger.info(f"    Portföy Değeri : {test_metrics['final_value']:>12,.2f} TL")
    logger.info(f"    Getiri         : {test_metrics['return_pct']:>+10.2f}%")
    logger.info(f"    Sharpe Ratio   : {test_metrics['sharpe_ratio']:>+10.4f}")
    logger.info(f"    Max Drawdown   : {test_metrics['max_drawdown_pct']:>10.2f}%")
    logger.info(f"    İşlem Sayısı   : {test_metrics['trade_count']:>10d}")
    logger.info(f"    Toplam Ödül    : {test_metrics['total_reward']:>+10.2f}")
    logger.info("=" * 70)

    # =================================================================
    # AŞAMA 4: Model Kaydetme
    # =================================================================
    save_dir: str = os.path.join(_PROJECT_ROOT, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    save_path: str = os.path.join(save_dir, "lstm_best.keras")
    agent.save(save_path)
    logger.info(f"[4/4] Model kaydedildi → {save_path}")
    logger.info("Pipeline tamamlandı. ✓")


if __name__ == "__main__":
    main()

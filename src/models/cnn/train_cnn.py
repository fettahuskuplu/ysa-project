"""
Borsa İstanbul (BIST) 1D-CNN Double DQN — Eğitim ve Değerlendirme Pipeline'ı
Yazar: Fettah & AI Pair Programmer

Bu modül; 1D-CNN ajanının eğitim döngüsünü yönetir ve performansı değerlendirir.
"""

import os
import sys

# --- Map pandas_ta to pandas_ta_classic dynamically for standard BISTDataLoader support ---
try:
    import pandas_ta
except ModuleNotFoundError:
    try:
        import pandas_ta_classic as pandas_ta
        sys.modules['pandas_ta'] = pandas_ta
    except ModuleNotFoundError:
        pass

import argparse
import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch

# --- Proje Kök Dizinini Python Path'e Ekle ---
# Bu, `src.common.*` ve `src.models.*` modüllerinin doğru şekilde import edilmesini sağlar.
_PROJECT_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics
from src.models.cnn.config import CNNDQNConfig
from src.models.cnn.agent import CNNDQNAgent

# --- Logging Yapılandırması ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("1D-CNN-DQN")


# ======================================================================
# Eğitim Döngüsü
# ======================================================================
def train_agent(
    agent: CNNDQNAgent,
    env: TradingEnvironment,
    n_episodes: int,
    batch_size: int,
) -> Dict[str, Any]:
    """
    1D-CNN DQN ajanını episode tabanlı olarak eğitir.

    Args:
        agent (CNNDQNAgent): Eğitilecek ajan.
        env (TradingEnvironment): BIST alım-satım ortamı (Train seti).
        n_episodes (int): Eğitim yapılacak toplam episode sayısı.
        batch_size (int): Replay mini-batch boyutu.

    Returns:
        Dict[str, Any]: Eğitim geçmişi ve en iyi sonuçlar.
    """
    best_portfolio_val: float = -np.inf
    sharpe_history: list[float] = []
    final_metrics: Dict[str, float] = {}

    save_dir: str = os.path.join(_PROJECT_ROOT, "outputs")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path: str = os.path.join(save_dir, "cnn_best.pth")

    for episode in range(1, n_episodes + 1):
        state: np.ndarray = env.reset()
        done: bool = False
        step_count: int = 0
        total_loss: float = 0.0
        loss_steps: int = 0

        # --- Episode İçi Döngü ---
        while not done:
            # 1. Epsilon-greedy ile aksiyon seç
            action: int = agent.select_action(state)

            # 2. Ortamda adım at
            next_state: np.ndarray
            reward: float
            info: Dict[str, Any]
            next_state, reward, done, info = env.step(action)

            # 3. Deneyimi hafızaya kaydet
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 4. Öğrenme Adımı (Replay)
            loss: Optional[float] = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_steps += 1

            state = next_state
            step_count += 1

        # Episode sonu Epsilon azaltma ve target güncelleme
        agent.on_episode_end()

        # Performans Metriklerini Hesapla
        metrics: Dict[str, float] = episode_metrics(
            portfolio_history=env.portfolio_history,
            reward_history=env.reward_history,
            action_history=env.action_history,
            initial_balance=env.initial_balance,
        )

        current_sharpe: float = metrics["sharpe_ratio"]
        sharpe_history.append(current_sharpe)
        avg_loss: float = total_loss / loss_steps if loss_steps > 0 else 0.0
        final_portfolio_val: float = metrics["final_value"]

        # En iyi model kontrolü (Portföy Değerine Göre)
        if final_portfolio_val > best_portfolio_val:
            best_portfolio_val = final_portfolio_val
            agent.save(best_model_path)
            logger.info(f"  ★ Yeni En İyi Model Kaydedildi! Portföy: {best_portfolio_val:,.2f} TL")

        final_metrics = metrics

        # Biçimlendirilmiş eğitim durum logu
        logger.info(
            f"Episode {episode:3d}/{n_episodes} | "
            f"Port: {metrics['final_value']:>10,.2f} TL | "
            f"Getiri: {metrics['return_pct']:>+7.2f}% | "
            f"Sharpe: {current_sharpe:>+6.3f} | "
            f"MDD: {metrics['max_drawdown_pct']:>6.2f}% | "
            f"İşlem: {metrics['trade_count']:>4d} | "
            f"Loss: {avg_loss:.4f} | "
            f"ε: {agent.epsilon:.4f}"
        )

    return {
        "best_portfolio_val": best_portfolio_val,
        "final_metrics": final_metrics,
        "sharpe_history": sharpe_history,
    }


# ======================================================================
# Değerlendirme (Evaluation)
# ======================================================================
def evaluate_agent(
    agent: CNNDQNAgent,
    market_data: np.ndarray,
    real_prices: np.ndarray,
    initial_balance: float = 10_000.0,
    window_size: int = 30,
) -> Dict[str, float]:
    """
    Eğitilmiş ajanı greedy politikayla (keşifsiz) test verisi üzerinde değerlendirir.

    Args:
        agent (CNNDQNAgent): Eğitilmiş 1D-CNN DQN ajanı.
        market_data (np.ndarray): Test seti durum verileri.
        real_prices (np.ndarray): Test seti gerçek kapanış fiyatları.
        initial_balance (float): Başlangıç bakiyesi.
        window_size (int): Pencere boyutu.

    Returns:
        Dict[str, float]: Değerlendirme metrikleri.
    """
    env = TradingEnvironment(
        market_data=market_data,
        real_prices=real_prices,
        initial_balance=initial_balance,
        window_size=window_size,
    )

    state: np.ndarray = env.reset()
    done: bool = False

    while not done:
        # Tamamen greedy aksiyon seçimi (keşif yok)
        action: int = agent.select_action_greedy(state)
        state, _, done, _ = env.step(action)

    metrics: Dict[str, float] = episode_metrics(
        portfolio_history=env.portfolio_history,
        reward_history=env.reward_history,
        action_history=env.action_history,
        initial_balance=initial_balance,
    )

    return metrics


# ======================================================================
# Ana Giriş Noktası (CLI)
# ======================================================================
def main() -> None:
    """
    1D-CNN Double DQN eğitim pipeline'ının ana giriş noktası.
    """
    # --- CLI Argümanları ---
    parser = argparse.ArgumentParser(
        description="BIST 1D-CNN Double DQN — Eğitim ve Test Modülü"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="THYAO",
        help="İşlenecek hisse senedi sembolü (Varsayılan: THYAO)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=CNNDQNConfig.EPISODES,
        help=f"Eğitim episode sayısı (Varsayılan: {CNNDQNConfig.EPISODES})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CNNDQNConfig.BATCH_SIZE,
        help=f"Replay batch boyutu (Varsayılan: {CNNDQNConfig.BATCH_SIZE})",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Veri klasörünün yolu (Varsayılan: data)",
    )
    args: argparse.Namespace = parser.parse_args()

    # --- Banner ---
    logger.info("=" * 70)
    logger.info("  📊 BIST 1D-CNN Double DQN — Eğitim Pipeline'ı")
    logger.info(f"  Sembol   : {args.symbol}")
    logger.info(f"  Episodes : {args.episodes} | Batch Size: {args.batch_size}")
    logger.info(f"  Device   : {CNNDQNConfig.DEVICE} | Cfg LR: {CNNDQNConfig.LEARNING_RATE}")
    logger.info("=" * 70)

    # =================================================================
    # AŞAMA 1: Veri Yükleme ve Hazırlık
    # =================================================================
    logger.info(f"[1/4] Veriler yükleniyor (Dizin: {args.data_dir})...")

    loader = BISTDataLoader(
        data_dir=args.data_dir,
        window_size=CNNDQNConfig.STATE_WINDOW_SIZE,
        test_split=0.2,
    )

    try:
        X_train, X_test, prices_train, prices_test = loader.get_pipeline_data(args.symbol)
    except FileNotFoundError as e:
        logger.error(f"Hata: {e}")
        logger.error("Lütfen veri setinizin belirtilen dizinde olduğundan emin olun.")
        logger.error("Masaüstündeki veri seti için komutu çalıştırırken şu parametreyi ekleyin:")
        logger.error('  --data_dir "C:\\Users\\Fettah\\OneDrive\\Masaüstü"')
        sys.exit(1)

    state_shape: Tuple[int, int] = (X_train.shape[1], X_train.shape[2])
    num_features: int = X_train.shape[2]

    logger.info(f"  ✓ Train Seti : {X_train.shape} → {len(prices_train)} gün")
    logger.info(f"  ✓ Test Seti  : {X_test.shape} → {len(prices_test)} gün")
    logger.info(f"  ✓ State Shape: {state_shape} | Öznitelikler: {num_features}")

    # =================================================================
    # AŞAMA 2: Ajan ve Ortam Kurulumu
    # =================================================================
    logger.info("[2/4] Ajan ve borsa ortamı kuruluyor...")

    agent = CNNDQNAgent(
        num_features=num_features,
        window_size=CNNDQNConfig.STATE_WINDOW_SIZE,
    )

    env = TradingEnvironment(
        market_data=X_train,
        real_prices=prices_train,
        initial_balance=10_000.0,
        window_size=CNNDQNConfig.STATE_WINDOW_SIZE,
    )

    # =================================================================
    # AŞAMA 3: Ajan Eğitimi
    # =================================================================
    logger.info("[3/4] Ajan eğitimi başlatılıyor...")

    train_result: Dict[str, Any] = train_agent(
        agent=agent,
        env=env,
        n_episodes=args.episodes,
        batch_size=args.batch_size,
    )

    logger.info(f"  ✓ Eğitim tamamlandı! En iyi portföy değeri: {train_result['best_portfolio_val']:,.2f} TL")

    # =================================================================
    # AŞAMA 4: Test Seti Değerlendirmesi (Görülmemiş Veri)
    # =================================================================
    logger.info("[4/4] Test seti üzerinde greedy değerlendirme yapılıyor...")

    # Test adımı öncesi en iyi kaydedilen modeli yükle
    best_model_path: str = os.path.join(_PROJECT_ROOT, "outputs", "cnn_best.pth")
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        logger.info("  ✓ En iyi ağırlıklar yüklendi, değerlendirme yapılıyor...")

    test_metrics: Dict[str, float] = evaluate_agent(
        agent=agent,
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=10_000.0,
        window_size=CNNDQNConfig.STATE_WINDOW_SIZE,
    )

    logger.info("=" * 70)
    logger.info("  📊 1D-CNN DOUBLE DQN — TEST SETİ DEĞERLENDİRME SONUÇLARI")
    logger.info(f"    Final Portföy Değeri : {test_metrics['final_value']:>12,.2f} TL")
    logger.info(f"    Toplam Net Getiri    : {test_metrics['return_pct']:>+12.2f}%")
    logger.info(f"    Sharpe Oranı         : {test_metrics['sharpe_ratio']:>+12.4f}")
    logger.info(f"    Maksimum Çekilme (MDD): {test_metrics['max_drawdown_pct']:>11.2f}%")
    logger.info(f"    Yapılan İşlem Sayısı : {test_metrics['trade_count']:>12d}")
    logger.info(f"    Biriken Toplam Ödül  : {test_metrics['total_reward']:>+12.2f}")
    logger.info("=" * 70)
    logger.info("Pipeline tamamlandı. ✓ Fettah/1D-CNN Model Ajanı Hazır!")


if __name__ == "__main__":
    main()

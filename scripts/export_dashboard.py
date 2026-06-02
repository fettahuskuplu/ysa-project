"""
Tüm modeller için dashboard JSON export.

Örnekler (proje kökünden):
  python scripts/export_dashboard.py --model dueling_dqn
  python scripts/export_dashboard.py --model all
  python scripts/export_dashboard.py --model mlp_dqn --symbol THYAO
"""
from __future__ import annotations

import sys
try:
    import pandas_ta
except ModuleNotFoundError:
    try:
        import pandas_ta_classic
        sys.modules["pandas_ta"] = pandas_ta_classic
    except ModuleNotFoundError:
        pass

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.dashboard_export import (
    INITIAL_BALANCE,
    WINDOW_SIZE,
    build_payload,
    greedy_rollout,
    load_test_data,
    write_dashboard_json,
)
from src.common.environment import TradingEnvironment

MODEL_REGISTRY = {
    "mlp_dqn": {
        "display": "MLP DQN",
        "weights": ROOT / "outputs" / "mlp_dqn_best.pth",
    },
    "lstm_dqn": {
        "display": "LSTM DQN",
        "weights": ROOT / "saved_models" / "lstm_best.keras",
    },
    "gru_dqn": {
        "display": "GRU DQN",
        "weights": ROOT / "outputs" / "gru_dqn_best.pth",
    },
    "cnn_dqn": {
        "display": "CNN DQN",
        "weights": ROOT / "outputs" / "cnn_best.pth",
    },
    "dueling_dqn": {
        "display": "Dueling DQN",
        "weights": ROOT / "outputs" / "dueling_dqn_best.pth",
    },
}


def _out_path(symbol: str, slug: str) -> Path:
    return ROOT / "outputs" / "dashboard" / symbol.upper() / f"{slug}.json"


def export_mlp_dqn(symbol: str, data_dir: str) -> Path:
    from src.models.mlp.agent import MLPDQNAgent

    meta = MODEL_REGISTRY["mlp_dqn"]
    weights = meta["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Model yok: {weights} — önce quick_train_dashboard.py çalıştırın.")

    _, _, X_test, _, prices_test, df, test_dates = load_test_data(symbol, data_dir)
    num_features = X_test.shape[2]
    agent = MLPDQNAgent(window_size=WINDOW_SIZE, num_features=num_features, action_size=3)
    agent.policy_net.load_state_dict(torch.load(weights, map_location=agent.device))
    agent.epsilon = 0.0

    env = TradingEnvironment(
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=INITIAL_BALANCE,
        window_size=WINDOW_SIZE,
    )
    env = greedy_rollout(env, agent.get_action)
    payload = build_payload(
        env=env,
        df=df,
        test_dates=test_dates,
        prices_test=prices_test,
        model_display_name=meta["display"],
        symbol=symbol,
    )
    return write_dashboard_json(payload, _out_path(symbol, "mlp_dqn"))


def export_dueling_dqn(symbol: str, data_dir: str) -> Path:
    dueling_dir = ROOT / "src" / "models" / "dueling_dqn"
    sys.path.insert(0, str(dueling_dir))
    from agent import DuelingDQNAgent  # noqa: E402
    from config import DuelingDQNConfig  # noqa: E402

    meta = MODEL_REGISTRY["dueling_dqn"]
    weights = meta["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Model yok: {weights}")

    _, _, X_test, _, prices_test, df, test_dates = load_test_data(symbol, data_dir)
    input_size = X_test.shape[1] * X_test.shape[2]
    agent = DuelingDQNAgent(input_size=input_size)
    agent.load(str(weights))
    agent.epsilon = 0.0

    env = TradingEnvironment(
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=INITIAL_BALANCE,
        window_size=DuelingDQNConfig.STATE_WINDOW_SIZE,
    )
    env = greedy_rollout(env, agent.select_action)
    payload = build_payload(
        env=env,
        df=df,
        test_dates=test_dates,
        prices_test=prices_test,
        model_display_name=meta["display"],
        symbol=symbol,
    )
    return write_dashboard_json(payload, _out_path(symbol, "dueling_dqn"))


def export_gru_dqn(symbol: str, data_dir: str) -> Path:
    gru_dir = ROOT / "src" / "models" / "gru"
    sys.path.insert(0, str(gru_dir))
    from gru_agent import GRUDQNAgent  # noqa: E402
    from config import GRUDQNConfig  # noqa: E402

    meta = MODEL_REGISTRY["gru_dqn"]
    weights = meta["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Model yok: {weights}")

    _, _, X_test, _, prices_test, df, test_dates = load_test_data(symbol, data_dir)
    input_size = X_test.shape[2]
    agent = GRUDQNAgent(input_size=input_size)
    agent.online_net.load_state_dict(
        torch.load(weights, map_location=agent.device)
    )
    agent.epsilon = 0.0

    env = TradingEnvironment(
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=INITIAL_BALANCE,
        window_size=GRUDQNConfig.STATE_WINDOW_SIZE,
    )
    env = greedy_rollout(env, agent.select_action)
    payload = build_payload(
        env=env,
        df=df,
        test_dates=test_dates,
        prices_test=prices_test,
        model_display_name=meta["display"],
        symbol=symbol,
    )
    return write_dashboard_json(payload, _out_path(symbol, "gru_dqn"))


def export_cnn_dqn(symbol: str, data_dir: str) -> Path:
    from src.models.cnn.agent import CNNDQNAgent
    from src.models.cnn.config import CNNDQNConfig

    meta = MODEL_REGISTRY["cnn_dqn"]
    weights = meta["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Model yok: {weights}")

    _, _, X_test, _, prices_test, df, test_dates = load_test_data(symbol, data_dir)
    num_features = X_test.shape[2]
    agent = CNNDQNAgent(num_features=num_features, window_size=CNNDQNConfig.STATE_WINDOW_SIZE)
    agent.load(str(weights))
    agent.epsilon = CNNDQNConfig.EPSILON_END

    env = TradingEnvironment(
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=INITIAL_BALANCE,
        window_size=CNNDQNConfig.STATE_WINDOW_SIZE,
    )
    env = greedy_rollout(env, agent.select_action_greedy)
    payload = build_payload(
        env=env,
        df=df,
        test_dates=test_dates,
        prices_test=prices_test,
        model_display_name=meta["display"],
        symbol=symbol,
    )
    return write_dashboard_json(payload, _out_path(symbol, "cnn_dqn"))


def export_lstm_dqn(symbol: str, data_dir: str) -> Path:
    from src.models.lstm.agent import LSTMDQNAgent

    meta = MODEL_REGISTRY["lstm_dqn"]
    weights = meta["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Model yok: {weights}")

    _, _, X_test, _, prices_test, df, test_dates = load_test_data(symbol, data_dir)
    state_shape = (X_test.shape[1], X_test.shape[2])
    agent = LSTMDQNAgent(state_shape=state_shape, action_size=3, n_units=32, learning_rate=0.001)
    agent.load(str(weights))
    agent.epsilon = 0.0

    env = TradingEnvironment(
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=INITIAL_BALANCE,
        window_size=WINDOW_SIZE,
    )
    env = greedy_rollout(env, agent.act_greedy)
    payload = build_payload(
        env=env,
        df=df,
        test_dates=test_dates,
        prices_test=prices_test,
        model_display_name=meta["display"],
        symbol=symbol,
    )
    return write_dashboard_json(payload, _out_path(symbol, "lstm_dqn"))


EXPORTERS = {
    "mlp_dqn": export_mlp_dqn,
    "lstm_dqn": export_lstm_dqn,
    "gru_dqn": export_gru_dqn,
    "cnn_dqn": export_cnn_dqn,
    "dueling_dqn": export_dueling_dqn,
}


def main():
    parser = argparse.ArgumentParser(description="Dashboard JSON export")
    parser.add_argument(
        "--model",
        default="all",
        help="mlp_dqn | lstm_dqn | gru_dqn | cnn_dqn | dueling_dqn | all",
    )
    parser.add_argument("--symbol", default="THYAO")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    slugs = list(EXPORTERS.keys()) if args.model == "all" else [args.model.strip().lower()]

    for slug in slugs:
        if slug not in EXPORTERS:
            print(f" Bilinmeyen model: {slug}")
            continue
        try:
            path = EXPORTERS[slug](args.symbol, args.data_dir)
            print(f"OK [{slug}] -> {path}")
        except Exception as exc:
            print(f"HATA [{slug}]: {exc}")


if __name__ == "__main__":
    main()

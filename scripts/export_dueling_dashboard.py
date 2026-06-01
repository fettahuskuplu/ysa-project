"""
Dueling DQN test seti → dashboard JSON (backend için).
Çalıştırma (proje kökünden):
  python scripts/export_dueling_dashboard.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DUELING_DIR = ROOT / "src" / "models" / "dueling_dqn"
sys.path.insert(0, str(DUELING_DIR))

from config import DuelingDQNConfig
from agent import DuelingDQNAgent

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics

SYMBOL = "THYAO"
MODEL_SLUG = "dueling_dqn"
MODEL_DISPLAY = "Dueling DQN"
WEIGHTS = ROOT / "outputs" / "dueling_dqn_best.pth"
OUT_FILE = ROOT / "outputs" / "dashboard" / SYMBOL / f"{MODEL_SLUG}.json"
DATA_DIR = "data"
INITIAL_BALANCE = 10_000.0


def buy_and_hold_curve(prices: np.ndarray, initial: float) -> list[float]:
    if len(prices) == 0:
        return [initial]
    shares = initial / float(prices[0])
    return [float(shares * p) for p in prices]


def build_action_signals(dates, actions, prices) -> list[dict]:
    tip = {1: "AL", 2: "SAT"}
    out = []
    n = min(len(dates), len(actions), len(prices))
    for i in range(n):
        if actions[i] not in tip:
            continue
        d = dates[i]
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        out.append({
            "tarih": str(d),
            "tip": tip[actions[i]],
            "fiyat": f"{float(prices[i]):.2f}",
        })
    return out

def _fmt_date(d) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]

def _macd_cols(df):
    cols = list(df.columns)
    macd = next((c for c in cols if c.startswith("MACD_") and "MACDh" not in c and "MACDs" not in c), None)
    signal = next((c for c in cols if c.startswith("MACDs_")), None)
    hist = next((c for c in cols if c.startswith("MACDh_")), None)
    return macd, signal, hist

def build_time_series(df, test_dates, action_history) -> list[dict]:
    macd_col, macd_sig_col, macd_hist_col = _macd_cols(df)
    rows = []
    n = min(len(test_dates), len(action_history))
    for i in range(n):
        dt = test_dates[i]
        if dt not in df.index:
            continue
        row = df.loc[dt]
        act = int(action_history[i])
        signal = 1 if act == 1 else (-1 if act == 2 else 0)
        rows.append({
            "time": _fmt_date(dt),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "signal": signal,
            "rsi": float(row["RSI_14"]) if "RSI_14" in row and row["RSI_14"] == row["RSI_14"] else 0.0,
            "macd": float(row[macd_col]) if macd_col and row[macd_col] == row[macd_col] else 0.0,
            "macdSignal": float(row[macd_sig_col]) if macd_sig_col and row[macd_sig_col] == row[macd_sig_col] else 0.0,
            "macdHistogram": float(row[macd_hist_col]) if macd_hist_col and row[macd_hist_col] == row[macd_hist_col] else 0.0,
        })
    return rows

def build_equity_series(test_dates, portfolio_history) -> list[dict]:
    """Portföy eğrisi — gerçek tarihlerle."""
    out = []
    if len(test_dates) == 0 or len(portfolio_history) == 0:
        return out
    out.append({
        "time": _fmt_date(test_dates[0]),
        "value": float(portfolio_history[0]),
    })
    for i in range(len(portfolio_history) - 1):
        if i >= len(test_dates):
            break
        out.append({
            "time": _fmt_date(test_dates[i]),
            "value": float(portfolio_history[i + 1]),
        })
    return out

def build_benchmark_series(test_dates, bist30_values) -> list[dict]:
    """Al-tut eğrisi — aynı tarihler."""
    out = []
    if len(test_dates) == 0:
        return out
    n = min(len(test_dates), len(bist30_values))
    for i in range(n):
        out.append({
            "time": _fmt_date(test_dates[i]),
            "value": float(bist30_values[i]),
        })
    return out

def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Model yok: {WEIGHTS}")

    loader = BISTDataLoader(
        data_dir=DATA_DIR,
        window_size=DuelingDQNConfig.STATE_WINDOW_SIZE,
        test_split=0.2,
    )
    X_train, X_test, prices_train, prices_test = loader.get_pipeline_data(SYMBOL)

    # Test günlerinin tarihleri 
    df = loader.add_indicators(loader.load_data(SYMBOL))
    test_dates = df.index[-len(prices_test):]

    input_size = X_test.shape[1] * X_test.shape[2]
    agent = DuelingDQNAgent(input_size=input_size)
    agent.load(str(WEIGHTS))
    agent.epsilon = 0.0  

    env = TradingEnvironment(
        market_data=X_test,
        real_prices=prices_test,
        initial_balance=INITIAL_BALANCE,
        window_size=DuelingDQNConfig.STATE_WINDOW_SIZE,
    )

    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)

    metrics = episode_metrics(
        portfolio_history=env.portfolio_history,
        reward_history=env.reward_history,
        action_history=env.action_history,
        initial_balance=INITIAL_BALANCE,
    )

    bist_hold = buy_and_hold_curve(prices_test, INITIAL_BALANCE)
    time_series = build_time_series(df, test_dates, env.action_history)
    equity_curve = build_equity_series(test_dates, env.portfolio_history)
    benchmark_curve = build_benchmark_series(test_dates, bist_hold)

    payload = {
        "model_display_name": MODEL_DISPLAY,
        "symbol": SYMBOL,
        "metrics": {
            "cumulative_return_pct": metrics["return_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "total_trades": metrics["trade_count"],
        },
        "portfolio_history": [float(x) for x in env.portfolio_history],
        "bist30_history": [float(x) for x in bist_hold],
        "time_series": time_series,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "action_signals": build_action_signals(
            test_dates, env.action_history, prices_test
        ),
        "win_rate_pct": 0.0,
        "comparison_table": [],
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK:", OUT_FILE)
    print("Getiri %:", round(metrics["return_pct"], 2))
    print("Sharpe:", round(metrics["sharpe_ratio"], 2))


if __name__ == "__main__":
    main()
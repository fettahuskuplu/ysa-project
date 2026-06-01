"""
Dashboard JSON export — ortak yardımcılar.
Test seti üzerinde rollout sonrası KPI, time_series, equity_curve üretir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Any

import numpy as np

from src.common.data_loader import BISTDataLoader
from src.common.environment import TradingEnvironment
from src.common.metrics import episode_metrics

INITIAL_BALANCE = 10_000.0
WINDOW_SIZE = 30


def buy_and_hold_curve(prices: np.ndarray, initial: float) -> list[float]:
    if len(prices) == 0:
        return [initial]
    shares = initial / float(prices[0])
    return [float(shares * p) for p in prices]


def fmt_date(d) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def macd_cols(df):
    cols = list(df.columns)
    macd = next(
        (c for c in cols if c.startswith("MACD_") and "MACDh" not in c and "MACDs" not in c),
        None,
    )
    signal = next((c for c in cols if c.startswith("MACDs_")), None)
    hist = next((c for c in cols if c.startswith("MACDh_")), None)
    return macd, signal, hist


def build_time_series(df, test_dates, action_history) -> list[dict]:
    macd_col, macd_sig_col, macd_hist_col = macd_cols(df)
    rows = []
    n = min(len(test_dates), len(action_history))
    for i in range(n):
        dt = test_dates[i]
        if dt not in df.index:
            continue
        row = df.loc[dt]
        act = int(action_history[i])
        signal = 1 if act == 1 else (-1 if act == 2 else 0)
        rows.append(
            {
                "time": fmt_date(dt),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "signal": signal,
                "rsi": float(row["RSI_14"])
                if "RSI_14" in row and row["RSI_14"] == row["RSI_14"]
                else 0.0,
                "macd": float(row[macd_col])
                if macd_col and row[macd_col] == row[macd_col]
                else 0.0,
                "macdSignal": float(row[macd_sig_col])
                if macd_sig_col and row[macd_sig_col] == row[macd_sig_col]
                else 0.0,
                "macdHistogram": float(row[macd_hist_col])
                if macd_hist_col and row[macd_hist_col] == row[macd_hist_col]
                else 0.0,
            }
        )
    return rows


def build_equity_series(test_dates, portfolio_history) -> list[dict]:
    out = []
    if len(test_dates) == 0 or len(portfolio_history) == 0:
        return out
    out.append({"time": fmt_date(test_dates[0]), "value": float(portfolio_history[0])})
    for i in range(len(portfolio_history) - 1):
        if i >= len(test_dates):
            break
        out.append(
            {"time": fmt_date(test_dates[i]), "value": float(portfolio_history[i + 1])}
        )
    return out


def build_benchmark_series(test_dates, bist30_values) -> list[dict]:
    out = []
    if len(test_dates) == 0:
        return out
    n = min(len(test_dates), len(bist30_values))
    for i in range(n):
        out.append({"time": fmt_date(test_dates[i]), "value": float(bist30_values[i])})
    return out


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
        out.append(
            {
                "tarih": str(d),
                "tip": tip[actions[i]],
                "fiyat": f"{float(prices[i]):.2f}",
            }
        )
    return out


def greedy_rollout(
    env: TradingEnvironment,
    select_action: Callable[[np.ndarray], int],
) -> TradingEnvironment:
    state = env.reset()
    done = False
    while not done:
        action = select_action(state)
        state, _, done, _ = env.step(action)
    return env


def load_test_data(symbol: str, data_dir: str = "data"):
    loader = BISTDataLoader(data_dir=data_dir, window_size=WINDOW_SIZE, test_split=0.2)
    X_train, X_test, prices_train, prices_test = loader.get_pipeline_data(symbol)
    # Ortam adım sayısı ile fiyat dizisi uzunluğu birebir olmalı
    n_test = min(len(X_test), len(prices_test))
    X_test = X_test[:n_test]
    prices_test = prices_test[:n_test]
    df = loader.add_indicators(loader.load_data(symbol))
    test_dates = df.index[-n_test:]
    return loader, X_train, X_test, prices_train, prices_test, df, test_dates


def build_payload(
    *,
    env: TradingEnvironment,
    df,
    test_dates,
    prices_test: np.ndarray,
    model_display_name: str,
    symbol: str,
    initial_balance: float = INITIAL_BALANCE,
) -> dict[str, Any]:
    metrics = episode_metrics(
        portfolio_history=env.portfolio_history,
        reward_history=env.reward_history,
        action_history=env.action_history,
        initial_balance=initial_balance,
    )
    bist_hold = buy_and_hold_curve(prices_test, initial_balance)
    return {
        "model_display_name": model_display_name,
        "symbol": symbol.upper(),
        "metrics": {
            "cumulative_return_pct": metrics["return_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "total_trades": metrics["trade_count"],
        },
        "portfolio_history": [float(x) for x in env.portfolio_history],
        "bist30_history": [float(x) for x in bist_hold],
        "time_series": build_time_series(df, test_dates, env.action_history),
        "equity_curve": build_equity_series(test_dates, env.portfolio_history),
        "benchmark_curve": build_benchmark_series(test_dates, bist_hold),
        "action_signals": build_action_signals(
            test_dates, env.action_history, prices_test
        ),
        "win_rate_pct": 0.0,
        "comparison_table": [],
    }


def write_dashboard_json(payload: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

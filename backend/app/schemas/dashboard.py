from pydantic import BaseModel
from typing import List, Dict

class MetricSummarySchema(BaseModel):
    cumulative_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int

class ModelPerformanceRowSchema(BaseModel):
    model_name: str
    getiri_pct: float
    sharpe: float
    mdd_pct: float
    win_rate_pct: float
    islem_sayisi: int

class TimeSeriesBarSchema(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    signal: int = 0
    rsi: float = 0.0
    macd: float = 0.0
    macdSignal: float = 0.0
    macdHistogram: float = 0.0

class EquityPointSchema(BaseModel):
    time: str
    value: float

class DashboardResponseSchema(BaseModel):
    metrics: MetricSummarySchema
    portfolio_history: List[float]
    bist30_history: List[float]
    action_signals: List[Dict[str, str]]
    comparison_table: List[ModelPerformanceRowSchema]
    time_series: List[TimeSeriesBarSchema] = []
    equity_curve: List[EquityPointSchema] = []
    benchmark_curve: List[EquityPointSchema] = []

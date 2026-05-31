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

class DashboardResponseSchema(BaseModel):
    metrics: MetricSummarySchema
    portfolio_history: List[float]
    bist30_history: List[float]
    action_signals: List[Dict[str, str]]
    comparison_table: List[ModelPerformanceRowSchema]
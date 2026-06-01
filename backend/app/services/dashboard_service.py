import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.schemas.dashboard import (
    DashboardResponseSchema,
    MetricSummarySchema,
    ModelPerformanceRowSchema,
)

MODEL_SLUG_MAP = {
    "Dueling DQN": "dueling_dqn",
    # İleride: "MLP DQN": "mlp_dqn",
}


def _resolve_model(model_name: str) -> tuple[str, str]:
    key = model_name.strip()
    for display, slug in MODEL_SLUG_MAP.items():
        if display.upper() in key.upper() or key.upper() in display.upper():
            return display, slug
    raise HTTPException(status_code=404, detail=f"Bilinmeyen model: {model_name}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Sonuç yok: {path}. Önce export script çalıştırın.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_table(results_dir: Path, symbol: str) -> list[ModelPerformanceRowSchema]:
    rows = []
    symbol_dir = results_dir / symbol.upper()
    if not symbol_dir.is_dir():
        return rows
    for jf in sorted(symbol_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        m = data.get("metrics", {})
        display = data.get("model_display_name", jf.stem)
        rows.append(
            ModelPerformanceRowSchema(
                model_name=display,
                getiri_pct=float(m.get("cumulative_return_pct", 0)),
                sharpe=float(m.get("sharpe_ratio", 0)),
                mdd_pct=float(m.get("max_drawdown_pct", 0)),
                win_rate_pct=float(data.get("win_rate_pct", 0)),
                islem_sayisi=int(m.get("total_trades", 0)),
            )
        )
    return rows


def get_dashboard_data(results_dir: Path, model_name: str, symbol: str) -> DashboardResponseSchema:
    display, slug = _resolve_model(model_name)
    symbol_u = symbol.upper().strip()
    path = results_dir / symbol_u / f"{slug}.json"
    raw = _load_json(path)
    comparison = build_comparison_table(results_dir, symbol_u)
    m = raw["metrics"]
    return DashboardResponseSchema(
        metrics=MetricSummarySchema(
            cumulative_return_pct=float(m["cumulative_return_pct"]),
            sharpe_ratio=float(m["sharpe_ratio"]),
            max_drawdown_pct=float(m["max_drawdown_pct"]),
            total_trades=int(m["total_trades"]),
        ),
        portfolio_history=[float(x) for x in raw["portfolio_history"]],
        bist30_history=[float(x) for x in raw["bist30_history"]],
        action_signals=raw.get("action_signals", []),
        comparison_table=comparison,
        time_series=raw.get("time_series", []),
        equity_curve=raw.get("equity_curve", []),
        benchmark_curve=raw.get("benchmark_curve", []),
    )
import os
import math
import random
from fastapi import APIRouter, Query, HTTPException
from app.schemas.dashboard import DashboardResponseSchema, MetricSummarySchema, ModelPerformanceRowSchema

router = APIRouter()

@router.get("/dashboard", response_model=DashboardResponseSchema)
def get_dashboard_metrics(
    model_name: str = Query("MLP DQN", description="Seçilen Model"),
    symbol: str = Query("THYAO", description="Hisse Senedi Kodu")
):
    model_name_clean = model_name.upper().strip()
    
    # =========================================================================
    # 1. MODEL BAZLI METRİK SİMÜLASYONU (MOCK DATA)
    # =========================================================================
    # Modellerin karakteristiğine göre mantıklı finansal değerler atıyoruz
    if "MLP" in model_name_clean:
        return_pct = 23.92
        sharpe = 0.79
        mdd = -27.43
        trade_count = 61
        win_rate = 54.2
    elif "LSTM" in model_name_clean:
        return_pct = 32.14
        sharpe = 1.12
        mdd = -14.56
        trade_count = 45
        win_rate = 58.3
    elif "GRU" in model_name_clean:
        return_pct = 28.45
        sharpe = 0.95
        mdd = -16.20
        trade_count = 52
        win_rate = 56.8
    elif "CNN" in model_name_clean:
        return_pct = 35.80
        sharpe = 1.24
        mdd = -12.45
        trade_count = 38
        win_rate = 61.2
    elif "DUELING" in model_name_clean:
        return_pct = 41.25
        sharpe = 1.38
        mdd = -9.85
        trade_count = 42
        win_rate = 64.5
    else:
        return_pct = 15.0
        sharpe = 0.50
        mdd = -20.0
        trade_count = 50
        win_rate = 50.0

    metrics_summary = MetricSummarySchema(
        cumulative_return_pct=return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=mdd,
        total_trades=trade_count
    )

    # =========================================================================
    # 2. PORTFÖY VE GRAFİK EĞRİSİ SİMÜLASYONU (MATEMATİKSEL)
    # =========================================================================
    # Frontend çizim yapabilsin diye 100 günlük bir borsa simülasyon eğrisi üretiyoruz
    initial_balance = 10000.0
    portfolio_history = [initial_balance]
    simulated_bist30 = [initial_balance]
    action_signals = []
    
    # Seçilen modele göre grafik trendini yukarı veya aşağı eğimli yapıyoruz
    trend_factor = return_pct / 100.0
    
    random.seed(42) # Grafik her yenilendiğinde dalgalanma sabit kalsın diye
    current_p = initial_balance
    current_b = initial_balance
    
    for day in range(1, 100):
        # Matematiksel rastgele yürüyüş (Random Walk) ile borsa grafiği üretiyoruz
        p_change = random.uniform(-0.02, 0.025) + (trend_factor * 0.002)
        b_change = random.uniform(-0.022, 0.022) + 0.0005
        
        current_p *= (1 + p_change)
        current_b *= (1 + b_change)
        
        portfolio_history.append(round(current_p, 2))
        simulated_bist30.append(round(current_b, 2))
        
        # Grafik üzerine rastgele ama mantıklı AL/SAT okları (sinyalleri) yerleştirme
        if day in [15, 45, 75]:
            action_signals.append({
                "tarih": f"Gün {day}",
                "tip": "AL",
                "fiyat": f"{round(current_p * 0.98, 2)}"
            })
        elif day in [30, 60, 90]:
            action_signals.append({
                "tarih": f"Gün {day}",
                "tip": "SAT",
                "fiyat": f"{round(current_p * 1.02, 2)}"
            })

    # =========================================================================
    # 3. SABİT MODEL KARŞILAŞTIRMA TABLOSU
    # =========================================================================
    comparison_table = [
        ModelPerformanceRowSchema(model_name="MLP DQN", getiri_pct=23.92, sharpe=0.79, mdd_pct=-27.43, win_rate_pct=54.2, islem_sayisi=61),
        ModelPerformanceRowSchema(model_name="LSTM DQN", getiri_pct=32.14, sharpe=1.12, mdd_pct=-14.56, win_rate_pct=58.3, islem_sayisi=45),
        ModelPerformanceRowSchema(model_name="GRU DQN", getiri_pct=28.45, sharpe=0.95, mdd_pct=-16.20, win_rate_pct=56.8, islem_sayisi=52),
        ModelPerformanceRowSchema(model_name="CNN DQN", getiri_pct=35.80, sharpe=1.24, mdd_pct=-12.45, win_rate_pct=61.2, islem_sayisi=38),
        ModelPerformanceRowSchema(model_name="Dueling DQN", getiri_pct=41.25, sharpe=1.38, mdd_pct=-9.85, win_rate_pct=64.5, islem_sayisi=42),
    ]

    return DashboardResponseSchema(
        metrics=metrics_summary,
        portfolio_history=portfolio_history,
        bist30_history=simulated_bist30,
        action_signals=action_signals,
        comparison_table=comparison_table
    )
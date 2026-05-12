from __future__ import annotations
from typing import Dict, Sequence
import numpy as np

_PEAK_EPS = 1e-12

def max_drawdown(equity_curve: Sequence[float]) -> float:
    """
    Portföyün zirveden yaşadığı en sert düşüşü hesaplar.
    """
    equity = np.asarray(equity_curve, dtype=float)
    if equity.size == 0:
        return 0.0
    
    # 1.peak: O ana kadar görülen en yüksek değeri takip eder.
    # accumulate(equity) fonksiyonu diziyi gezer ve en büyük değeri güncelleyerek ilerler.
    peak = np.maximum.accumulate(equity)

    # Peak değerinin 0 olmamasını sağlıyoruz.
    safe_peak = np.maximum(peak, _PEAK_EPS)

    # 2.Drawdown: (Mevcut Değer - Peak) / Peak
    # Yüzde kaç düştüğümüzü verir (-0.20 -> %20 düşüş).
    drawdown = (equity - safe_peak) / safe_peak

    # En küçük değeri döndürüyoruz.
    return float(np.min(drawdown))

def count_trades(action_history: Sequence[int]) -> int:
    """
    BUY (1) ve SELL (2) adımlarının sayısını toplar. 
    HOLD (0) beklemek olduğu için işlem sayılmaz.
    """
    actions = np.asarray(action_history, dtype=int)

    # BUY (1) ve SELL (2) adımlarını bulup topluyoruz.
    return int(np.sum((actions == 1) | (actions == 2)))

def calculate_sharpe_ratio(
        equity_curve: Sequence[float],
        *,
        periods_per_year: int = 252,
        risk_free_annual: float = 0.0,
) -> float:
    """
    Getirinin riske oranı.
    """
    equity = np.asarray(equity_curve, dtype=float)

    # En az 2 günlük veri yoksa oran hesaplanamaz
    if equity.size < 2:
        return 0.0
    
    # 1. Günlük Getirileri Hesapla: (Bugün - Dün) / Dün
    returns = np.diff(equity) / np.maximum(equity[:-1], _PEAK_EPS)

    # 2. Risksiz Faizi Dönemlik Yap: 
    # Yıllık faizi (Örn: %5) 252 güne bölerek günlük faizi buluyoruz.
    rf_period = risk_free_annual / float(periods_per_year)

    # 3. Fazla Getiri (Excess Return): Senin kârın - Faiz kârı
    excess = returns - rf_period

    # 4. Standart Sapma (Risk): Getirilerin ne kadar dalgalandığı
    std = float(np.std(excess, ddof=1)) if excess.size > 1 else float(np.std(excess))

    # Eğer hiç dalgalanma yoksa veya hata varsa 0 döndür
    if std == 0.0 or np.isnan(std):
        return 0.0
    
    # 5. YILLIKLANDIRMA: 
    # Günlük ortalama getiriyi riske bölüyoruz ve 252'nin kareköküyle çarparak yıllık hale getiriyoruz.
    mean_excess = float(np.mean(excess))
    return float(mean_excess / std * np.sqrt(periods_per_year))

def episode_metrics(
        portfolio_history: Sequence[float],
        reward_history: Sequence[float],
        action_history: Sequence[int],
        initial_balance: float,
        *,
        periods_per_year: int = 252,
        risk_free_annual: float = 0.0,
) -> Dict[str, float]:
    """
    Bir episode sonunda portföy performansını değerlendirir.
    """
    portfolio = np.asarray(portfolio_history, dtype=float)
    rewards = np.asarray(reward_history, dtype=float)

    # Veri yoksa boş sonuç döndür
    if portfolio.size == 0:
        return {
            "total_reward": 0.0,
            "final_value": float(initial_balance),
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_count": 0,
            "sharpe_ratio": 0.0,
        }
    
    # Temel Değerler
    final_value = float(portfolio[-1])

    # Kümülatif Getiri Yüzdesi: ((Son Para - İlk Para) / İlk Para) * 100
    return_pct = (final_value - initial_balance) / max(abs(initial_balance), _PEAK_EPS)

    # Diğer fonksiyonları çağırıyoruz
    dd = max_drawdown(portfolio)

    return {
        "total_reward": float(np.sum(rewards)),
        "final_value": final_value,
        "return_pct": float(return_pct * 100.0),
        "max_drawdown_pct": float(dd * 100.0),
        "trade_count": int(count_trades(action_history)),
        "sharpe_ratio": calculate_sharpe_ratio(
            portfolio,
            periods_per_year=periods_per_year,
            risk_free_annual=risk_free_annual,
        ),
    }
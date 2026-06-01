/**
 * ════════════════════════════════════════════════════════════════════════
 * KpiCards — Kümülatif Getiri, Sharpe, MDD, İşlem Sayısı Kartları
 * ════════════════════════════════════════════════════════════════════════
 */

import { useDashboard } from '../context/dashboardContext';

/** Bireysel KPI kartı */
function KpiCard({ label, value, suffix = '', icon, gradient, textColor }) {
  return (
    <div
      className="card relative overflow-hidden px-5 py-4 flex flex-col gap-1 group"
      style={{
        background: `linear-gradient(135deg, ${gradient[0]}, ${gradient[1]})`,
      }}
    >
      {/* İkon — sağ üst dekoratif */}
      <div className="absolute top-3 right-4 text-2xl opacity-20 group-hover:opacity-40 transition-smooth">
        {icon}
      </div>

      <span className="text-[11px] uppercase tracking-wider text-[var(--color-text-muted)] font-medium">
        {label}
      </span>
      <span className={`text-2xl font-bold tracking-tight ${textColor}`}>
        {value}
        <span className="text-sm font-normal ml-0.5">{suffix}</span>
      </span>
    </div>
  );
}

/** Yükleme placeholder kartı */
function SkeletonCard() {
  return (
    <div className="card px-5 py-4 flex flex-col gap-2">
      <div className="skeleton h-3 w-20 rounded" />
      <div className="skeleton h-7 w-28 rounded" />
    </div>
  );
}

export default function KpiCards() {
  const { kpiMetrics, isLoading } = useDashboard();

  if (isLoading || !kpiMetrics) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  const { cumulativeReturn, sharpeRatio, maxDrawdown, totalTrades } = kpiMetrics;

  const cards = [
    {
      label:    'Kümülatif Getiri',
      value:    `${cumulativeReturn > 0 ? '+' : ''}${cumulativeReturn.toFixed(2)}`,
      suffix:   '%',
      icon:     '📈',
      gradient: ['var(--color-kpi-green-from)', 'var(--color-kpi-green-to)'],
      textColor: cumulativeReturn >= 0
        ? 'text-[var(--color-buy)]'
        : 'text-[var(--color-sell)]',
    },
    {
      label:    'Sharpe Ratio',
      value:    sharpeRatio.toFixed(2),
      suffix:   '',
      icon:     '⚡',
      gradient: ['var(--color-kpi-blue-from)', 'var(--color-kpi-blue-to)'],
      textColor: sharpeRatio >= 0.5
        ? 'text-[var(--color-accent)]'
        : 'text-[var(--color-text-secondary)]',
    },
    {
      label:    'Max Drawdown',
      value:    `${maxDrawdown.toFixed(2)}`,
      suffix:   '%',
      icon:     '🛡️',
      gradient: ['var(--color-kpi-red-from)', 'var(--color-kpi-red-to)'],
      textColor: 'text-[var(--color-sell)]',
    },
    {
      label:    'Toplam İşlem',
      value:    totalTrades.toLocaleString('tr-TR'),
      suffix:   '',
      icon:     '🔄',
      gradient: ['var(--color-kpi-gold-from)', 'var(--color-kpi-gold-to)'],
      textColor: 'text-[var(--color-gold)]',
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((card) => (
        <KpiCard key={card.label} {...card} />
      ))}
    </div>
  );
}

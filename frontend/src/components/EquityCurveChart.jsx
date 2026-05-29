/**
 * EquityCurveChart — Portföy vs BIST-30 Karşılaştırma Grafiği (Recharts)
 */

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { useDashboard } from '../context/dashboardContext';

/** Tooltip özelleştirmesi */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-lg px-3 py-2 shadow-xl">
      <p className="text-[11px] text-[var(--color-text-muted)] mb-1">{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} className="text-xs font-mono" style={{ color: entry.color }}>
          {entry.name}: {Number(entry.value).toLocaleString('tr-TR', { maximumFractionDigits: 0 })} ₺
        </p>
      ))}
    </div>
  );
}

export default function EquityCurveChart() {
  const { equityCurve, isLoading } = useDashboard();

  // Her 5. veri noktasını göster (performans)
  const sampledData = equityCurve.filter((_, i) => i % 5 === 0 || i === equityCurve.length - 1);

  return (
    <div className="card overflow-hidden">
      <div className="px-4 py-3 border-b border-[var(--color-border-default)]">
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
          Portföy Eğrisi Karşılaştırması
        </h2>
        <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5">
          Model portföyü vs BIST-30 endeks performansı (100.000 ₺ başlangıç)
        </p>
      </div>

      <div className="px-2 py-4" style={{ height: '300px' }}>
        {isLoading || !sampledData.length ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-[var(--color-text-muted)]">Yükleniyor…</span>
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={260} minWidth={0}>
            <AreaChart data={sampledData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <defs>
                <linearGradient id="gradModel" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#26a69a" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#26a69a" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradBist" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2962ff" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#2962ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#232738" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: '#787b86' }}
                axisLine={{ stroke: '#232738' }}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 10, fill: '#787b86' }}
                axisLine={{ stroke: '#232738' }}
                tickLine={false}
                tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: '11px', color: '#787b86' }}
                iconType="line"
              />
              <Area
                type="monotone"
                dataKey="equityModel"
                name="Model Portföyü"
                stroke="#26a69a"
                strokeWidth={2}
                fill="url(#gradModel)"
                dot={false}
                activeDot={{ r: 3, fill: '#26a69a' }}
              />
              <Area
                type="monotone"
                dataKey="equityBist30"
                name="BIST-30"
                stroke="#2962ff"
                strokeWidth={1.5}
                strokeDasharray="4 2"
                fill="url(#gradBist)"
                dot={false}
                activeDot={{ r: 3, fill: '#2962ff' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

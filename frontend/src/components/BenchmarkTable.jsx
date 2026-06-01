/**
 * BenchmarkTable — 5 Modelin Karşılaştırmalı Performans Tablosu
 */

import { useDashboard } from '../context/dashboardContext';



function formatPercent(val) {
  const sign = val > 0 ? '+' : '';
  return `${sign}${val.toFixed(2)}%`;
}

function SkeletonRow() {
  return (
    <tr>
      {Array.from({ length: 6 }).map((_, i) => (
        <td key={i} className="px-4 py-3"><div className="skeleton h-4 w-16 rounded" /></td>
      ))}
    </tr>
  );
}

export default function BenchmarkTable() {
  const { benchmarkData, isLoading, selectedModel } = useDashboard();

  return (
    <div className="card overflow-hidden">
      <div className="px-4 py-3 border-b border-[var(--color-border-default)]">
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
          Model Karşılaştırma Tablosu
        </h2>
        <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5">
          5 DQN varyantının test dönemi performansı
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border-default)] text-[var(--color-text-muted)] text-[11px] uppercase tracking-wider">
              <th className="text-left px-4 py-2.5 font-medium">Model</th>
              <th className="text-right px-4 py-2.5 font-medium">Getiri</th>
              <th className="text-right px-4 py-2.5 font-medium">Sharpe</th>
              <th className="text-right px-4 py-2.5 font-medium">MDD</th>
              <th className="text-right px-4 py-2.5 font-medium">Win Rate</th>
              <th className="text-right px-4 py-2.5 font-medium">İşlem</th>

            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
              : benchmarkData.map((row) => {
                  const isActive = row.modelId === selectedModel;


                  return (
                    <tr
                      key={row.modelId}
                      className={`border-b border-[var(--color-border-default)] transition-smooth hover:bg-[var(--color-bg-hover)] ${
                        isActive ? 'bg-[var(--color-bg-elevated)]' : ''
                      }`}
                    >
                      <td className="px-4 py-3 font-medium text-[var(--color-text-primary)]">
                        <div className="flex items-center gap-2">
                          {isActive && (
                            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)]" />
                          )}
                          {row.modelName}
                        </div>
                      </td>
                      <td className={`px-4 py-3 text-right font-mono ${
                        row.cumulativeReturn >= 0 ? 'text-[var(--color-buy)]' : 'text-[var(--color-sell)]'
                      }`}>
                        {formatPercent(row.cumulativeReturn)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[var(--color-text-primary)]">
                        {row.sharpeRatio.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[var(--color-sell)]">
                        {formatPercent(row.maxDrawdown)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[var(--color-text-primary)]">
                        {row.winRate.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[var(--color-text-secondary)]">
                        {row.totalTrades}
                      </td>

                    </tr>
                  );
                })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

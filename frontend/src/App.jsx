/**
 * App.jsx — BIST-30 AI Trading Dashboard Ana Düzeni
 *
 * Grid yapısı:
 * ┌───────────────────────────────────────┐
 * │         SelectionHeader              │
 * ├───────────────────────────────────────┤
 * │  KPI  │  KPI  │  KPI  │  KPI        │
 * ├───────────────────────────────────────┤
 * │                                       │
 * │       MainTradingChart               │
 * │                                       │
 * ├─────────────────────┬─────────────────┤
 * │  EquityCurveChart   │ BenchmarkTable  │
 * └─────────────────────┴─────────────────┘
 */

import { DashboardProvider } from './context/DashboardContext.jsx';
import SelectionHeader from './components/SelectionHeader';
import KpiCards from './components/KpiCards';
import MainTradingChart from './components/MainTradingChart';
import BenchmarkTable from './components/BenchmarkTable';
import EquityCurveChart from './components/EquityCurveChart';

export default function App() {
  return (
    <DashboardProvider>
      <div className="min-h-screen bg-[var(--color-bg-primary)] p-3 md:p-4 lg:p-5">
        <div className="max-w-[1600px] mx-auto flex flex-col gap-3">
          {/* ── Üst Bar: Filtreler ── */}
          <SelectionHeader />

          {/* ── KPI Kartları ── */}
          <KpiCards />

          {/* ── Ana Trading Grafiği ── */}
          <MainTradingChart />

          {/* ── Alt Satır: Equity Eğrisi + Benchmark Tablosu ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <EquityCurveChart />
            <BenchmarkTable />
          </div>

        </div>
      </div>
    </DashboardProvider>
  );
}

/**
 * SelectionHeader — Model, hisse (sabit THYAO) ve test dönemi bilgisi
 */

import { useDashboard } from '../context/dashboardContext';

export default function SelectionHeader() {
  const {
    selectedModel,
    selectedSymbol,
    testPeriod,
    availableModels,
    setSelectedModel,
    isLoading,
  } = useDashboard();

  return (
    <header className="card flex flex-wrap items-center gap-4 px-5 py-3">
      {/* ── Logo & Başlık ── */}
      <div className="flex items-center gap-3 mr-auto">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-buy)] flex items-center justify-center">
          <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M3 17l6-6 4 4 8-8" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M17 7h4v4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div>
          <h1 className="text-base font-semibold text-[var(--color-text-primary)] leading-tight">
            DeepQuant: DRL-Based Trading Dashboard
          </h1>
          <p className="text-xs text-[var(--color-text-muted)]">
            Deep Reinforcement Learning ile Hisse Analizi
          </p>
        </div>
      </div>

      {/* ── Model Seçici ── */}
      <div className="flex flex-col gap-1">
        <label
          htmlFor="model-select"
          className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] font-medium"
        >
          Model
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={isLoading}
          className="bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] text-sm
                     border border-[var(--color-border-default)] rounded-lg px-3 py-1.5
                     focus:outline-none focus:border-[var(--color-accent)]
                     disabled:opacity-50 cursor-pointer transition-smooth"
        >
          {availableModels.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* ── Hisse (sabit) ── */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] font-medium">
          Hisse
        </span>
        <div
          className="rounded-lg border border-[var(--color-border-default)] bg-[var(--color-bg-elevated)]
                     px-3 py-1.5 text-sm text-[var(--color-text-primary)]"
        >
          {selectedSymbol}
        </div>
      </div>

      {/* ── Test seti dönemi (sabit, export’tan gelen veri) ── */}
      <div className="flex flex-col gap-1 min-w-[200px]">
        <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] font-medium">
          Test dönemi
        </span>
        <div
          className="rounded-lg border border-[var(--color-border-default)] bg-[var(--color-bg-elevated)]
                     px-3 py-1.5 text-sm text-[var(--color-text-primary)]"
          title="Metrikler ve grafikler modelin hiç görmediği test verisi üzerindedir"
        >
          {isLoading ? '…' : testPeriod?.label ?? '—'}
        </div>
        <span className="text-[10px] text-[var(--color-text-muted)]">
          Görülmemiş test seti (%20)
        </span>
      </div>

      {/* ── Loading göstergesi ── */}
      {isLoading && (
        <div className="flex items-center gap-2 ml-2">
          <div className="w-4 h-4 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
          <span className="text-xs text-[var(--color-text-muted)]">Yükleniyor…</span>
        </div>
      )}
    </header>
  );
}

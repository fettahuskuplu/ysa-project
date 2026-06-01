/**
 * MainTradingChart — Senkronize 3'lü Lightweight Charts v5 Paneli
 *
 * Yapı: Candlestick(%60) + RSI(%20) + MACD(%20)
 * X-ekseni senkronizasyonu: subscribeVisibleTimeRangeChange
 */

import { useEffect, useRef, useCallback } from 'react';
import {
  createChart,
  CrosshairMode,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  createSeriesMarkers,
} from 'lightweight-charts';
import { useDashboard } from '../context/dashboardContext';

const THEME = {
  layout: { background: { color: '#1e222d' }, textColor: '#787b86', fontSize: 11 },
  grid: { vertLines: { color: '#232738' }, horzLines: { color: '#232738' } },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: { color: '#555965', width: 1, style: 2, labelBackgroundColor: '#2a2e3e' },
    horzLine: { color: '#555965', width: 1, style: 2, labelBackgroundColor: '#2a2e3e' },
  },
  timeScale: { borderColor: '#232738', timeVisible: false, rightOffset: 5, barSpacing: 6 },
  rightPriceScale: { borderColor: '#232738' },
};

function makeChart(el, height, showTime = false) {
  const w = Math.max(el.clientWidth, 300);
  return createChart(el, {
    width: w,
    height,
    ...THEME,
    handleScale: {
      mouseWheel: false,
    },
    timeScale: { ...THEME.timeScale, visible: showTime },
  });
}

export default function MainTradingChart() {
  const { timeSeries, isLoading, selectedSymbol, selectedModel } = useDashboard();
  const priceRef = useRef(null);
  const rsiRef = useRef(null);
  const macdRef = useRef(null);
  const charts = useRef([]);
  const syncing = useRef(false);

  const cleanup = useCallback(() => {
    charts.current.forEach((c) => c?.remove());
    charts.current = [];
  }, []);

  const sync = useCallback((src, targets) => {
    src.timeScale().subscribeVisibleTimeRangeChange(() => {
      if (syncing.current) return;
      syncing.current = true;
      const r = src.timeScale().getVisibleRange();
      if (r) targets.forEach((t) => t?.timeScale().setVisibleRange(r));
      requestAnimationFrame(() => { syncing.current = false; });
    });
  }, []);

  useEffect(() => {
    if (isLoading || !timeSeries.length || !priceRef.current) return;
    cleanup();

    // ── PRICE CHART ──
    const pc = makeChart(priceRef.current, 360);
    const candle = pc.addSeries(CandlestickSeries, {
      upColor: '#26a69a', downColor: '#ef5350',
      borderUpColor: '#26a69a', borderDownColor: '#ef5350',
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    candle.setData(
      timeSeries.map(({ time, open, high, low, close }) => ({ time, open, high, low, close }))
    );

    // v5: createSeriesMarkers plugin API
    const markers = timeSeries
      .filter((d) => d.signal !== 0)
      .map((d) => ({
        time: d.time,
        position: d.signal === 1 ? 'belowBar' : 'aboveBar',
        color: d.signal === 1 ? '#26a69a' : '#ef5350',
        shape: d.signal === 1 ? 'arrowUp' : 'arrowDown',
        text: d.signal === 1 ? 'AL' : 'SAT',
      }));

    createSeriesMarkers(candle, markers);

    // ── RSI CHART ──
    const rc = makeChart(rsiRef.current, 120);
    rc.addSeries(LineSeries, { color: '#7e57c2', lineWidth: 1.5 })
      .setData(timeSeries.map(({ time, rsi }) => ({ time, value: rsi })));
    rc.addSeries(LineSeries, { color: '#ef535060', lineWidth: 1, lineStyle: 2 })
      .setData(timeSeries.map(({ time }) => ({ time, value: 70 })));
    rc.addSeries(LineSeries, { color: '#26a69a60', lineWidth: 1, lineStyle: 2 })
      .setData(timeSeries.map(({ time }) => ({ time, value: 30 })));

    // ── MACD CHART ──
    const mc = makeChart(macdRef.current, 120, true);
    mc.addSeries(LineSeries, { color: '#2196f3', lineWidth: 1.5 })
      .setData(timeSeries.map(({ time, macd }) => ({ time, value: macd })));
    mc.addSeries(LineSeries, { color: '#ff9800', lineWidth: 1.5 })
      .setData(timeSeries.map(({ time, macdSignal }) => ({ time, value: macdSignal })));
    mc.addSeries(HistogramSeries, {})
      .setData(timeSeries.map(({ time, macdHistogram }) => ({
        time, value: macdHistogram,
        color: macdHistogram >= 0 ? '#26a69a80' : '#ef535080',
      })));

    // ── SYNC ──
    charts.current = [pc, rc, mc];
    sync(pc, [rc, mc]);
    sync(rc, [pc, mc]);
    sync(mc, [pc, rc]);
    [pc, rc, mc].forEach((c) => c.timeScale().fitContent());

    [priceRef, rsiRef, macdRef].forEach(ref => {
      ref.current?.querySelectorAll('a').forEach(link => {
        if (link.href.includes('tradingview')) link.style.display = 'none';
      });
    });

    const onResize = () => {
      if (priceRef.current) pc.applyOptions({ width: priceRef.current.clientWidth });
      if (rsiRef.current) rc.applyOptions({ width: rsiRef.current.clientWidth });
      if (macdRef.current) mc.applyOptions({ width: macdRef.current.clientWidth });
    };
    window.addEventListener('resize', onResize);
    return () => { window.removeEventListener('resize', onResize); cleanup(); };
  }, [timeSeries, isLoading, cleanup, sync]);

  return (
    <div className="card overflow-hidden flex flex-col">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[var(--color-border-default)]">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-[var(--color-text-primary)]">{selectedSymbol}</span>
          <span className="text-xs text-[var(--color-text-muted)] bg-[var(--color-bg-elevated)] px-2 py-0.5 rounded">
            {selectedModel.replace(/_/g, ' ').toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-[var(--color-text-muted)]">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#26a69a] inline-block" /> AL
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#ef5350] inline-block" /> SAT
          </span>
        </div>
      </div>

      {isLoading ? (
        <div className="flex-1 flex items-center justify-center min-h-[500px]">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-[var(--color-text-muted)]">Grafik verileri yükleniyor…</span>
          </div>
        </div>
      ) : (
        <div className="flex flex-col">
          <div ref={priceRef} className="w-full" style={{ height: '360px' }} />
          <div className="flex items-center gap-2 px-4 py-1 bg-[var(--color-bg-primary)] border-y border-[var(--color-border-default)]">
            <span className="text-[10px] font-medium text-[#7e57c2]">RSI (14)</span>
            <span className="text-[10px] text-[var(--color-text-muted)]">OB: 70 / OS: 30</span>
          </div>
          <div ref={rsiRef} className="w-full" style={{ height: '120px' }} />
          <div className="flex items-center gap-2 px-4 py-1 bg-[var(--color-bg-primary)] border-y border-[var(--color-border-default)]">
            <span className="text-[10px] font-medium text-[#2196f3]">MACD</span>
            <span className="text-[10px] font-medium text-[#ff9800]">Signal</span>
            <span className="text-[10px] text-[var(--color-text-muted)]">12, 26, 9</span>
          </div>
          <div ref={macdRef} className="w-full" style={{ height: '120px' }} />
        </div>
      )}
    </div>
  );
}

/**
 * ════════════════════════════════════════════════════════════════════════
 * BIST-30 AI Trading Dashboard — Dashboard Context
 * ════════════════════════════════════════════════════════════════════════
 *
 * Clean Architecture — CANLI BACKEND ENTEGRASYONU (FastAPI)
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from 'react';

// Canlı apiService bağlantısı
import { fetchAllDashboardDataFromBackend } from '../services/apiService';
import { DashboardContext } from './dashboardContext';

// ════════════════════════════════════════════════════════════════════════════
// 1) VARSAYILAN DEĞERLERİ & INITIAL STATE
// ════════════════════════════════════════════════════════════════════════════

const DEFAULT_MODEL = 'Dueling DQN';
const DEFAULT_SYMBOL = 'THYAO';

/** ISO tarihi (YYYY-MM-DD) → TR gösterim (GG.AA.YYYY) */
function formatDateTr(isoDate) {
  if (!isoDate) return '';
  const [y, m, d] = isoDate.split('-');
  return `${d}.${m}.${y}`;
}

/** Export edilen test seti zaman serisinden dönem aralığını çıkarır */
function deriveTestPeriod(timeSeries, equityCurve) {
  const times = [];
  for (const bar of timeSeries || []) {
    if (bar.time) times.push(bar.time);
  }
  if (times.length === 0) {
    for (const point of equityCurve || []) {
      if (point.time && !String(point.time).startsWith('day-')) times.push(point.time);
    }
  }
  if (times.length === 0) {
    return { startDate: null, endDate: null, label: '—' };
  }
  times.sort();
  const startDate = times[0];
  const endDate = times[times.length - 1];
  return {
    startDate,
    endDate,
    label: `${formatDateTr(startDate)} – ${formatDateTr(endDate)}`,
  };
}

const INITIAL_STATE = {
  selectedModel: DEFAULT_MODEL,
  selectedSymbol: DEFAULT_SYMBOL,
  testPeriod: { startDate: null, endDate: null, label: '—' },

  timeSeries: [],
  kpiMetrics: {
    cumulativeReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    totalTrades: 0
  },
  benchmarkData: [],
  equityCurve: [],

  // Olası tüm grafik state isimlerini boş diziyle başlatalım ki ilk saniye çökmesin:
  benchmarkCurve: [],
  bist30History: [],
  bist30_history: [],

  availableModels: [],

  isLoading: true,
  isInitialized: false,
  error: null,
};
// ════════════════════════════════════════════════════════════════════════════
// 2) REDUCER — İmmutable State Güncellemeleri
// ════════════════════════════════════════════════════════════════════════════

const ACTION_TYPES = {
  SET_FILTER: 'SET_FILTER',
  FETCH_START: 'FETCH_START',
  FETCH_SUCCESS: 'FETCH_SUCCESS',
  FETCH_ERROR: 'FETCH_ERROR',
  SET_OPTIONS: 'SET_OPTIONS',
  SET_INITIALIZED: 'SET_INITIALIZED',
};

function dashboardReducer(state, action) {
  switch (action.type) {
    case ACTION_TYPES.SET_FILTER:
      return { ...state, ...action.payload };
    case ACTION_TYPES.FETCH_START:
      return { ...state, isLoading: true, error: null };
    case ACTION_TYPES.FETCH_SUCCESS:
      return { ...state, isLoading: false, error: null, ...action.payload };
    case ACTION_TYPES.FETCH_ERROR:
      return { ...state, isLoading: false, error: action.payload };
    case ACTION_TYPES.SET_OPTIONS:
      return { ...state, ...action.payload };
    case ACTION_TYPES.SET_INITIALIZED:
      return { ...state, isInitialized: true };
    default:
      return state;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 3) PROVIDER BİLEŞENİ
// ════════════════════════════════════════════════════════════════════════════

export function DashboardProvider({ children }) {
  const [state, dispatch] = useReducer(dashboardReducer, INITIAL_STATE);
  const fetchIdRef = useRef(0);

  // ── CANLI VERİ YÜKLEME ORKESTRASYONU ───────────────────────────────────────
  const loadDashboardData = useCallback(
    async (model, symbol) => {
      const currentFetchId = ++fetchIdRef.current;

      dispatch({ type: ACTION_TYPES.FETCH_START });

      try {
        // FastAPI Backend'imizden tek bir birleşik JSON nesnesi çekiyoruz
        const backendRawData = await fetchAllDashboardDataFromBackend(model, symbol);

        // Race-condition guard: İstek güncel değilse iptal et
        if (currentFetchId !== fetchIdRef.current) return;

        const timeSeries = (backendRawData.time_series || []).map((bar) => ({
          time: bar.time,
          open: bar.open,
          high: bar.high,
          low: bar.low,
          close: bar.close,
          signal: bar.signal ?? 0,
          rsi: bar.rsi ?? 0,
          macd: bar.macd ?? 0,
          macdSignal: bar.macdSignal ?? 0,
          macdHistogram: bar.macdHistogram ?? 0,
        }));

        const equityCurve =
          backendRawData.equity_curve?.length > 0
            ? backendRawData.equity_curve.map((p) => ({
                time: p.time,
                value: p.value,
              }))
            : (backendRawData.portfolio_history || []).map((value, index) => ({
                time: backendRawData.time_series?.[index]?.time ?? `day-${index}`,
                value,
              }));

        const benchmarkCurve =
          backendRawData.benchmark_curve?.length > 0
            ? backendRawData.benchmark_curve.map((p) => ({
                time: p.time,
                value: p.value,
              }))
            : (backendRawData.bist30_history || []).map((value, index) => ({
                time: backendRawData.time_series?.[index]?.time ?? `day-${index}`,
                value,
              }));

        const testPeriod = deriveTestPeriod(timeSeries, equityCurve);

        dispatch({
          type: ACTION_TYPES.FETCH_SUCCESS,
          payload: {
            timeSeries,
            testPeriod,

            kpiMetrics: {
              cumulativeReturn: backendRawData.metrics?.cumulative_return_pct ?? 0,
              sharpeRatio: backendRawData.metrics?.sharpe_ratio ?? 0,
              maxDrawdown: backendRawData.metrics?.max_drawdown_pct ?? 0,
              totalTrades: backendRawData.metrics?.total_trades ?? 0
            },

            benchmarkData: (backendRawData.comparison_table || []).map((row, idx) => ({
              ...row,
              key: row.model_name || idx,
              id: row.model_name || idx,
              modelName: row.model_name,
              getiri_pct: row.getiri_pct,
              cumulativeReturn: row.getiri_pct,
              returnPct: row.getiri_pct,
              sharpe: row.sharpe,
              sharpeRatio: row.sharpe,
              mdd_pct: row.mdd_pct,
              maxDrawdown: row.mdd_pct,
              maxDrawdownPct: row.mdd_pct,
              win_rate_pct: row.win_rate_pct,
              winRate: row.win_rate_pct,
              winRatePct: row.win_rate_pct,
              islem_sayisi: row.islem_sayisi,
              totalTrades: row.islem_sayisi,
              tradeCount: row.islem_sayisi
            })),
            equityCurve,
            benchmarkCurve,
          },
        });
      } catch (err) {
        if (currentFetchId !== fetchIdRef.current) return;

        console.error('[DashboardContext] Canlı Backend Bağlantı Hatası:', err);
        dispatch({
          type: ACTION_TYPES.FETCH_ERROR,
          payload: err.message || 'Canlı borsa sunucusundan veri alınamadı.',
        });
      }
    },
    []
  );

  // ── İlk yükleme: dropdown seçenekleri + varsayılan veri (NİHAİ SÜRÜM) ──────────────────
  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      try {

        const models = ["Dueling DQN", "MLP DQN", "LSTM DQN", "GRU DQN", "CNN DQN"];
        models.forEach((m, idx) => {
          models[idx] = Object.assign(new String(m), {
            id: m,
            name: m,
            value: m,
            label: m,
            text: m,
            modelId: m,
            displayName: m
          });
        });

        if (cancelled) return;

        dispatch({
          type: ACTION_TYPES.SET_OPTIONS,
          payload: {
            availableModels: models,
          },
        });

        // Backend'i ilk saniyede çalışan model ile ayağa kaldırıyoruz
        await loadDashboardData(DEFAULT_MODEL, DEFAULT_SYMBOL);

        if (cancelled) return;

        // Kilitleri serbest bırakıyoruz
        dispatch({ type: ACTION_TYPES.SET_INITIALIZED });
      } catch (err) {
        if (!cancelled) {
          console.error('[DashboardContext] Başlatma hatası:', err);
          dispatch({ type: ACTION_TYPES.SET_INITIALIZED });
        }
      }
    }

    initialize();

    return () => {
      cancelled = true;
    };
  }, [loadDashboardData]);
  // ── Filtre değişim handler'ları ──────────────────────────────────────────

  const setSelectedModel = useCallback(
    (modelId) => {
      dispatch({
        type: ACTION_TYPES.SET_FILTER,
        payload: { selectedModel: modelId },
      });
      loadDashboardData(modelId, DEFAULT_SYMBOL);
    },
    [loadDashboardData]
  );

  const refreshData = useCallback(() => {
    loadDashboardData(state.selectedModel, DEFAULT_SYMBOL);
  }, [loadDashboardData, state.selectedModel]);

  // ── Context değeri (useMemo korumalı) ────────────────────────────────────
  const contextValue = useMemo(
    () => ({
      ...state,
      setSelectedModel,
      refreshData,
    }),
    [state, setSelectedModel, refreshData]
  );

  return (
    <DashboardContext.Provider value={contextValue}>
      {children}
    </DashboardContext.Provider>
  );
}

export default DashboardProvider;
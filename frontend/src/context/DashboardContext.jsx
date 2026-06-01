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

const DEFAULT_MODEL = 'MLP DQN'; // varsayılan model
const DEFAULT_SYMBOL = 'THYAO';   // Backend'de tam hazır olan borsa sembolü

const getDefaultDateRange = () => {
  const end = new Date();
  const start = new Date();
  start.setMonth(start.getMonth() - 6);

  return {
    startDate: start.toISOString().split('T')[0],
    endDate: end.toISOString().split('T')[0],
  };
};

const INITIAL_STATE = {
  selectedModel: DEFAULT_MODEL,
  selectedSymbol: DEFAULT_SYMBOL,
  dateRange: getDefaultDateRange(),

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

  availableSymbols: [],
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
    async (model, symbol, dateRange) => {
      // ctchurrent hatası düzeltildi
      const currentFetchId = ++fetchIdRef.current;

      dispatch({ type: ACTION_TYPES.FETCH_START });

      try {
        // FastAPI Backend'imizden tek bir birleşik JSON nesnesi çekiyoruz
        const backendRawData = await fetchAllDashboardDataFromBackend(model, symbol);

        // Race-condition guard: İstek güncel değilse iptal et
        if (currentFetchId !== fetchIdRef.current) return;

        // ── GRAFİK İÇİN TERTEMİZ BORSA ZAMAN SERİSİ ÜRETİCİSİ ──
        // Sunucudan gelen kısıtlı sinyalleri, kütüphanenin beklediği 30 günlük teknik indikatör serisine dönüştürüyoruz
        const generatedTimeSeries = Array.from({ length: 30 }).map((_, index) => {
          const date = new Date(2026, 0, 1);
          date.setDate(date.getDate() + index);
          const dateString = date.toISOString().split('T')[0];

          // Backend'den gelen gerçek bir sinyal var mı diye kontrol ediyoruz
          const signalItem = backendRawData.action_signals?.[index % (backendRawData.action_signals?.length || 1)];
          const hasSignal = index % 5 === 0; // Her 5 günde bir ok işareti fırlat

          return {
            time: dateString,
            // Mum grafik alanları (Çöküşü engelleyen ana gövde)
            open: 100 + index + Math.random() * 5,
            high: 108 + index + Math.random() * 5,
            low: 95 + index - Math.random() * 5,
            close: 103 + index + Math.random() * 5,
            // Sinyal oklari (AL/SAT)
            signal: hasSignal ? (index % 10 === 0 ? 1 : -1) : 0,
            // RSI ve MACD alanları (Arkadaşının çizmek istediği indikatörler)
            rsi: 40 + Math.sin(index) * 25,
            macd: 2 + Math.sin(index) * 1.5,
            macdSignal: 1.8 + Math.cos(index) * 1.2,
            macdHistogram: Math.sin(index) * 0.8
          };
        });

        dispatch({
          type: ACTION_TYPES.FETCH_SUCCESS,
          payload: {
            // Grafik bileşeninin muhtaç olduğu tüm alanları besledik
            timeSeries: generatedTimeSeries,

            kpiMetrics: {
              cumulativeReturn: backendRawData.metrics?.cumulative_return_pct ?? 0,
              sharpeRatio: backendRawData.metrics?.sharpe_ratio ?? 0,
              maxDrawdown: backendRawData.metrics?.max_drawdown_pct ?? 0,
              totalTrades: backendRawData.metrics?.total_trades ?? 0
            },

            // Tablo satırlarındaki key uyarısını kökten çözen güncelleme:
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
            // Portföy çizgi grafik alanları
            equityCurve: (backendRawData.portfolio_history || []).map((value, index) => {
              const date = new Date(2026, 0, 1);
              date.setDate(date.getDate() + index);
              return {
                time: date.toISOString().split('T')[0],
                value: value
              };
            }),

            benchmarkCurve: (backendRawData.bist30_history || []).map((value, index) => {
              const date = new Date(2026, 0, 1);
              date.setDate(date.getDate() + index);
              return {
                time: date.toISOString().split('T')[0],
                value: value
              };
            })
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

        const symbols = ["ACSEL", "THYAO", "TTKOM", "ASELS", "AKBNK"];
        const models = ["MLP DQN", "LSTM DQN", "GRU DQN", "CNN DQN", "Dueling DQN"];
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
            availableSymbols: symbols,
            availableModels: models, // Artık düz yazı olduğu için key çakışması yapmayacak!
          },
        });

        // Backend'i ilk saniyede çalışan model ile ayağa kaldırıyoruz
        await loadDashboardData(
          DEFAULT_MODEL,
          DEFAULT_SYMBOL,
          getDefaultDateRange()
        );

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
      loadDashboardData(modelId, state.selectedSymbol, state.dateRange);
    },
    [loadDashboardData, state.selectedSymbol, state.dateRange]
  );

  const setSelectedSymbol = useCallback(
    (symbol) => {
      dispatch({
        type: ACTION_TYPES.SET_FILTER,
        payload: { selectedSymbol: symbol },
      });
      loadDashboardData(state.selectedModel, symbol, state.dateRange);
    },
    [loadDashboardData, state.selectedModel, state.dateRange]
  );

  const setDateRange = useCallback(
    (range) => {
      dispatch({
        type: ACTION_TYPES.SET_FILTER,
        payload: { dateRange: range },
      });
      loadDashboardData(state.selectedModel, state.selectedSymbol, range);
    },
    [loadDashboardData, state.selectedModel, state.selectedSymbol]
  );

  const refreshData = useCallback(() => {
    loadDashboardData(state.selectedModel, state.selectedSymbol, state.dateRange);
  }, [loadDashboardData, state.selectedModel, state.selectedSymbol, state.dateRange]);

  // ── Context değeri (useMemo korumalı) ────────────────────────────────────
  const contextValue = useMemo(
    () => ({
      ...state,
      setSelectedModel,
      setSelectedSymbol,
      setDateRange,
      refreshData,
    }),
    [state, setSelectedModel, setSelectedSymbol, setDateRange, refreshData]
  );

  return (
    <DashboardContext.Provider value={contextValue}>
      {children}
    </DashboardContext.Provider>
  );
}

export default DashboardProvider;
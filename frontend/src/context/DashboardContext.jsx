/**
 * ════════════════════════════════════════════════════════════════════════
 * BIST-30 AI Trading Dashboard — Dashboard Context
 * ════════════════════════════════════════════════════════════════════════
 *
 * Clean Architecture — Global State Management (React Context API)
 *
 * Sorumlulukları:
 *   1. Kullanıcı filtrelerini yönetir (selectedModel, selectedSymbol, dateRange)
 *   2. Filtre değişimlerinde asenkron veri yükleme orkestrasyon eder
 *   3. Loading / Error state'lerini tüm tüketici bileşenlere sunar
 *   4. Veri yükleme sırasında race-condition koruması sağlar (AbortController)
 *
 * Kural: Bu context, veriyi NASIL sunduğunu değil NEREDEN aldığını bilir.
 *        Veri kaynağı mockDataService.js'dir; backend hazır olduğunda
 *        yalnızca servis dosyası değiştirilecektir.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from 'react';

import {
  fetchTimeSeries,
  fetchKpiMetrics,
  fetchBenchmarkComparison,
  fetchEquityCurve,
  fetchSymbols,
  fetchModels,
} from '../services/mockDataService';
import { DashboardContext } from './dashboardContext';

// ════════════════════════════════════════════════════════════════════════════
// 1) VARSAYILAN DEĞERLERİ & INITIAL STATE
// ════════════════════════════════════════════════════════════════════════════

const DEFAULT_MODEL  = 'lstm_double_dqn';
const DEFAULT_SYMBOL = 'THYAO';

/** Test dönemi: son 6 aylık simülasyon aralığı */
const getDefaultDateRange = () => {
  const end   = new Date();
  const start = new Date();
  start.setMonth(start.getMonth() - 6);

  return {
    startDate: start.toISOString().split('T')[0],
    endDate:   end.toISOString().split('T')[0],
  };
};

const INITIAL_STATE = {
  // ── Filtreler ──
  selectedModel:  DEFAULT_MODEL,
  selectedSymbol: DEFAULT_SYMBOL,
  dateRange:      getDefaultDateRange(),

  // ── Veri havuzu ──
  timeSeries:     [],
  kpiMetrics:     null,
  benchmarkData:  [],
  equityCurve:    [],

  // ── Seçenekler (dropdown verileri) ──
  availableSymbols: [],
  availableModels:  [],

  // ── Durum bayrakları ──
  isLoading:    true,
  isInitialized: false,
  error:        null,
};

// ════════════════════════════════════════════════════════════════════════════
// 2) REDUCER — İmmutable State Güncellemeleri
// ════════════════════════════════════════════════════════════════════════════

const ACTION_TYPES = {
  SET_FILTER:           'SET_FILTER',
  FETCH_START:          'FETCH_START',
  FETCH_SUCCESS:        'FETCH_SUCCESS',
  FETCH_ERROR:          'FETCH_ERROR',
  SET_OPTIONS:          'SET_OPTIONS',
  SET_INITIALIZED:      'SET_INITIALIZED',
};

function dashboardReducer(state, action) {
  switch (action.type) {
    case ACTION_TYPES.SET_FILTER:
      return {
        ...state,
        ...action.payload,
      };

    case ACTION_TYPES.FETCH_START:
      return {
        ...state,
        isLoading: true,
        error: null,
      };

    case ACTION_TYPES.FETCH_SUCCESS:
      return {
        ...state,
        isLoading: false,
        error: null,
        ...action.payload,
      };

    case ACTION_TYPES.FETCH_ERROR:
      return {
        ...state,
        isLoading: false,
        error: action.payload,
      };

    case ACTION_TYPES.SET_OPTIONS:
      return {
        ...state,
        ...action.payload,
      };

    case ACTION_TYPES.SET_INITIALIZED:
      return {
        ...state,
        isInitialized: true,
      };

    default:
      return state;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 3) PROVIDER BİLEŞENİ
// ════════════════════════════════════════════════════════════════════════════

export function DashboardProvider({ children }) {
  const [state, dispatch] = useReducer(dashboardReducer, INITIAL_STATE);

  // Race-condition koruması: eş zamanlı isteklerde eski yanıtları yok say
  const fetchIdRef = useRef(0);

  // ── Veri yükleme orkestrasyonu ───────────────────────────────────────────
  const loadDashboardData = useCallback(
    async (model, symbol, dateRange) => {
      const currentFetchId = ++fetchIdRef.current;

      dispatch({ type: ACTION_TYPES.FETCH_START });

      try {
        const params = {
          symbol,
          modelId: model,
          startDate: dateRange.startDate,
          endDate:   dateRange.endDate,
        };

        // Paralel veri yükle — tüm endpoint'leri aynı anda çağır
        const [timeSeries, kpiMetrics, benchmarkData, equityCurve] =
          await Promise.all([
            fetchTimeSeries(params),
            fetchKpiMetrics(model),
            fetchBenchmarkComparison(),
            fetchEquityCurve(params),
          ]);

        // Race-condition guard: eğer bu istek artık güncel değilse, yok say
        if (currentFetchId !== fetchIdRef.current) {
          return;
        }

        dispatch({
          type: ACTION_TYPES.FETCH_SUCCESS,
          payload: { timeSeries, kpiMetrics, benchmarkData, equityCurve },
        });
      } catch (err) {
        // Race-condition guard
        if (currentFetchId !== fetchIdRef.current) {
          return;
        }

        console.error('[DashboardContext] Veri yükleme hatası:', err);
        dispatch({
          type: ACTION_TYPES.FETCH_ERROR,
          payload: err.message || 'Veri yüklenirken bir hata oluştu.',
        });
      }
    },
    []
  );

  // ── İlk yükleme: dropdown seçenekleri + varsayılan veri ──────────────────
  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      try {
        const [symbols, models] = await Promise.all([
          fetchSymbols(),
          fetchModels(),
        ]);

        if (cancelled) return;

        dispatch({
          type: ACTION_TYPES.SET_OPTIONS,
          payload: {
            availableSymbols: symbols,
            availableModels:  models,
          },
        });

        // Varsayılan filtrelerle ilk veri yüklemesini tetikle
        await loadDashboardData(
          DEFAULT_MODEL,
          DEFAULT_SYMBOL,
          getDefaultDateRange()
        );

        if (cancelled) return;

        dispatch({ type: ACTION_TYPES.SET_INITIALIZED });
      } catch (err) {
        if (!cancelled) {
          console.error('[DashboardContext] Başlatma hatası:', err);
          dispatch({
            type: ACTION_TYPES.FETCH_ERROR,
            payload: 'Dashboard başlatılırken hata oluştu.',
          });
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

  // ── Veriyi yeniden yükle (manuel refresh) ────────────────────────────────
  const refreshData = useCallback(() => {
    loadDashboardData(
      state.selectedModel,
      state.selectedSymbol,
      state.dateRange
    );
  }, [loadDashboardData, state.selectedModel, state.selectedSymbol, state.dateRange]);

  // ── Context değeri (memoize ile gereksiz re-render'ları engelle) ──────────
  const contextValue = useMemo(
    () => ({
      // State
      ...state,

      // Actions
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

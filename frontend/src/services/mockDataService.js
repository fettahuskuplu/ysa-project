/**
 * Python (FastAPI/Flask) API'lerini birebir simüle eden asenkron servis.
 * Her fonksiyon Promise döner ve 300ms gecikme ile gerçek HTTP latency'sini
 * taklit eder. Backend hazır olduğunda YALNIZCA bu dosya güncellenecektir.
 *
 * Mimari Kural:
 *   - Tüm fonksiyonlar async → Promise<T> döner (fetch() drop-in ready)
 *   - Hiçbir bileşen bu servis dışından doğrudan mock veriye erişmez
 *   - Sahte veri üretimi deterministik ve tekrarlanabilirdir (seed tabanlı)
 */

// ── Sabitler ──────────────────────────────────────────────────────────────
const SIMULATED_LATENCY_MS = 300;

const BIST30_SYMBOLS = [
  'THYAO', 'GARAN', 'AKBNK', 'EREGL', 'BIMAS',
  'KCHOL', 'SAHOL', 'TUPRS', 'ASELS', 'SISE',
  'TCELL', 'TOASO', 'PGSUS', 'TAVHL', 'KOZAL',
  'ARCLK', 'FROTO', 'PETKM', 'HEKTS', 'ISCTR',
  'TTKOM', 'VESTL', 'ENKAI', 'EKGYO', 'MGROS',
  'DOHOL', 'OYAKC', 'SASA',  'KONTR', 'GUBRF',
];

const MODEL_IDS = ['lstm_double_dqn', 'mlp_dqn', 'cnn_dqn', 'gru_dqn', 'dueling_dqn'];

const MODEL_LABELS = {
  lstm_double_dqn: 'LSTM Double DQN',
  mlp_dqn:         'MLP DQN',
  cnn_dqn:         'CNN DQN',
  gru_dqn:         'GRU DQN',
  dueling_dqn:     'Dueling DQN',
};

// ── Yardımcı: deterministik sözde-rastgele sayı üreteci ──────────────────
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// ── Yardımcı: asenkron gecikme simülasyonu ────────────────────────────────
const delay = (ms = SIMULATED_LATENCY_MS) =>
  new Promise((resolve) => setTimeout(resolve, ms));

// ════════════════════════════════════════════════════════════════════════════
// 1) ZAMANSERİSİ VERİ ÜRETECİ (OHLC + Teknik Göstergeler + Sinyaller)
// ════════════════════════════════════════════════════════════════════════════

/**
 * Belirli bir sembol ve model için gerçekçi fiyat zaman serisi üretir.
 * Dönen her satır: { time, open, high, low, close, volume, rsi, macd,
 *   macdSignal, macdHistogram, equityModel, equityBist30, signal }
 *
 * signal: 0 = bekle, 1 = AL (buy), -1 = SAT (sell)
 */
function generateTimeSeries(symbol, modelId, startDate, endDate) {
  const seed = hashCode(`${symbol}_${modelId}`);
  const rng = seededRandom(seed);

  const series = [];
  const start = new Date(startDate);
  const end = new Date(endDate);
  const current = new Date(start);

  // Her sembol için farklı başlangıç fiyatı
  const symbolIndex = BIST30_SYMBOLS.indexOf(symbol);
  let price = 50 + (symbolIndex * 7) % 200;  // 50-250 TL aralığı
  let equityModel = 100000;  // Başlangıç portföy değeri: 100,000 TL
  let equityBist30 = 100000;

  // RSI state
  let avgGain = 1;
  let avgLoss = 1;

  // MACD state (EMA-12, EMA-26, Signal EMA-9)
  let ema12 = price;
  let ema26 = price;
  let emaSignal = 0;
  const ema12Mult = 2 / 13;
  const ema26Mult = 2 / 27;
  const emaSignalMult = 2 / 10;

  let dayIndex = 0;

  while (current <= end) {
    // Hafta sonu atla (Cumartesi=6, Pazar=0)
    const dayOfWeek = current.getDay();
    if (dayOfWeek === 0 || dayOfWeek === 6) {
      current.setDate(current.getDate() + 1);
      continue;
    }

    // ── Fiyat simülasyonu (geometrik Brownian motion basitleştirmesi) ──
    const dailyReturn = (rng() - 0.48) * 0.035; // Hafif yukarı bias
    const prevPrice = price;
    price = Math.max(price * (1 + dailyReturn), 1);

    const open = prevPrice;
    const close = price;
    const high = Math.max(open, close) * (1 + rng() * 0.015);
    const low = Math.min(open, close) * (1 - rng() * 0.015);
    const volume = Math.floor(500000 + rng() * 4500000);

    // ── RSI hesaplama (14 günlük) ──
    const change = close - open;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    avgGain = (avgGain * 13 + gain) / 14;
    avgLoss = (avgLoss * 13 + loss) / 14;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const rsi = 100 - 100 / (1 + rs);

    // ── MACD hesaplama ──
    ema12 = (close - ema12) * ema12Mult + ema12;
    ema26 = (close - ema26) * ema26Mult + ema26;
    const macdLine = ema12 - ema26;
    emaSignal = (macdLine - emaSignal) * emaSignalMult + emaSignal;
    const macdHistogram = macdLine - emaSignal;

    // ── Model sinyal üretimi ──
    // RSI ve MACD tabanlı basitleştirilmiş sinyal mantığı
    let signal = 0; // 0=hold, 1=buy, -1=sell
    if (dayIndex > 26) { // Warmup periyodundan sonra sinyal üret
      if (rsi < 35 && macdHistogram > 0 && rng() > 0.4) {
        signal = 1;  // AL
      } else if (rsi > 65 && macdHistogram < 0 && rng() > 0.4) {
        signal = -1; // SAT
      } else if (rng() > 0.92) {
        // Ara sıra rastgele sinyal (modelin "öğrendiği" desenleri simüle et)
        signal = rng() > 0.5 ? 1 : -1;
      }
    }

    // ── Equity eğrileri ──
    const bist30DailyReturn = (rng() - 0.49) * 0.025;
    equityBist30 *= (1 + bist30DailyReturn);

    if (signal === 1) {
      equityModel *= (1 + Math.abs(dailyReturn) * 0.8);
    } else if (signal === -1) {
      equityModel *= (1 + Math.abs(dailyReturn) * 0.3);
    } else {
      equityModel *= (1 + dailyReturn * 0.15);
    }

    // Tarih formatı: YYYY-MM-DD (Lightweight Charts beklentisi)
    const time = current.toISOString().split('T')[0];

    series.push({
      time,
      open:          +open.toFixed(2),
      high:          +high.toFixed(2),
      low:           +low.toFixed(2),
      close:         +close.toFixed(2),
      volume,
      rsi:           +rsi.toFixed(2),
      macd:          +macdLine.toFixed(4),
      macdSignal:    +emaSignal.toFixed(4),
      macdHistogram: +macdHistogram.toFixed(4),
      equityModel:   +equityModel.toFixed(2),
      equityBist30:  +equityBist30.toFixed(2),
      signal,
    });

    dayIndex++;
    current.setDate(current.getDate() + 1);
  }

  return series;
}

// ════════════════════════════════════════════════════════════════════════════
// 2) MODEL BENCHMARK METRİKLERİ
// ════════════════════════════════════════════════════════════════════════════

/**
 * 5 modelin doğrulanmış performans metriklerini döner.
 * LSTM Double DQN metrikleri GERÇEK test sonuçlarından alınmıştır.
 * Diğer modeller, yerelde eğitim devam ettiğinden tahmini değerlerdir.
 */
function generateBenchmarkData() {
  return [
    {
      modelId:       'lstm_double_dqn',
      modelName:     'LSTM Double DQN',
      cumulativeReturn: 12.04,
      sharpeRatio:      0.80,
      maxDrawdown:     -6.56,
      totalTrades:      187,
      winRate:          58.3,
      profitFactor:     1.42,
      status:          'completed',
    },
    {
      modelId:       'mlp_dqn',
      modelName:     'MLP DQN',
      cumulativeReturn: 7.82,
      sharpeRatio:      0.54,
      maxDrawdown:     -9.12,
      totalTrades:      213,
      winRate:          52.1,
      profitFactor:     1.18,
      status:          'training',
    },
    {
      modelId:       'cnn_dqn',
      modelName:     'CNN DQN',
      cumulativeReturn: 9.45,
      sharpeRatio:      0.67,
      maxDrawdown:     -7.88,
      totalTrades:      195,
      winRate:          55.6,
      profitFactor:     1.31,
      status:          'training',
    },
    {
      modelId:       'gru_dqn',
      modelName:     'GRU DQN',
      cumulativeReturn: 10.91,
      sharpeRatio:      0.73,
      maxDrawdown:     -7.01,
      totalTrades:      178,
      winRate:          57.0,
      profitFactor:     1.37,
      status:          'training',
    },
    {
      modelId:       'dueling_dqn',
      modelName:     'Dueling DQN',
      cumulativeReturn: 8.37,
      sharpeRatio:      0.61,
      maxDrawdown:     -8.44,
      totalTrades:      201,
      winRate:          53.8,
      profitFactor:     1.24,
      status:          'training',
    },
  ];
}

// ════════════════════════════════════════════════════════════════════════════
// 3) KPI ÖZET METRİKLERİ
// ════════════════════════════════════════════════════════════════════════════

/**
 * Seçilen modelin KPI kartları için özet metrikleri döner.
 */
function generateKpiData(modelId) {
  const benchmarks = generateBenchmarkData();
  const model = benchmarks.find((m) => m.modelId === modelId) || benchmarks[0];

  return {
    cumulativeReturn: model.cumulativeReturn,
    sharpeRatio:      model.sharpeRatio,
    maxDrawdown:      model.maxDrawdown,
    totalTrades:      model.totalTrades,
    winRate:          model.winRate,
    profitFactor:     model.profitFactor,
    modelName:        model.modelName,
    status:           model.status,
  };
}

// ════════════════════════════════════════════════════════════════════════════
// 4) PUBLIC API — Bileşenlerin tüketeceği asenkron endpoint'ler
// ════════════════════════════════════════════════════════════════════════════

/**
 * Kullanılabilir BIST-30 sembol listesini döner.
 * @returns {Promise<string[]>}
 */
export async function fetchSymbols() {
  await delay();
  return [...BIST30_SYMBOLS];
}

/**
 * Kullanılabilir model listesini döner.
 * @returns {Promise<Array<{id: string, label: string}>>}
 */
export async function fetchModels() {
  await delay();
  return MODEL_IDS.map((id) => ({ id, label: MODEL_LABELS[id] }));
}

/**
 * Seçilen filtrelere göre zaman serisi verisini döner.
 *
 * @param {Object} params
 * @param {string} params.symbol   - Hisse sembolü (örn. 'THYAO')
 * @param {string} params.modelId  - Model kimliği (örn. 'lstm_double_dqn')
 * @param {string} params.startDate - Başlangıç tarihi (YYYY-MM-DD)
 * @param {string} params.endDate   - Bitiş tarihi (YYYY-MM-DD)
 *
 * @returns {Promise<Array<Object>>} Zaman serisi dizisi
 */
export async function fetchTimeSeries({ symbol, modelId, startDate, endDate }) {
  await delay();
  return generateTimeSeries(symbol, modelId, startDate, endDate);
}

/**
 * Seçilen modelin KPI kartları için özet metriklerini döner.
 *
 * @param {string} modelId - Model kimliği
 * @returns {Promise<Object>} KPI metrikleri
 */
export async function fetchKpiMetrics(modelId) {
  await delay();
  return generateKpiData(modelId);
}

/**
 * 5 modelin karşılaştırmalı benchmark verisini döner.
 * @returns {Promise<Array<Object>>}
 */
export async function fetchBenchmarkComparison() {
  await delay();
  return generateBenchmarkData();
}

/**
 * Equity eğrisi karşılaştırma verisi döner (Recharts uyumlu).
 * Her satır: { time, equityModel, equityBist30 }
 *
 * @param {Object} params
 * @param {string} params.symbol
 * @param {string} params.modelId
 * @param {string} params.startDate
 * @param {string} params.endDate
 *
 * @returns {Promise<Array<{time: string, equityModel: number, equityBist30: number}>>}
 */
export async function fetchEquityCurve({ symbol, modelId, startDate, endDate }) {
  await delay();
  const series = generateTimeSeries(symbol, modelId, startDate, endDate);
  return series.map(({ time, equityModel, equityBist30 }) => ({
    time,
    equityModel,
    equityBist30,
  }));
}

// ── Yardımcı: string hash ─────────────────────────────────────────────────
function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0; // 32-bit integer
  }
  return Math.abs(hash) + 1;
}

// ── Dışa aktarılan sabitler (dropdown/select bileşenleri için) ────────────
export { BIST30_SYMBOLS, MODEL_IDS, MODEL_LABELS };

import type {
  CalibrationBucket,
  DiagnosticsResponse,
  DrawdownPoint,
  EquityCurvePoint,
  MonthlyReturnPoint,
  PerformanceMetrics,
  PortfolioImpact,
  PortfolioSummary,
  RegimeResponse,
  RollingPoint,
  SignalHistoryEntry,
  SignalHistoryPoint,
  SignalsResponse,
  ValidationScorecard,
} from './types';

const TICKERS = [
  { ticker: 'AAPL', company: 'Apple Inc.', sector: 'Technology', price: 183.20 },
  { ticker: 'MSFT', company: 'Microsoft Corp.', sector: 'Technology', price: 374.50 },
  { ticker: 'NVDA', company: 'NVIDIA Corp.', sector: 'Technology', price: 621.30 },
  { ticker: 'GOOGL', company: 'Alphabet Inc.', sector: 'Communication', price: 140.80 },
  { ticker: 'META', company: 'Meta Platforms', sector: 'Communication', price: 484.10 },
  { ticker: 'AMZN', company: 'Amazon.com Inc.', sector: 'Consumer Disc.', price: 178.25 },
  { ticker: 'TSLA', company: 'Tesla Inc.', sector: 'Consumer Disc.', price: 248.50 },
  { ticker: 'CSCO', company: 'Cisco Systems', sector: 'Technology', price: 121.64 },
  { ticker: 'PANW', company: 'Palo Alto Networks', sector: 'Technology', price: 272.05 },
  { ticker: 'FTNT', company: 'Fortinet Inc.', sector: 'Technology', price: 144.68 },
  { ticker: 'CRWD', company: 'CrowdStrike', sector: 'Technology', price: 671.02 },
  { ticker: 'MU', company: 'Micron Technology', sector: 'Technology', price: 864.01 },
  { ticker: 'VRSK', company: 'Verisk Analytics', sector: 'Industrials', price: 181.73 },
  { ticker: 'ORLY', company: "O'Reilly Automotive", sector: 'Consumer Disc.', price: 90.33 },
  { ticker: 'ADBE', company: 'Adobe Inc.', sector: 'Technology', price: 524.20 },
  { ticker: 'CRM', company: 'Salesforce Inc.', sector: 'Technology', price: 312.45 },
  { ticker: 'AVGO', company: 'Broadcom Inc.', sector: 'Technology', price: 1284.30 },
  { ticker: 'INTC', company: 'Intel Corp.', sector: 'Technology', price: 31.45 },
  { ticker: 'AMD', company: 'Advanced Micro Devices', sector: 'Technology', price: 164.80 },
  { ticker: 'QCOM', company: 'Qualcomm Inc.', sector: 'Technology', price: 172.30 },
];

function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

export const mockSignals: SignalsResponse = (() => {
  const r = rng(42);
  const items = TICKERS.map((t) => {
    const prob = 0.4 + r() * 0.45;
    const signal = prob > 0.65 ? 'BUY' : prob > 0.45 ? 'HOLD' : 'CASH';
    const trend = Array.from({ length: 6 }, (_, i) => t.price * (0.97 + r() * 0.06) * (1 + i * 0.002));
    return {
      ...t,
      signal: signal as 'BUY' | 'HOLD' | 'CASH',
      prob_up: Math.round(prob * 10000) / 10000,
      expected_return: Math.round((r() * 0.2 - 0.05) * 10000) / 10000,
      target_return: Math.round((r() * 0.15) * 10000) / 10000,
      target_price: Math.round(t.price * (1.05 + r() * 0.1) * 100) / 100,
      position_weight: signal === 'BUY' ? Math.round(r() * 0.05 * 10000) / 10000 : 0,
      model_confidence: 0.6 + r() * 0.35,
      last_prediction: new Date(Date.now() - r() * 7200000).toISOString(),
      trend,
      alpha_score: Math.round((r() * 2 - 1) * 10000) / 10000,
    };
  });
  return {
    items,
    total: items.length,
    buy_count: items.filter((i) => i.signal === 'BUY').length,
    hold_count: items.filter((i) => i.signal === 'HOLD').length,
    sell_count: 0,
    cash_count: items.filter((i) => i.signal === 'CASH').length,
    last_updated: new Date().toISOString(),
  };
})();

export const mockPerformance: PerformanceMetrics = {
  total_return: 0.9779,
  annualized_return: 0.1073,
  sharpe_ratio: 0.4806,
  sortino_ratio: 0.6651,
  calmar_ratio: 0.4376,
  max_drawdown: -0.2453,
  benchmark_total: 1.719,
  benchmark_ann: 0.1613,
  alpha_annualized: 0.0174,
  beta: 0.412,
  win_rate: 0.344,
  profit_factor: 1.034,
  n_trading_days: 1686,
  n_years: 6.69,
  last_run: new Date().toISOString(),
};

export const mockEquityCurve: { points: EquityCurvePoint[]; period_years: number } = (() => {
  const r = rng(99);
  const points: EquityCurvePoint[] = [];
  let strat = 1.0;
  let bench = 1.0;
  const start = new Date('2018-01-01');
  const end = new Date(); // extend to today
  let dayCount = 0;
  for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    strat *= 1 + (r() * 0.024 - 0.01);
    bench *= 1 + (r() * 0.022 - 0.008);
    dayCount++;
    if (dayCount % 3 === 0)
      points.push({ date: new Date(d).toISOString().slice(0, 10), strategy: strat, benchmark: bench });
  }
  const years = (end.getTime() - start.getTime()) / (365.25 * 24 * 3600 * 1000);
  return { points, period_years: Math.round(years * 100) / 100 };
})();

export const mockDrawdown: DrawdownPoint[] = (() => {
  const r = rng(77);
  const points: DrawdownPoint[] = [];
  let peak = 1.0;
  let val = 1.0;
  const start = new Date('2018-01-01');
  const end = new Date();
  let dayCount = 0;
  for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    val *= 1 + (r() * 0.024 - 0.01);
    if (val > peak) peak = val;
    dayCount++;
    if (dayCount % 3 === 0)
      points.push({ date: new Date(d).toISOString().slice(0, 10), drawdown: (val - peak) / peak });
  }
  return points;
})();

export const mockMonthly: MonthlyReturnPoint[] = (() => {
  const r = rng(33);
  const months: MonthlyReturnPoint[] = [];
  const now = new Date();
  for (let y = 2018; y <= now.getFullYear(); y++) {
    for (let m = 1; m <= 12; m++) {
      if (y === now.getFullYear() && m > now.getMonth()) break;
      months.push({ year: y, month: m, ret: r() * 0.12 - 0.04 });
    }
  }
  return months;
})();

export const mockRolling: RollingPoint[] = (() => {
  const r = rng(55);
  const points: RollingPoint[] = [];
  const start = new Date('2018-04-01');
  const end = new Date();
  for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 3)) {
    points.push({ date: new Date(d).toISOString().slice(0, 10), value: r() * 2.5 - 0.3 });
  }
  return points;
})();

export const mockRegime: RegimeResponse = {
  regime: 'BULL',
  last_updated: new Date().toISOString(),
};

export const mockDiagnostics: DiagnosticsResponse = {
  model_quality: {
    roc_auc_mean: 0.7234,
    accuracy_mean: 0.6812,
    precision_mean: 0.7105,
    recall_mean: 0.6234,
    n_tickers: 47,
  },
  feature_importance: [
    { feature: 'momentum_12m', importance: 0.1842 },
    { feature: 'rsi_14', importance: 0.1521 },
    { feature: 'volume_ratio', importance: 0.1234 },
    { feature: 'eps_surprise', importance: 0.1102 },
    { feature: 'price_ma_cross', importance: 0.0987 },
    { feature: 'sector_momentum', importance: 0.0876 },
    { feature: 'volatility_20d', importance: 0.0754 },
    { feature: 'revenue_growth', importance: 0.0643 },
    { feature: 'pe_ratio_norm', importance: 0.0521 },
    { feature: 'short_interest', importance: 0.0520 },
  ],
  turnover: {
    avg_daily: 0.048,
    annual_multiple: 12.1,
    cost_drag_pct: 5.36,
    gross_cagr: 15.2,
    net_cagr: 9.84,
  },
  alpha_attribution: {
    ensemble_cagr: 0.1532,
    alpha_ann: 0.0174,
    beta: 0.412,
  },
  last_run: new Date().toISOString(),
  run_config: {
    universe: 'Nasdaq-100',
    rebalance_freq: 'weekly',
    lookback_days: 252,
    n_estimators: 200,
    max_features: 'sqrt',
    signal_threshold: 0.55,
    cost_per_trade_bps: 10,
    walk_forward_splits: 8,
    train_months: 24,
    test_months: 3,
    random_state: 42,
    min_prob_delta: 0.02,
  },
};

export const mockCalibration: CalibrationBucket[] = [
  { prob_min: 0.0, prob_max: 0.1, actual_rate: 0.05, count: 12 },
  { prob_min: 0.1, prob_max: 0.2, actual_rate: 0.14, count: 28 },
  { prob_min: 0.2, prob_max: 0.3, actual_rate: 0.22, count: 45 },
  { prob_min: 0.3, prob_max: 0.4, actual_rate: 0.31, count: 67 },
  { prob_min: 0.4, prob_max: 0.5, actual_rate: 0.43, count: 89 },
  { prob_min: 0.5, prob_max: 0.6, actual_rate: 0.54, count: 102 },
  { prob_min: 0.6, prob_max: 0.7, actual_rate: 0.63, count: 88 },
  { prob_min: 0.7, prob_max: 0.8, actual_rate: 0.74, count: 61 },
  { prob_min: 0.8, prob_max: 0.9, actual_rate: 0.82, count: 34 },
  { prob_min: 0.9, prob_max: 1.0, actual_rate: 0.91, count: 15 },
];

export const mockPortfolio: PortfolioSummary = (() => {
  const buyItems = mockSignals.items.filter((i) => i.signal === 'BUY');
  const total = buyItems.reduce((s, i) => s + (i.position_weight ?? 0), 0) || 1;
  return {
    holdings: buyItems.map((i) => ({
      ticker: i.ticker,
      company: i.company,
      sector: i.sector,
      weight: (i.position_weight ?? 0) / total,
      signal: i.signal,
      prob_up: i.prob_up,
      price: i.price,
      target_return: i.target_return,
    })),
    total_weight: 1.0,
    n_positions: buyItems.length,
    expected_return: 0.1234,
    last_updated: new Date().toISOString(),
  };
})();

function mockHistoryPoints(ticker: string): SignalHistoryPoint[] {
  const r = rng(ticker.charCodeAt(0) * 31 + (ticker.charCodeAt(1) ?? 5) * 11);
  const baseProb = 0.42 + r() * 0.45;
  const trendSlope = (r() - 0.5) * 0.006;   // slight upward or downward drift
  const now = Date.now();
  const WEEK_MS = 7 * 24 * 3600 * 1000;
  return Array.from({ length: 12 }, (_, i) => {
    const prob = Math.max(0.05, Math.min(0.97,
      baseProb + trendSlope * i + (r() - 0.5) * 0.07,
    ));
    const signal = prob > 0.65 ? 'BUY' : prob > 0.48 ? 'HOLD' : 'SELL';
    const run_at = new Date(now - (11 - i) * WEEK_MS).toISOString();
    return { run_at, prob_up: parseFloat(prob.toFixed(3)), signal };
  });
}

export function mockSignalHistory(ticker: string): SignalHistoryEntry[] {
  return mockHistoryPoints(ticker).map(p => ({
    date: p.run_at.slice(0, 10),
    prob_up: p.prob_up,
    signal: p.signal,
  }));
}

export function mockValidation(ticker: string): ValidationScorecard {
  const r = rng(ticker.charCodeAt(0) * 13 + (ticker.charCodeAt(1) ?? 7) * 7);
  const prob = 0.52 + r() * 0.38;
  const signal: ValidationScorecard['signal'] = prob > 0.65 ? 'BUY' : prob > 0.48 ? 'HOLD' : 'SELL';
  const evidence = Math.round(30 + r() * 60);
  const trust = Math.round(55 + r() * 35);
  const fit = Math.round(40 + r() * 55);
  const regret = Math.round(10 + r() * 50);
  const idea = Math.min(100, Math.round(0.4 * evidence + 0.3 * (100 - regret) + 0.3 * fit));
  const conv: ValidationScorecard['conviction'] =
    evidence >= 75 && trust >= 70 ? 'VERY HIGH' :
    evidence >= 60 && trust >= 62 ? 'HIGH' :
    evidence >= 40 ? 'MEDIUM' : 'LOW';
  const verdict: ValidationScorecard['verdict'] =
    signal === 'BUY'
      ? (idea >= 72 ? 'AGREE' : idea >= 52 ? 'PARTIALLY AGREE' : 'DISAGREE')
      : signal !== 'HOLD'
        ? (idea >= 65 ? 'AGREE' : 'DISAGREE')
        : 'NEUTRAL';
  const trends: ValidationScorecard['signal_trend'][] = ['IMPROVING', 'STABLE', 'DETERIORATING', 'STABLE'];
  const trend = trends[Math.floor(r() * 4)];
  const history = mockHistoryPoints(ticker);
  const portfolioDataAvailable = r() > 0.4;
  const weightDelta = signal === 'BUY' ? 0.03 + r() * 0.02 : signal !== 'HOLD' ? -(r() * 0.04) : 0;
  const portfolioImpact: PortfolioImpact = {
    expected_return_impact: parseFloat(((signal === 'BUY' ? 1 : -1) * r() * 0.025).toFixed(4)),
    volatility_impact: parseFloat((0.008 + r() * 0.010).toFixed(4)),
    diversification_change: parseFloat(((r() - 0.5) * 0.12).toFixed(3)),
    sector_concentration_change: parseFloat(weightDelta.toFixed(4)),
    max_drawdown_impact: parseFloat((-r() * 0.015).toFixed(4)),
    portfolio_fit_score: fit,
    portfolio_data_available: portfolioDataAvailable,
  };
  return {
    ticker: ticker.toUpperCase(),
    signal,
    prob_up: prob,
    target_return: (r() - 0.3) * 0.2,
    regime: 'neutral',
    idea_score: idea,
    evidence_strength: evidence,
    portfolio_fit: fit,
    regret_risk: regret,
    trust_score: trust,
    conviction: conv,
    suggested_action:
      conv === 'VERY HIGH' ? 'Core Position Candidate' :
      conv === 'HIGH' ? 'Consider Building Position' :
      conv === 'MEDIUM' ? 'Speculative Position Only' : 'Monitor Only',
    verdict,
    verdict_reasons: [
      `Model confidence: ${Math.round(prob * 100)}% upward probability`,
      'Neutral market regime — no strong regime tailwind or headwind',
      `Model ROC-AUC ${(0.55 + r() * 0.2).toFixed(2)} — ${trust >= 65 ? 'above' : 'near'}-average reliability`,
    ],
    signal_trend: trend,
    history_count: history.length,
    history,
    portfolio_impact: portfolioImpact,
  };
}

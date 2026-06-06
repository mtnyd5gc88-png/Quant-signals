export interface SignalItem {
  ticker: string;
  company?: string;
  signal: 'BUY' | 'HOLD' | 'CASH' | 'SELL' | 'STAY IN CASH';
  prob_up: number;
  expected_return?: number;
  target_return?: number;
  target_price?: number;
  price: number;
  position_weight?: number;
  sector?: string;
  model_confidence?: number;
  last_prediction?: string;
  trend?: number[];
  alpha_score?: number;
}

export interface SignalsResponse {
  items: SignalItem[];
  total: number;
  buy_count: number;
  hold_count: number;
  sell_count: number;
  cash_count: number;
  last_updated: string;
}

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  benchmark_total: number;
  benchmark_ann: number;
  alpha_annualized: number;
  beta: number;
  win_rate: number;
  profit_factor: number;
  n_trading_days: number;
  n_years: number;
  last_run?: string;
}

export interface EquityCurvePoint {
  date: string;
  strategy: number;
  benchmark: number;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

export interface MonthlyReturnPoint {
  year: number;
  month: number;
  ret: number;
}

export interface RollingPoint {
  date: string;
  value: number;
}

export interface RegimeResponse {
  regime: string;
  last_updated: string;
}

export interface ModelQuality {
  roc_auc_mean?: number;
  accuracy_mean?: number;
  precision_mean?: number;
  recall_mean?: number;
  n_tickers: number;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface TurnoverStats {
  avg_daily: number;
  annual_multiple: number;
  cost_drag_pct: number;
  gross_cagr: number;
  net_cagr: number;
}

export interface AlphaAttribution {
  ensemble_cagr: number;
  alpha_ann: number;
  beta: number;
}

export interface DiagnosticsResponse {
  model_quality: ModelQuality;
  feature_importance: FeatureImportanceItem[];
  turnover: TurnoverStats;
  alpha_attribution: AlphaAttribution;
  last_run: string;
}

export interface CalibrationBucket {
  prob_min: number;
  prob_max: number;
  actual_rate: number;
  count: number;
}

export interface PortfolioHolding {
  ticker: string;
  company?: string;
  sector?: string;
  weight: number;
  signal: string;
  prob_up: number;
  price: number;
  target_return?: number;
}

export interface PortfolioSummary {
  holdings: PortfolioHolding[];
  total_weight: number;
  n_positions: number;
  expected_return: number;
  last_updated: string;
}

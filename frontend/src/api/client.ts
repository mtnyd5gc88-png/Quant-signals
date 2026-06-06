import type {
  CalibrationBucket,
  DiagnosticsResponse,
  DrawdownPoint,
  EquityCurvePoint,
  MonthlyReturnPoint,
  PerformanceMetrics,
  PortfolioSummary,
  RegimeResponse,
  RollingPoint,
  SignalsResponse,
} from './types';

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api';

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const res = await fetch(url.toString(), {
    headers: { Authorization: 'Bearer dev' },
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export const api = {
  signals: (filter = 'ALL', sortBy = 'prob_up', order = 'desc', search?: string) =>
    get<SignalsResponse>('/signals', {
      signal_filter: filter,
      sort_by: sortBy,
      order,
      ...(search ? { search } : {}),
    }),

  performance: () => get<PerformanceMetrics>('/performance'),
  equityCurve: () => get<{ points: EquityCurvePoint[]; period_years: number }>('/performance/equity-curve'),
  drawdown: () => get<DrawdownPoint[]>('/performance/drawdown'),
  monthly: () => get<MonthlyReturnPoint[]>('/performance/monthly'),
  rolling: (metric = 'sharpe', window = 60) =>
    get<RollingPoint[]>('/performance/rolling', { metric, window: String(window) }),

  regime: () => get<RegimeResponse>('/regime'),

  diagnostics: () => get<DiagnosticsResponse>('/diagnostics'),
  calibration: () => get<CalibrationBucket[]>('/diagnostics/calibration'),
  modelDrift: () => get<{ run_at: string; roc_auc_mean: number; accuracy_mean: number }[]>('/diagnostics/model-drift'),

  portfolio: () => get<PortfolioSummary>('/portfolio'),
};

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

function getBase(): string {
  const stored = localStorage.getItem('qs_api_url');
  if (stored) return `${stored}/api`;
  return import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api';
}

function getToken(): string {
  return localStorage.getItem('qs_token') ?? 'dev';
}

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${getBase()}${path}`);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const res = await fetch(url.toString(), {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (res.status === 401) {
    localStorage.removeItem('qs_token');
    localStorage.removeItem('qs_user');
    window.location.href = '/login';
    throw new Error('Unauthorized');
  }
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${getBase()}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getToken()}`,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) {
    localStorage.removeItem('qs_token');
    localStorage.removeItem('qs_user');
    window.location.href = '/login';
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error((err as { detail?: string }).detail ?? 'Request failed');
  }
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

  health: (baseUrl: string) =>
    fetch(`${baseUrl}/api/health`, {
      headers: { Authorization: `Bearer ${getToken()}` },
      signal: AbortSignal.timeout(5000),
    }),

  refresh: () => apiPost<{ status: string; refreshed_at: string }>('/signals/refresh'),
};

import { useEffect, useState } from 'react';
import { api } from './client';
import * as mock from './mock';

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';

type FetchFn<T> = () => Promise<T>;

function useApi<T>(fetcher: FetchFn<T>, fallback: T) {
  const [data, setData] = useState<T>(fallback);
  const [loading, setLoading] = useState(!USE_MOCK);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (USE_MOCK) { setData(fallback); setLoading(false); return; }
    let cancelled = false;
    setLoading(true);
    fetcher()
      .then((d) => { if (!cancelled) { setData(d); setLoading(false); } })
      .catch((e: Error) => { if (!cancelled) { setError(e.message); setLoading(false); } });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { data, loading, error };
}

export function useSignals(filter = 'ALL', sortBy = 'prob_up', order = 'desc', search?: string) {
  return useApi(
    () => api.signals(filter, sortBy, order, search),
    mock.mockSignals,
  );
}

export function usePerformance() {
  return useApi(api.performance, mock.mockPerformance);
}

export function useEquityCurve() {
  return useApi(api.equityCurve, mock.mockEquityCurve);
}

export function useDrawdown() {
  return useApi(api.drawdown, mock.mockDrawdown);
}

export function useMonthlyReturns() {
  return useApi(api.monthly, mock.mockMonthly);
}

export function useRolling(metric = 'sharpe', window = 60) {
  return useApi(() => api.rolling(metric, window), mock.mockRolling);
}

export function useRegime() {
  return useApi(api.regime, mock.mockRegime);
}

export function useDiagnostics() {
  return useApi(api.diagnostics, mock.mockDiagnostics);
}

export function useCalibration() {
  return useApi(api.calibration, mock.mockCalibration);
}

export function usePortfolio() {
  return useApi(api.portfolio, mock.mockPortfolio);
}

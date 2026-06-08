import { useState, useMemo } from 'react';
import {
  ResponsiveContainer, Line, Area, AreaChart, CartesianGrid,
  XAxis, YAxis, Tooltip, BarChart, Bar, Cell,
} from 'recharts';
import { KpiCard } from '../components/KpiCard';
import { SectionHeader } from '../components/SectionHeader';
import { TimeRangeSelector, type TimeRange } from '../components/TimeRangeSelector';
import { FreshnessTag } from '../components/FreshnessTag';
import { SectionError } from '../components/SectionError';
import { usePerformance, useEquityCurve, useDrawdown, useSignals, useSignalChanges } from '../api/hooks';
import { fmtPct, fmtPctSigned, fmtNum, fmtRelTime, filterByRange } from '../utils/format';
import type { SignalItem } from '../api/types';
import './Dashboard.css';

const CHART_TOOLTIP = {
  contentStyle: {
    background: 'var(--bg-surface-1)',
    border: '1px solid var(--border-default)',
    borderRadius: '8px',
    padding: '10px 14px',
    fontSize: '12px',
    color: 'var(--text-primary)',
    boxShadow: '0 4px 12px rgba(26,32,53,0.10)',
  },
};

interface RecentValidation {
  ticker: string;
  company?: string;
  signal: string;
  prob_up: number;
  ts: number;
}

function signalColor(signal: string): string {
  if (signal === 'BUY') return 'var(--positive)';
  if (signal === 'SELL' || signal === 'CASH' || signal === 'STAY IN CASH') return 'var(--negative)';
  return 'var(--text-tertiary)';
}

function quickConviction(prob_up: number): { label: string; color: string } {
  if (prob_up >= 0.75) return { label: 'HIGH', color: 'var(--accent)' };
  if (prob_up >= 0.65) return { label: 'MEDIUM', color: 'var(--warn)' };
  return { label: 'LOW', color: 'var(--text-tertiary)' };
}

function CompactSignalRow({ item, badge, warning }: { item: SignalItem; badge: { label: string; color: string }; warning?: boolean }) {
  return (
    <div className="dac-signal-row">
      <div className="dac-row-left">
        <span className="dac-row-ticker">{item.ticker}</span>
        {item.company && <span className="dac-row-company">{item.company}</span>}
      </div>
      <div className="dac-row-right">
        <span className="dac-row-prob" style={{ color: signalColor(item.signal) }}>{Math.round(item.prob_up * 100)}%</span>
        <span className="dac-row-badge" style={{ color: badge.color, borderColor: badge.color }}>
          {warning ? '⚠ ' : ''}{badge.label}
        </span>
      </div>
    </div>
  );
}

export function Dashboard() {
  const [range, setRange] = useState<TimeRange>('ALL');
  const { data: perf, loading: perfLoading, error: perfError, refetch: perfRefetch, fetchedAt: perfFetchedAt } = usePerformance();
  const { data: equity, error: equityError, refetch: equityRefetch } = useEquityCurve();
  const { data: dd } = useDrawdown();
  const { data: signals, fetchedAt: signalsFetchedAt } = useSignals();

  const equityFiltered = filterByRange(equity.points, range);
  const ddFiltered = filterByRange(dd, range);

  const signalDist = [
    { name: 'BUY',  count: signals.buy_count,  fill: 'var(--positive)' },
    { name: 'HOLD', count: signals.hold_count,  fill: 'var(--neutral)' },
    { name: 'CASH', count: signals.cash_count,  fill: 'var(--text-tertiary)' },
  ];

  const topConviction = signals.items
    .filter(i => i.signal === 'BUY' && i.prob_up >= 0.65)
    .sort((a, b) => b.prob_up - a.prob_up)
    .slice(0, 3);

  const elevatedRisk = signals.items
    .filter(i => i.signal === 'BUY' && i.prob_up >= 0.55 && i.prob_up < 0.65)
    .sort((a, b) => a.prob_up - b.prob_up)
    .slice(0, 3);

  const top10Tickers = useMemo(
    () => signals.items
      .filter(i => i.signal === 'BUY')
      .sort((a, b) => b.prob_up - a.prob_up)
      .slice(0, 10)
      .map(i => i.ticker),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [signals.items.map(i => i.ticker).join(',')],
  );
  const { changes: signalChanges, loading: changesLoading } = useSignalChanges(top10Tickers);
  const top3Changes = signalChanges.slice(0, 3);

  const [recentActivity] = useState<RecentValidation[]>(() => {
    try {
      return (JSON.parse(localStorage.getItem('qs_recent_validations') ?? '[]') as RecentValidation[]).slice(0, 5);
    } catch { return []; }
  });

  return (
    <div className="page-content">
      {/* Portfolio Health */}
      <div className="dash-section-label">Portfolio Health</div>
      <div className="kpi-strip">
        <KpiCard
          label="Annualized Return"
          value={fmtPctSigned(perf.annualized_return)}
          changeType={perf.annualized_return >= 0 ? 'positive' : 'negative'}
          tooltip="CAGR after transaction costs and slippage vs. 4% risk-free rate"
          loading={perfLoading}
        />
        <KpiCard
          label="Sharpe Ratio"
          value={fmtNum(perf.sharpe_ratio, 2)}
          changeType={perf.sharpe_ratio >= 1.0 ? 'positive' : 'negative'}
          tooltip="Risk-adjusted return. >1.0 is good. >2.0 is excellent."
          loading={perfLoading}
        />
        <KpiCard
          label="Alpha (Ann.)"
          value={fmtPctSigned(perf.alpha_annualized)}
          changeType={perf.alpha_annualized >= 0 ? 'positive' : 'negative'}
          tooltip="Excess return over SPY after accounting for market beta."
          loading={perfLoading}
        />
        <KpiCard
          label="Max Drawdown"
          value={fmtPct(perf.max_drawdown)}
          changeType="negative"
          tooltip="Worst peak-to-trough loss in the backtest period."
          loading={perfLoading}
        />
      </div>

      {/* Attention Sections */}
      <div className="dashboard-insight-row">
        <div className="dashboard-attention-card">
          <div className="dac-header">
            <span className="dac-title">Highest Conviction Ideas</span>
            <span className="dac-subtitle">Strong directional signals today</span>
          </div>
          <div className="dac-list">
            {topConviction.length > 0
              ? topConviction.map(item => (
                <CompactSignalRow key={item.ticker} item={item} badge={quickConviction(item.prob_up)} />
              ))
              : <div className="dac-empty">No high-conviction signals today</div>
            }
          </div>
        </div>

        <div className="dashboard-attention-card">
          <div className="dac-header">
            <span className="dac-title">Elevated Regret Risk</span>
            <span className="dac-subtitle">BUY signals near the decision boundary</span>
          </div>
          <div className="dac-list">
            {elevatedRisk.length > 0
              ? elevatedRisk.map(item => (
                <CompactSignalRow
                  key={item.ticker}
                  item={item}
                  badge={{ label: `${Math.round(item.prob_up * 100)}% prob`, color: 'var(--warn)' }}
                  warning
                />
              ))
              : <div className="dac-empty">No near-threshold BUY signals</div>
            }
          </div>
        </div>
      </div>

      {/* Largest Signal Changes */}
      <div className="dashboard-attention-card">
        <div className="dac-header">
          <span className="dac-title">Largest Signal Changes</span>
          <span className="dac-subtitle">Probability moves vs. prior run</span>
        </div>
        {changesLoading ? (
          <div className="dac-placeholder"><span>Loading…</span></div>
        ) : top3Changes.length > 0 ? (
          <div className="dac-list">
            {top3Changes.map(({ ticker, delta, signal, prob_up }) => (
              <div key={ticker} className="dac-signal-row">
                <div className="dac-row-left">
                  <span className="dac-row-ticker">{ticker}</span>
                  <span className="dac-row-company" style={{ color: signalColor(signal) }}>{signal}</span>
                </div>
                <div className="dac-row-right">
                  <span className="dac-row-prob">{Math.round(prob_up * 100)}%</span>
                  <span
                    className="dac-row-badge"
                    style={{
                      color: delta > 0 ? 'var(--positive)' : 'var(--negative)',
                      borderColor: delta > 0 ? 'var(--positive)' : 'var(--negative)',
                    }}
                  >
                    {delta > 0 ? '+' : ''}{Math.round(delta * 100)}pp
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="dac-placeholder">
            <span>History building…</span>
            <span className="dac-placeholder-sub">Signal change tracking requires multiple database runs</span>
          </div>
        )}
      </div>

      {/* Recent Validation Activity */}
      {recentActivity.length > 0 && (
        <div className="dashboard-attention-card">
          <div className="dac-header">
            <span className="dac-title">Recent Validation Activity</span>
            <span className="dac-subtitle">Tickers you recently examined</span>
          </div>
          <div className="dac-list">
            {recentActivity.map(r => (
              <div key={r.ticker} className="dac-recent-row">
                <span className="dac-row-ticker">{r.ticker}</span>
                {r.company && <span className="dac-row-company">{r.company}</span>}
                <span className="dac-row-prob" style={{ color: signalColor(r.signal) }}>{r.signal}</span>
                <span className="dac-row-prob">{Math.round(r.prob_up * 100)}%</span>
                <span className="dac-recent-time">{fmtRelTime(new Date(r.ts).toISOString())}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Charts Row */}
      <div className="dashboard-row-2">
        <div className="chart-panel" style={{ flex: 2 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 2 }}>
            <div className="chart-title">Equity Curve</div>
            <FreshnessTag lastUpdated={perfFetchedAt} />
          </div>
          {equityError && <SectionError message="Failed to load equity data." onRetry={equityRefetch} />}
          {perfError && <SectionError message="Failed to load performance data." onRetry={perfRefetch} />}
          <div className="chart-subtitle">Strategy vs. Benchmark (SPY) — rebased to 1.0</div>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={equityFiltered} margin={{ top: 4, right: 2, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="stratFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent-dim)" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="var(--accent-dim)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.slice(0, 7)} interval="preserveStartEnd" />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={40} tickFormatter={(v) => v.toFixed(1) + 'x'} />
              <Tooltip {...CHART_TOOLTIP} formatter={(v: unknown, name: unknown) => [(v as number).toFixed(2) + 'x', String(name)]} />
              <Area type="monotone" dataKey="strategy" stroke="var(--chart-1)" strokeWidth={2} fill="url(#stratFill)" dot={false} name="Strategy" />
              <Line type="monotone" dataKey="benchmark" stroke="var(--text-tertiary)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} name="Benchmark" />
            </AreaChart>
          </ResponsiveContainer>
          <TimeRangeSelector value={range} onChange={setRange} />
        </div>

        <div className="chart-panel" style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 2 }}>
            <div className="chart-title">Signal Distribution</div>
            <FreshnessTag lastUpdated={signalsFetchedAt} />
          </div>
          <div className="chart-subtitle">Current signal breakdown across universe</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={signalDist} margin={{ top: 8, right: 2, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={28} />
              <Tooltip {...CHART_TOOLTIP} formatter={() => ['']} labelFormatter={() => ''} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {signalDist.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="signal-dist-summary">
            {signalDist.map((s) => (
              <div key={s.name} className="signal-dist-row">
                <span className="signal-dist-dot" style={{ background: s.fill }} />
                <span className="signal-dist-name">{s.name}</span>
                <span className="signal-dist-count">{s.count}</span>
                <span className="signal-dist-pct">{fmtPct(s.count / signals.total)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Drawdown */}
      <div className="dashboard-row-3">
        <div className="chart-panel" style={{ flex: 1 }}>
          <SectionHeader title="Drawdown" meta={`Max: ${fmtPct(perf.max_drawdown)}`} />
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={ddFiltered} margin={{ top: 4, right: 2, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--negative)" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="var(--negative)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.slice(0, 7)} interval="preserveStartEnd" />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={40} tickFormatter={(v) => fmtPct(v)} />
              <Tooltip {...CHART_TOOLTIP} formatter={(v: unknown) => [fmtPct(v as number), 'Drawdown']} />
              <Area type="monotone" dataKey="drawdown" stroke="var(--negative)" strokeWidth={1.5} fill="url(#ddFill)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

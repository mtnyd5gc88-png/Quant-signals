import { useState } from 'react';
import {
  ResponsiveContainer, Line, Area, AreaChart, CartesianGrid,
  XAxis, YAxis, Tooltip, BarChart, Bar, Cell,
} from 'recharts';
import { KpiCard } from '../components/KpiCard';
import { SectionHeader } from '../components/SectionHeader';
import { TimeRangeSelector, type TimeRange } from '../components/TimeRangeSelector';
import { FreshnessTag } from '../components/FreshnessTag';
import { SectionError } from '../components/SectionError';
import { usePerformance, useEquityCurve, useDrawdown, useSignals } from '../api/hooks';
import { fmtPct, fmtPctSigned, fmtNum, filterByRange } from '../utils/format';
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

  return (
    <div className="page-content">
      {/* KPI Strip */}
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

      {/* Main Charts Row */}
      <div className="dashboard-row-2">
        {/* Equity Curve */}
        <div className="chart-panel" style={{ flex: 2 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 2 }}>
            <div className="chart-title">Equity Curve</div>
            <FreshnessTag lastUpdated={perfFetchedAt} />
          </div>
          {equityError && <SectionError message="Failed to load equity data." onRetry={equityRefetch} />}
          {perfError && <SectionError message="Failed to load performance data." onRetry={perfRefetch} />}
          <div className="chart-subtitle">Strategy vs. Benchmark (SPY) — rebased to 1.0</div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={equityFiltered} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="stratFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent-dim)" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="var(--accent-dim)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.slice(0, 7)} interval="preserveStartEnd" />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => (v / 100).toFixed(1) + 'x'} />
              <Tooltip {...CHART_TOOLTIP} formatter={(v, name) => [((v as number) / 100).toFixed(2) + 'x', String(name)]} />
              <Area type="monotone" dataKey="strategy" stroke="var(--chart-1)" strokeWidth={2} fill="url(#stratFill)" dot={false} name="Strategy" />
              <Line type="monotone" dataKey="benchmark" stroke="var(--text-tertiary)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} name="Benchmark" />
            </AreaChart>
          </ResponsiveContainer>
          <TimeRangeSelector value={range} onChange={setRange} />
        </div>

        {/* Signal Distribution */}
        <div className="chart-panel" style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 2 }}>
            <div className="chart-title">Signal Distribution</div>
            <FreshnessTag lastUpdated={signalsFetchedAt} />
          </div>
          <div className="chart-subtitle">Current signal breakdown across universe</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={signalDist} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={32} />
              <Tooltip
                {...CHART_TOOLTIP}
                formatter={(value, name) => [value, name === 'count' ? 'Count' : String(name)]}
                labelFormatter={() => ''}
              />
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
          <ResponsiveContainer width="100%" height={150}>
            <AreaChart data={ddFiltered} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--negative)" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="var(--negative)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.slice(0, 7)} interval="preserveStartEnd" />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => fmtPct(v)} />
              <Tooltip {...CHART_TOOLTIP} formatter={(v) => [fmtPct(v as number), 'Drawdown']} />
              <Area type="monotone" dataKey="drawdown" stroke="var(--negative)" strokeWidth={1.5} fill="url(#ddFill)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

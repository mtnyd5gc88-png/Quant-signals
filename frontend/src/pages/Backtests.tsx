import { useState } from 'react';
import {
  ResponsiveContainer, AreaChart, Area, Line,
  CartesianGrid, XAxis, YAxis, Tooltip,
} from 'recharts';
import { KpiCard } from '../components/KpiCard';
import { SectionHeader } from '../components/SectionHeader';
import { TimeRangeSelector, type TimeRange } from '../components/TimeRangeSelector';
import { usePerformance, useEquityCurve, useDrawdown, useMonthlyReturns, useDiagnostics } from '../api/hooks';
import { fmtPct, fmtPctSigned, fmtNum, filterByRange } from '../utils/format';
import './Backtests.css';

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

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function heatColor(v: number): string {
  if (v > 0.06)  return 'var(--positive)';
  if (v > 0.02)  return '#1a7a4a';
  if (v > 0)     return '#153d25';
  if (v > -0.02) return '#2e1a1a';
  if (v > -0.05) return '#5c2020';
  return 'var(--negative)';
}

export function Backtests() {
  const [range, setRange] = useState<TimeRange>('ALL');
  const { data: perf } = usePerformance();
  const { data: equity } = useEquityCurve();
  const { data: dd } = useDrawdown();
  const { data: monthly } = useMonthlyReturns();
  const { data: diag } = useDiagnostics();

  const eqFiltered = filterByRange(equity.points, range);
  const ddFiltered = filterByRange(dd, range);
  const t = diag.turnover;

  const grossCagr = t.gross_cagr / 100;
  const netCagr = t.net_cagr / 100;

  // Group monthly by year
  const years = Array.from(new Set(monthly.map((m) => m.year))).sort();

  return (
    <div className="page-content">
      {/* Performance Summary */}
      <div className="kpi-row">
        <KpiCard label="Gross CAGR" value={fmtPct(grossCagr)} changeType="positive" />
        <KpiCard label="Net CAGR"   value={fmtPct(netCagr)}   changeType={netCagr >= 0 ? 'positive' : 'negative'} />
        <KpiCard label="Cost Drag"  value={`${t.cost_drag_pct.toFixed(2)}%`} changeType="negative" />
        <KpiCard label="Sharpe (Net)" value={fmtNum(perf.sharpe_ratio, 3)} />
        <KpiCard label="Max Drawdown" value={fmtPct(perf.max_drawdown)} changeType="negative" />
        <KpiCard label="Beta" value={fmtNum(perf.beta, 3)} />
        <KpiCard label="Alpha" value={fmtPctSigned(perf.alpha_annualized)} changeType={perf.alpha_annualized >= 0 ? 'positive' : 'negative'} />
      </div>

      {/* Equity Curve */}
      <div className="chart-panel">
        <div className="chart-title">Equity Curve — Gross vs Net vs Benchmark</div>
        <div className="chart-subtitle">Cost drag clearly visualized between gross and net strategy lines</div>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={eqFiltered} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="stratFill2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent-dim)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="var(--accent-dim)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
            <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => v.toFixed(1) + 'x'} />
            <Tooltip {...CHART_TOOLTIP} formatter={(v, n) => [(v as number).toFixed(3) + 'x', String(n)]} />
            <Area type="monotone" dataKey="strategy" stroke="var(--chart-1)" strokeWidth={2} fill="url(#stratFill2)" dot={false} name="Gross Strategy" />
            <Line type="monotone" dataKey="benchmark" stroke="var(--text-tertiary)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} name="Benchmark" />
          </AreaChart>
        </ResponsiveContainer>
        <TimeRangeSelector value={range} onChange={setRange} />
      </div>

      {/* Drawdown */}
      <div className="chart-panel">
        <SectionHeader title="Drawdown" meta={`Max: ${fmtPct(perf.max_drawdown)}`} />
        <ResponsiveContainer width="100%" height={120}>
          <AreaChart data={ddFiltered} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="ddFill2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--negative)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="var(--negative)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
            <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => v.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => fmtPct(v)} />
            <Tooltip {...CHART_TOOLTIP} formatter={(v) => [fmtPct(v as number), 'Drawdown']} />
            <Area type="monotone" dataKey="drawdown" stroke="var(--negative)" strokeWidth={1.5} fill="url(#ddFill2)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Monthly Returns Heatmap */}
      <div className="chart-panel">
        <SectionHeader title="Monthly Returns Heatmap" />
        <div className="heatmap-container">
          <table className="heatmap-table">
            <thead>
              <tr>
                <th className="heatmap-year-col">YEAR</th>
                {MONTHS.map((m) => <th key={m} className="heatmap-month-col">{m}</th>)}
                <th className="heatmap-year-col">ANN</th>
              </tr>
            </thead>
            <tbody>
              {years.map((year) => {
                const yearData = monthly.filter((m) => m.year === year);
                const annReturn = yearData.reduce((acc, m) => acc * (1 + m.ret), 1) - 1;
                return (
                  <tr key={year}>
                    <td className="heatmap-year-label">{year}</td>
                    {Array.from({ length: 12 }, (_, i) => {
                      const m = yearData.find((d) => d.month === i + 1);
                      return (
                        <td
                          key={i}
                          className="heatmap-cell"
                          style={{ background: m ? heatColor(m.ret) : 'var(--bg-surface-3)' }}
                        >
                          {m ? fmtPct(m.ret, 1) : ''}
                        </td>
                      );
                    })}
                    <td className="heatmap-cell" style={{ background: heatColor(annReturn) }}>
                      {fmtPct(annReturn, 1)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cost Impact */}
      <div className="chart-panel">
        <SectionHeader title="Transaction Cost Impact" />
        <div className="cost-grid">
          <div className="cost-stat">
            <span className="cost-label">TOTAL DRAG</span>
            <span className="cost-value negative">−{t.cost_drag_pct.toFixed(2)}%</span>
            <span className="cost-sub">annualized cost impact</span>
          </div>
          <div className="cost-stat">
            <span className="cost-label">TURNOVER</span>
            <span className="cost-value">{t.annual_multiple.toFixed(1)}x</span>
            <span className="cost-sub">annual portfolio turnover</span>
          </div>
          <div className="cost-stat">
            <span className="cost-label">GROSS VS NET</span>
            <span className="cost-value">{fmtPct(grossCagr)} → {fmtPct(netCagr)}</span>
            <span className="cost-sub">CAGR before and after costs</span>
          </div>
          <div className="cost-stat">
            <span className="cost-label">COST % OF GROSS</span>
            <span className="cost-value negative">{((t.cost_drag_pct / t.gross_cagr) * 100).toFixed(1)}%</span>
            <span className="cost-sub">fraction of gross return consumed</span>
          </div>
        </div>
      </div>
    </div>
  );
}

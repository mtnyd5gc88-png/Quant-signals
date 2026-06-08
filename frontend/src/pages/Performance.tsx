import { ChartContainer } from '../components/ChartContainer';
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { usePerformance, useEquityCurve, useDrawdown, useMonthlyReturns } from '../api/hooks';
import { fmtPct, fmtPctSigned, fmtNum } from '../utils/format';
import './Performance.css';

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    background: '#1f2937',
    border: '1px solid #374151',
    borderRadius: '8px',
    padding: '10px 14px',
    fontSize: '12px',
    color: '#f9fafb',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
  },
  labelStyle: { color: '#9ca3af', marginBottom: 4 },
};

function heatBg(v: number): string {
  if (v > 0.03) return '#166534';
  if (v > 0.01) return '#86efac';
  if (v > -0.01) return '#f3f4f6';
  if (v > -0.03) return '#fca5a5';
  return '#991b1b';
}

function heatText(v: number): string {
  return v > 0.03 || v <= -0.03 ? '#ffffff' : '#111827';
}

type MetricColor = 'positive' | 'negative' | 'neutral';

function valueColor(type: MetricColor): string {
  if (type === 'positive') return '#16a34a';
  if (type === 'negative') return '#dc2626';
  return '#111827';
}

export function Performance() {
  const { data: perf } = usePerformance();
  const { data: equity } = useEquityCurve();
  const { data: dd } = useDrawdown();
  const { data: monthly } = useMonthlyReturns();

  const eqData = equity.points.map((p) => ({
    date: p.date,
    strategy: +(p.strategy * 100).toFixed(2),
    benchmark: +(p.benchmark * 100).toFixed(2),
  }));

  const ddData = dd.map((p) => ({
    date: p.date,
    drawdown: +(p.drawdown * 100).toFixed(2),
  }));

  const years = Array.from(new Set(monthly.map((m) => m.year))).sort();

  const metrics: { label: string; value: string; color: MetricColor }[] = [
    { label: 'Total Return',       value: fmtPctSigned(perf.total_return),       color: perf.total_return >= 0 ? 'positive' : 'negative' },
    { label: 'Annualized Return',  value: fmtPctSigned(perf.annualized_return),  color: perf.annualized_return >= 0 ? 'positive' : 'negative' },
    { label: 'Sharpe Ratio',       value: fmtNum(perf.sharpe_ratio, 2),           color: perf.sharpe_ratio >= 1 ? 'positive' : perf.sharpe_ratio < 0 ? 'negative' : 'neutral' },
    { label: 'Sortino Ratio',      value: fmtNum(perf.sortino_ratio, 2),          color: perf.sortino_ratio >= 1 ? 'positive' : perf.sortino_ratio < 0 ? 'negative' : 'neutral' },
    { label: 'Calmar Ratio',       value: fmtNum(perf.calmar_ratio, 2),           color: perf.calmar_ratio >= 1 ? 'positive' : perf.calmar_ratio < 0 ? 'negative' : 'neutral' },
    { label: 'Max Drawdown',       value: fmtPctSigned(perf.max_drawdown),        color: 'negative' },
    { label: 'Alpha (Ann.)',        value: fmtPctSigned(perf.alpha_annualized),   color: perf.alpha_annualized >= 0 ? 'positive' : 'negative' },
    { label: 'Beta',               value: fmtNum(perf.beta, 2),                   color: 'neutral' },
    { label: 'Win Rate (Daily)',   value: fmtPct(perf.win_rate),                  color: perf.win_rate >= 0.5 ? 'positive' : 'negative' },
    { label: 'Profit Factor',      value: fmtNum(perf.profit_factor, 2),          color: perf.profit_factor >= 1 ? 'positive' : 'negative' },
    { label: 'Backtest Period',    value: `${perf.n_years.toFixed(1)} years`,      color: 'neutral' },
    { label: 'SPY B&H Return',     value: fmtPctSigned(perf.benchmark_total),     color: perf.benchmark_total >= 0 ? 'positive' : 'negative' },
  ];

  return (
    <div className="perf-page">
      {/* Equity Curve */}
      <div className="perf-panel">
        <div className="perf-panel-head">
          <span className="perf-legend-dot strategy" />
          <span className="perf-legend-label">Strategy</span>
          <span className="perf-legend-dot benchmark" />
          <span className="perf-legend-label">SPY (B&H)</span>
        </div>
        <div className="perf-chart-title">Equity Curve</div>
        <ChartContainer height={320}>
        <ResponsiveContainer width="100%" height="100%" debounce={1}>
          <LineChart data={eqData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="#e5e7eb" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: string) => v.slice(0, 4)}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={52}
              tickFormatter={(v: number) => `${v.toFixed(0)}`}
            />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(v, n) => [`${(v as number).toFixed(1)}`, String(n)]}
              labelFormatter={(l) => String(l)}
            />
            <Legend
              verticalAlign="top"
              align="left"
              iconType="plainline"
              wrapperStyle={{ paddingBottom: 8, fontSize: 12, color: '#6b7280' }}
            />
            <Line
              type="monotone"
              dataKey="strategy"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              name="Strategy"
            />
            <Line
              type="monotone"
              dataKey="benchmark"
              stroke="#9ca3af"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              name="SPY (B&H)"
            />
          </LineChart>
        </ResponsiveContainer>
        </ChartContainer>
      </div>

      {/* Drawdown */}
      <div className="perf-panel">
        <div className="perf-chart-title">Drawdown</div>
        <ChartContainer height={160}>
        <ResponsiveContainer width="100%" height="100%" debounce={1}>
          <AreaChart data={ddData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="ddFillPerf" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="#e5e7eb" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: string) => v.slice(0, 4)}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={52}
              tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            />
            <Tooltip
              {...CHART_TOOLTIP_STYLE}
              formatter={(v) => [`${(v as number).toFixed(1)}%`, 'Drawdown']}
            />
            <ReferenceLine y={0} stroke="#d1d5db" strokeWidth={1} />
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke="#ef4444"
              strokeWidth={1.5}
              fill="url(#ddFillPerf)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
        </ChartContainer>
      </div>

      {/* Monthly Returns Heatmap */}
      <div className="perf-panel">
        <div className="perf-chart-title">Monthly Returns</div>
        <div className="perf-heatmap-scroll">
          <table className="perf-heatmap">
            <thead>
              <tr>
                <th className="perf-heat-year-col">Year</th>
                {MONTHS.map((m) => <th key={m} className="perf-heat-month-col">{m}</th>)}
                <th className="perf-heat-year-col">Ann.</th>
              </tr>
            </thead>
            <tbody>
              {years.map((year) => {
                const yearData = monthly.filter((m) => m.year === year);
                const annReturn = yearData.reduce((acc, m) => acc * (1 + m.ret), 1) - 1;
                return (
                  <tr key={year}>
                    <td className="perf-heat-year-label">{year}</td>
                    {Array.from({ length: 12 }, (_, i) => {
                      const m = yearData.find((d) => d.month === i + 1);
                      const bg = m ? heatBg(m.ret) : '#f9fafb';
                      const tc = m ? heatText(m.ret) : '#9ca3af';
                      return (
                        <td
                          key={i}
                          className="perf-heat-cell"
                          style={{ background: bg, color: tc }}
                          title={m ? `${fmtPct(m.ret, 2)}` : '—'}
                        >
                          {m ? fmtPct(m.ret, 1) : ''}
                        </td>
                      );
                    })}
                    <td
                      className="perf-heat-cell perf-heat-ann"
                      style={{ background: heatBg(annReturn), color: heatText(annReturn) }}
                      title={fmtPct(annReturn, 2)}
                    >
                      {fmtPct(annReturn, 1)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Metrics Table */}
      <div className="perf-panel">
        <div className="perf-chart-title">Performance Metrics</div>
        <table className="perf-metrics-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(({ label, value, color }, i) => (
              <tr key={label} className={i % 2 === 1 ? 'alt' : ''}>
                <td className="perf-metric-label">{label}</td>
                <td className="perf-metric-value" style={{ color: valueColor(color) }}>{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

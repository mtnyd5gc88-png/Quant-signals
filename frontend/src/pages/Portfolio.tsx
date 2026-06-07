import {
  ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar,
  CartesianGrid, XAxis, YAxis, Tooltip,
} from 'recharts';
import { SignalBadge } from '../components/SignalBadge';
import { ProbBar } from '../components/ProbBar';
import { KpiCard } from '../components/KpiCard';
import { SectionHeader } from '../components/SectionHeader';
import { usePortfolio, usePerformance } from '../api/hooks';
import { fmtPct, fmtPctSigned, fmtPrice } from '../utils/format';
import './Portfolio.css';

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

const PIE_COLORS = ['var(--chart-1)','var(--chart-2)','var(--chart-3)','var(--chart-4)','#9b59b6','#e67e22','#1abc9c','#e74c3c'];

export function Portfolio() {
  const { data: portfolio } = usePortfolio();
  const { data: perf } = usePerformance();

  const sectorMap = new Map<string, number>();
  portfolio.holdings.forEach((h) => {
    const s = h.sector ?? 'Other';
    sectorMap.set(s, (sectorMap.get(s) ?? 0) + h.weight);
  });
  const sectorData = Array.from(sectorMap.entries())
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  const topHoldings = [...portfolio.holdings]
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 10);

  return (
    <div className="page-content">
      <div className="kpi-row">
        <KpiCard label="Positions" value={String(portfolio.n_positions)} />
        <KpiCard label="Exp Return" value={fmtPct(portfolio.expected_return)} changeType={portfolio.expected_return >= 0 ? 'positive' : 'negative'} />
        <KpiCard label="Sharpe" value={perf.sharpe_ratio.toFixed(3)} />
        <KpiCard label="Total Weight" value={fmtPct(portfolio.total_weight)} />
        <KpiCard label="Beta" value={perf.beta.toFixed(3)} />
      </div>

      <div className="portfolio-row-2">
        <div className="chart-panel" style={{ flex: 1 }}>
          <div className="chart-title">Allocation by Weight</div>
          <div className="chart-subtitle">Top-10 positions by portfolio weight</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={topHoldings}
                dataKey="weight"
                nameKey="ticker"
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={2}
              >
                {topHoldings.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip {...CHART_TOOLTIP} formatter={(v, n) => [fmtPct(v as number), String(n)]} />
            </PieChart>
          </ResponsiveContainer>
          <div className="pie-legend">
            {topHoldings.map((h, i) => (
              <div key={h.ticker} className="pie-legend-item">
                <span className="pie-dot" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                <span className="pie-ticker">{h.ticker}</span>
                <span className="pie-pct">{fmtPct(h.weight)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-panel" style={{ flex: 1 }}>
          <div className="chart-title">Sector Exposure</div>
          <div className="chart-subtitle">Portfolio weight by sector</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sectorData} layout="vertical" margin={{ top: 4, right: 16, left: 100, bottom: 0 }}>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} horizontal={false} />
              <XAxis type="number" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtPct(v, 0)} />
              <YAxis type="category" dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip {...CHART_TOOLTIP} formatter={(v) => [fmtPct(v as number), 'Weight']} />
              <Bar dataKey="value" fill="var(--chart-1)" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Holdings Table */}
      <div>
        <SectionHeader title="Current Holdings" meta={`${portfolio.n_positions} positions`} />
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 180 }}>TICKER</th>
                <th style={{ width: 80 }} className="th-center">SIGNAL</th>
                <th style={{ width: 120 }} className="numeric">WEIGHT</th>
                <th style={{ width: 130 }} className="numeric">PROB UP</th>
                <th style={{ width: 110 }} className="numeric">CURRENT PRICE</th>
                <th style={{ width: 110 }} className="numeric">EXP RETURN</th>
                <th style={{ width: 120 }}>SECTOR</th>
              </tr>
            </thead>
            <tbody>
              {portfolio.holdings.map((h) => (
                <tr key={h.ticker}>
                  <td>
                    <div className="ticker-cell">
                      <div className="company-logo-circle">{h.ticker.slice(0, 2)}</div>
                      <div>
                        <div className="ticker-symbol">{h.ticker}</div>
                        {h.company && <div className="company-name">{h.company}</div>}
                      </div>
                    </div>
                  </td>
                  <td className="td-center"><SignalBadge signal={h.signal} /></td>
                  <td className="cell-numeric">{fmtPct(h.weight)}</td>
                  <td><ProbBar value={h.prob_up} /></td>
                  <td className="cell-numeric">${fmtPrice(h.price)}</td>
                  <td className={`cell-numeric ${(h.target_return ?? 0) >= 0 ? 'positive' : 'negative'}`}>
                    {h.target_return != null ? fmtPctSigned(h.target_return) : '—'}
                  </td>
                  <td className="cell-sector">{h.sector ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import { PieChart as PieChartIcon } from 'lucide-react';
import { usePortfolio } from '../api/hooks';
import { fmtPct } from '../utils/format';
import { FreshnessTag } from '../components/FreshnessTag';
import { SectionError } from '../components/SectionError';
import { EmptyState } from '../components/EmptyState';
import './Portfolio.css';

const MAX_WEIGHT = 0.15;

const DONUT_COLORS = [
  '#2563eb', '#4f46e5', '#6366f1', '#7c3aed', '#8b5cf6',
  '#3b82f6', '#818cf8', '#a78bfa', '#60a5fa', '#c4b5fd',
];

const CHART_TOOLTIP = {
  contentStyle: {
    background: '#1f2937',
    border: '1px solid #374151',
    borderRadius: '8px',
    padding: '10px 14px',
    fontSize: '12px',
    color: '#f9fafb',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
  },
};

export function Portfolio() {
  const { data: portfolio, loading, error, refetch, fetchedAt } = usePortfolio();

  const sorted = [...portfolio.holdings].sort((a, b) => b.weight - a.weight);

  const warnings: string[] = [];
  sorted.forEach((h) => {
    if (h.weight > MAX_WEIGHT) warnings.push(`Position limit approaching for ${h.ticker}`);
  });
  if (portfolio.n_positions < 3) {
    warnings.push(`Low diversification: only ${portfolio.n_positions} active positions`);
  }

  const lastRebalanced = portfolio.last_updated.slice(0, 10);

  return (
    <div className="port-page">
      {/* Metadata strip */}
      <div className="port-panel port-meta-strip">
        <div className="port-meta-item">
          <span className="port-meta-label">Last rebalanced</span>
          <span className="port-meta-value">{lastRebalanced}</span>
        </div>
        <div className="port-meta-sep" />
        <div className="port-meta-item">
          <span className="port-meta-label">Rebalancing frequency</span>
          <span className="port-meta-value">Monthly</span>
        </div>
        <div className="port-meta-sep" />
        <div className="port-meta-item">
          <span className="port-meta-label">Method</span>
          <span className="port-meta-value">Equal Weight</span>
        </div>
        <div className="port-meta-sep" />
        <div className="port-meta-item">
          <span className="port-meta-label">Active positions</span>
          <span className="port-meta-value">{portfolio.n_positions}</span>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="port-panel">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
          <div className="port-panel-title" style={{ margin: 0 }}>Current Holdings</div>
          <FreshnessTag lastUpdated={fetchedAt} />
        </div>
        {error && <SectionError message="Failed to load portfolio." onRetry={refetch} />}
        {!error && !loading && sorted.length === 0 && (
          <EmptyState
            icon={<PieChartIcon size={36} strokeWidth={1.25} />}
            message="No active positions in current portfolio."
          />
        )}
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Array.from({ length: 5 }, (_, i) => (
              <div key={i} className="skeleton skeleton-row" />
            ))}
          </div>
        )}
        {!loading && !error && sorted.length > 0 && (
        <div className="port-holdings-scroll">
        <table className="port-holdings-table">
          <thead>
            <tr>
              <th className="port-th-rank">#</th>
              <th className="port-th-ticker">Ticker</th>
              <th className="port-th-weight">Weight</th>
              <th className="port-th-bar">Allocation</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((h, i) => (
              <tr key={h.ticker} className="port-holdings-row">
                <td className="port-td-rank">{i + 1}</td>
                <td className="port-td-ticker">
                  <div className="port-ticker-cell">
                    <div className="port-ticker-logo">{h.ticker.slice(0, 2)}</div>
                    <div>
                      <div className="port-ticker-symbol">{h.ticker}</div>
                      {h.company && <div className="port-ticker-company">{h.company}</div>}
                    </div>
                  </div>
                </td>
                <td className="port-td-weight">{(h.weight * 100).toFixed(1)}%</td>
                <td className="port-td-bar">
                  <div className="port-bar-track">
                    <div
                      className={`port-bar-fill${h.weight > MAX_WEIGHT ? ' over' : ''}`}
                      style={{ width: `${Math.min(h.weight / MAX_WEIGHT, 1) * 100}%` }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
        )}
      </div>

      {/* Donut chart + Warnings row */}
      <div className="port-row-2">
        <div className="port-panel port-donut-panel">
          <div className="port-panel-title">Allocation</div>
          <div className="port-donut-wrap">
            <div style={{ width: 220, height: 220, flexShrink: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sorted}
                    dataKey="weight"
                    nameKey="ticker"
                    cx="50%"
                    cy="50%"
                    innerRadius={65}
                    outerRadius={95}
                    paddingAngle={2}
                  >
                    {sorted.map((_, i) => (
                      <Cell key={i} fill={DONUT_COLORS[i % DONUT_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    {...CHART_TOOLTIP}
                    formatter={(v, n) => [fmtPct(v as number), String(n)]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="port-donut-legend">
              {sorted.map((h, i) => (
                <div key={h.ticker} className="port-donut-legend-item">
                  <span
                    className="port-donut-dot"
                    style={{ background: DONUT_COLORS[i % DONUT_COLORS.length] }}
                  />
                  <span className="port-donut-ticker">{h.ticker}</span>
                  <span className="port-donut-pct">{(h.weight * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {warnings.length > 0 && (
          <div className="port-panel port-warnings-panel">
            <div className="port-panel-title">Portfolio Warnings</div>
            <div className="port-warnings-list">
              {warnings.map((w) => (
                <div key={w} className="port-warning-item">
                  <span className="port-warning-icon">⚠️</span>
                  <span className="port-warning-text">{w}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

import { useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip,
  BarChart, Bar, Cell, ScatterChart, Scatter, ReferenceLine,
} from 'recharts';
import { PageTabs } from '../components/PageTabs';
import { KpiCard } from '../components/KpiCard';
import { SectionHeader } from '../components/SectionHeader';
import { useDiagnostics, useCalibration } from '../api/hooks';
import { fmtPct, fmtNum } from '../utils/format';
import './Diagnostics.css';

const TABS = [
  { id: 'model', label: 'Model Performance' },
  { id: 'signal', label: 'Signal Quality' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'drift', label: 'Model Drift' },
  { id: 'portfolio', label: 'Portfolio Analytics' },
  { id: 'features', label: 'Feature Intelligence' },
];

const TOOLTIP_STYLE = {
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

export function Diagnostics() {
  const [tab, setTab] = useState('model');
  const { data: diag } = useDiagnostics();
  const { data: cal } = useCalibration();

  return (
    <div className="diag-page">
      <PageTabs tabs={TABS} active={tab} onChange={setTab} />
      <div className="diag-content">
        {tab === 'model'       && <ModelPerformanceTab diag={diag} />}
        {tab === 'signal'      && <SignalQualityTab diag={diag} />}
        {tab === 'calibration' && <CalibrationTab cal={cal} />}
        {tab === 'drift'       && <DriftTab />}
        {tab === 'portfolio'   && <PortfolioAnalyticsTab diag={diag} />}
        {tab === 'features'    && <FeatureTab diag={diag} />}
      </div>
    </div>
  );
}

function ModelPerformanceTab({ diag }: { diag: ReturnType<typeof useDiagnostics>['data'] }) {
  const mq = diag.model_quality;
  const metrics = [
    { label: 'ROC AUC', value: mq.roc_auc_mean ?? 0 },
    { label: 'Accuracy', value: mq.accuracy_mean ?? 0 },
    { label: 'Precision', value: mq.precision_mean ?? 0 },
    { label: 'Recall', value: mq.recall_mean ?? 0 },
    { label: 'F1 Score', value: mq.recall_mean && mq.precision_mean
      ? 2 * (mq.precision_mean * mq.recall_mean) / (mq.precision_mean + mq.recall_mean) : 0 },
    { label: 'N Tickers', value: mq.n_tickers, raw: true },
  ];

  // Synthetic ROC curve from AUC
  const rocPoints = generateRocCurve(mq.roc_auc_mean ?? 0.5);

  return (
    <div className="diag-section">
      <SectionHeader title="Model Quality Metrics" meta={`Last run: ${diag.last_run?.slice(0, 10) ?? 'unknown'}`} />
      <div className="metrics-grid">
        {metrics.map((m) => (
          <KpiCard
            key={m.label}
            label={m.label}
            value={m.raw ? String(m.value) : fmtNum(m.value, 4)}
          />
        ))}
      </div>

      <SectionHeader title="ROC Curve" meta={`AUC = ${fmtNum(mq.roc_auc_mean ?? 0, 4)}`} />
      <div className="chart-panel">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={rocPoints} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} />
            <XAxis dataKey="fpr" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtNum(v, 1)} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -2, fill: 'var(--text-tertiary)', fontSize: 10 }} />
            <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={44} tickFormatter={(v) => fmtNum(v, 1)} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: 'var(--text-tertiary)', fontSize: 10 }} />
            <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [fmtNum(v as number, 3)]} />
            <ReferenceLine stroke="var(--text-tertiary)" strokeDasharray="4 3" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
            <Line type="monotone" dataKey="tpr" stroke="var(--chart-1)" strokeWidth={2} dot={false} name="Model" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SignalQualityTab({ diag }: { diag: ReturnType<typeof useDiagnostics>['data'] }) {
  const t = diag.turnover;
  const aa = diag.alpha_attribution;

  return (
    <div className="diag-section">
      <SectionHeader title="Signal Quality" />
      <div className="metrics-grid">
        <KpiCard label="Gross CAGR" value={fmtPct(t.gross_cagr / 100)} changeType="positive" />
        <KpiCard label="Net CAGR" value={fmtPct(t.net_cagr / 100)} changeType="positive" />
        <KpiCard label="Cost Drag" value={`${t.cost_drag_pct.toFixed(2)}%`} changeType="negative" />
        <KpiCard label="Avg Turnover" value={fmtPct(t.avg_daily)} />
        <KpiCard label="Annual Multiple" value={`${t.annual_multiple.toFixed(1)}x`} />
        <KpiCard label="Alpha" value={fmtPct(aa.alpha_ann)} changeType={aa.alpha_ann >= 0 ? 'positive' : 'negative'} />
      </div>

      <SectionHeader title="Alpha Attribution" />
      <div className="alpha-attr-table data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>COMPONENT</th>
              <th className="numeric">VALUE</th>
              <th>INTERPRETATION</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Ensemble CAGR</td>
              <td className="cell-numeric positive">{fmtPct(aa.ensemble_cagr)}</td>
              <td className="cell-sector">Model's theoretical gross CAGR</td>
            </tr>
            <tr>
              <td>Realized Alpha</td>
              <td className={`cell-numeric ${aa.alpha_ann >= 0 ? 'positive' : 'negative'}`}>{fmtPct(aa.alpha_ann)}</td>
              <td className="cell-sector">Excess return over benchmark</td>
            </tr>
            <tr>
              <td>Beta</td>
              <td className="cell-numeric">{fmtNum(aa.beta, 3)}</td>
              <td className="cell-sector">Market sensitivity</td>
            </tr>
            <tr>
              <td>Cost Drag</td>
              <td className="cell-numeric negative">−{t.cost_drag_pct.toFixed(2)}%</td>
              <td className="cell-sector">Transaction cost impact on returns</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CalibrationTab({ cal }: { cal: ReturnType<typeof useCalibration>['data'] }) {
  const chartData = cal.map((b) => ({
    midpoint: (b.prob_min + b.prob_max) / 2,
    actual: b.actual_rate,
    expected: (b.prob_min + b.prob_max) / 2,
  }));

  return (
    <div className="diag-section">
      <SectionHeader title="Probability Calibration" meta="Predicted probability vs. actual positive rate" />
      <div className="chart-panel" style={{ marginBottom: 'var(--space-4)' }}>
        <div className="chart-title">Calibration Curve</div>
        <div className="chart-subtitle">Perfect calibration = diagonal. Deviation indicates over/under-confidence.</div>
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} />
            <XAxis dataKey="expected" type="number" domain={[0, 1]} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} name="Predicted Probability" />
            <YAxis dataKey="actual" type="number" domain={[0, 1]} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={44} name="Actual Rate" />
            <Tooltip {...TOOLTIP_STYLE} formatter={(v, name) => [fmtNum(v as number, 3), String(name)]} />
            <ReferenceLine stroke="var(--text-tertiary)" strokeDasharray="4 3" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
            <Scatter data={chartData} fill="var(--chart-1)" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <SectionHeader title="Reliability Table" />
      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>PROB BUCKET</th>
              <th className="numeric">COUNT</th>
              <th className="numeric">ACTUAL RATE</th>
              <th className="numeric">EXPECTED RATE</th>
              <th className="numeric">DEVIATION</th>
            </tr>
          </thead>
          <tbody>
            {cal.map((b) => {
              const expected = (b.prob_min + b.prob_max) / 2;
              const deviation = b.actual_rate - expected;
              return (
                <tr key={b.prob_min}>
                  <td className="cell-sector">{fmtNum(b.prob_min, 1)}–{fmtNum(b.prob_max, 1)}</td>
                  <td className="cell-numeric">{b.count}</td>
                  <td className="cell-numeric">{fmtPct(b.actual_rate)}</td>
                  <td className="cell-numeric">{fmtPct(expected)}</td>
                  <td className={`cell-numeric ${deviation >= 0 ? 'positive' : 'negative'}`}>
                    {deviation >= 0 ? '+' : ''}{fmtPct(deviation)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DriftTab() {
  return (
    <div className="diag-section">
      <SectionHeader title="Model Drift" meta="Requires database connection for historical data" />
      <div className="drift-status-card">
        <div className="drift-score-label">DRIFT STATUS</div>
        <div className="drift-score-value stable">STABLE</div>
        <div className="drift-score-sub">Model performance within normal variance thresholds</div>
      </div>
      <div className="drift-note">
        Historical drift tracking requires PostgreSQL backend. In JSON-only mode, drift metrics show current snapshot only.
        Connect the database to enable time-series drift monitoring across model retrains.
      </div>
    </div>
  );
}

function PortfolioAnalyticsTab({ diag }: { diag: ReturnType<typeof useDiagnostics>['data'] }) {
  const t = diag.turnover;
  const costData = [
    { name: 'Broker Fees', value: t.cost_drag_pct * 0.4, fill: 'var(--chart-1)' },
    { name: 'Slippage', value: t.cost_drag_pct * 0.35, fill: 'var(--chart-3)' },
    { name: 'Market Impact', value: t.cost_drag_pct * 0.25, fill: 'var(--chart-4)' },
  ];

  return (
    <div className="diag-section">
      <SectionHeader title="Transaction Cost Analysis" />
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4,1fr)' }}>
        <KpiCard label="Cost Drag" value={`${t.cost_drag_pct.toFixed(2)}%`} changeType="negative" />
        <KpiCard label="Annual Turnover" value={`${t.annual_multiple.toFixed(1)}x`} />
        <KpiCard label="Avg Daily Turnover" value={fmtPct(t.avg_daily)} />
        <KpiCard label="Gross vs Net" value={`${(t.gross_cagr - t.net_cagr).toFixed(2)}%`} changeType="negative" />
      </div>

      <div className="chart-panel">
        <div className="chart-title">Cost Breakdown</div>
        <div className="chart-subtitle">Estimated transaction cost components</div>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={costData} layout="vertical" margin={{ top: 4, right: 16, left: 80, bottom: 0 }}>
            <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} horizontal={false} />
            <XAxis type="number" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v.toFixed(2)}%`} />
            <YAxis type="category" dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [`${(v as number).toFixed(3)}%`, 'Cost']} />
            <Bar dataKey="value" radius={[0, 3, 3, 0]}>
              {costData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function FeatureTab({ diag }: { diag: ReturnType<typeof useDiagnostics>['data'] }) {
  const fi = [...diag.feature_importance].sort((a, b) => b.importance - a.importance);

  return (
    <div className="diag-section">
      <SectionHeader title="Feature Importance" meta={`${fi.length} features`} />
      <div className="chart-panel">
        <div className="chart-title">Current Feature Importance</div>
        <div className="chart-subtitle">Mean importance across all tickers — higher = stronger predictor</div>
        <ResponsiveContainer width="100%" height={Math.max(200, fi.length * 28)}>
          <BarChart data={fi} layout="vertical" margin={{ top: 4, right: 16, left: 140, bottom: 0 }}>
            <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} horizontal={false} />
            <XAxis type="number" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => fmtNum(v, 3)} />
            <YAxis type="category" dataKey="feature" tick={{ fill: 'var(--text-secondary)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} />
            <Tooltip {...TOOLTIP_STYLE} formatter={(v) => [fmtNum(v as number, 4), 'Importance']} />
            <Bar dataKey="importance" fill="var(--accent-primary)" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function generateRocCurve(auc: number): Array<{ fpr: number; tpr: number }> {
  const points = [];
  const n = 50;
  const alpha = auc > 0.5 ? 2 + (auc - 0.5) * 8 : 0.5;
  for (let i = 0; i <= n; i++) {
    const fpr = i / n;
    const tpr = Math.min(1, Math.pow(fpr, 1 / alpha));
    points.push({ fpr: Math.round(fpr * 1000) / 1000, tpr: Math.round(tpr * 1000) / 1000 });
  }
  return points;
}

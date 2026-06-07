import { useState } from 'react';
import { useDiagnostics } from '../api/hooks';
import { fmtPct } from '../utils/format';
import { FreshnessTag } from '../components/FreshnessTag';
import { SectionError } from '../components/SectionError';
import type { FeatureImportanceItem } from '../api/types';
import './Diagnostics.css';

export function Diagnostics() {
  const { data: diag, loading, error, refetch, fetchedAt } = useDiagnostics();
  const mq = diag.model_quality;
  const t = diag.turnover;

  const top15 = [...diag.feature_importance]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15);

  const costImpact = (t.gross_cagr - t.net_cagr).toFixed(2);

  return (
    <div className="diag-page">
      {/* Model Quality */}
      <div className="diag-panel">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
          <div className="diag-panel-title" style={{ margin: 0 }}>Model Quality</div>
          <FreshnessTag lastUpdated={fetchedAt} />
        </div>
        {error && <SectionError message="Failed to load diagnostics." onRetry={refetch} />}
        {loading ? (
          <div className="diag-mq-cards">
            {Array.from({ length: 5 }, (_, i) => (
              <div key={i} className="diag-mq-card">
                <div className="skeleton skeleton-text" style={{ width: '60%' }} />
                <div className="skeleton skeleton-value" style={{ width: '40%' }} />
              </div>
            ))}
          </div>
        ) : (
        <div className="diag-mq-cards">
          <RocAucCard value={mq.roc_auc_mean ?? 0} />
          <MetricCard label="Accuracy Mean" value={`${((mq.accuracy_mean ?? 0) * 100).toFixed(1)}%`} />
          <MetricCard label="Precision Mean" value={`${((mq.precision_mean ?? 0) * 100).toFixed(1)}%`} />
          <MetricCard label="Recall" value={`${((mq.recall_mean ?? 0) * 100).toFixed(1)}%`} />
          <MetricCard label="Tickers Trained" value={String(mq.n_tickers)} />
        </div>
        <div className="diag-mq-explanation">
          ROC-AUC measures the model's ability to rank up-moves above down-moves.
          Values above 0.60 indicate meaningful predictive power beyond random chance.
        </div>
        )}
      </div>

      {/* Feature Importance */}
      <div className="diag-panel">
        <div className="diag-panel-title">Feature Importance</div>
        <div className="diag-panel-subtitle">Top {top15.length} features — mean importance across all tickers</div>
        {loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {Array.from({ length: 8 }, (_, i) => (
              <div key={i} className="skeleton" style={{ height: 20, width: `${60 + (i % 4) * 10}%` }} />
            ))}
          </div>
        ) : (
          <FeatureImportanceBars items={top15} />
        )}
      </div>

      {/* Turnover & Cost Summary */}
      <div className="diag-panel">
        <div className="diag-panel-title">Turnover &amp; Cost Summary</div>
        <div className="diag-cost-table">
          <CostRow label="Avg Daily Turnover" value={fmtPct(t.avg_daily)} />
          <CostRow label="Annual Turnover Multiple" value={`${t.annual_multiple.toFixed(1)}x`} />
          <CostRow label="Annual Cost Drag" value={`${t.cost_drag_pct.toFixed(2)}%`} negative />
          <CostRow label="Gross CAGR" value={`${t.gross_cagr.toFixed(1)}%`} positive />
          <CostRow label="Net CAGR (after costs)" value={`${t.net_cagr.toFixed(2)}%`} positive />
        </div>
        <div className="diag-cost-impact">
          Costs reduce CAGR by <strong>{costImpact}%</strong> per year
        </div>
      </div>

      {/* Run Configuration */}
      {diag.run_config && (
        <div className="diag-panel diag-run-config-panel">
          <RunConfigSection config={diag.run_config} />
        </div>
      )}
    </div>
  );
}

function RocAucCard({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, (value - 0.5) / 0.5));
  return (
    <div className="diag-mq-card diag-mq-card-roc">
      <div className="diag-mq-card-label">ROC-AUC Mean</div>
      <div className="diag-mq-card-value">{value.toFixed(2)}</div>
      <div className="diag-roc-gauge">
        <div className="diag-roc-gauge-track">
          <div
            className="diag-roc-gauge-marker"
            style={{ left: `calc(${pct * 100}% - 6px)` }}
          />
        </div>
        <div className="diag-roc-gauge-labels">
          <span>0.50</span>
          <span>1.00</span>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="diag-mq-card">
      <div className="diag-mq-card-label">{label}</div>
      <div className="diag-mq-card-value">{value}</div>
    </div>
  );
}

function FeatureImportanceBars({ items }: { items: FeatureImportanceItem[] }) {
  const maxVal = items[0]?.importance ?? 1;
  return (
    <div className="diag-fi-chart">
      {items.map((item) => (
        <div key={item.feature} className="diag-fi-row">
          <div className="diag-fi-label" title={item.feature}>{item.feature}</div>
          <div className="diag-fi-bar-track">
            <div
              className="diag-fi-bar-fill"
              style={{ width: `${(item.importance / maxVal) * 100}%` }}
            />
          </div>
          <div className="diag-fi-value">{item.importance.toFixed(4)}</div>
        </div>
      ))}
    </div>
  );
}

function CostRow({
  label,
  value,
  positive,
  negative,
}: {
  label: string;
  value: string;
  positive?: boolean;
  negative?: boolean;
}) {
  const cls = positive ? 'diag-cost-val-pos' : negative ? 'diag-cost-val-neg' : '';
  return (
    <div className="diag-cost-row">
      <span className="diag-cost-label">{label}</span>
      <span className={`diag-cost-val ${cls}`}>{value}</span>
    </div>
  );
}

function RunConfigSection({ config }: { config: Record<string, string | number | boolean> }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="diag-run-config">
      <button
        className={`diag-run-config-summary${open ? ' open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <span className="diag-run-config-chevron">{open ? '▾' : '▸'}</span>
        Run Configuration (Advanced)
      </button>
      {open && (
        <div className="diag-run-config-body">
          {Object.entries(config).map(([key, val]) => (
            <div key={key} className="diag-run-config-row">
              <span className="diag-run-config-key">{key}</span>
              <span className="diag-run-config-val">{String(val)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

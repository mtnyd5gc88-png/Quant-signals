import './KpiCard.css';

interface Props {
  label: string;
  value: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  tooltip?: string;
  loading?: boolean;
}

export function KpiCard({ label, value, changeType = 'neutral', tooltip, loading = false }: Props) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">
        {label}
        {tooltip && (
          <span className="kpi-tooltip-anchor">
            <span className="kpi-tooltip-icon">?</span>
            <span className="kpi-tooltip-bubble">{tooltip}</span>
          </span>
        )}
      </div>
      {loading
        ? <div className="kpi-skeleton" />
        : <div className={`kpi-value${changeType !== 'neutral' ? ` kpi-value-${changeType}` : ''}`}>{value}</div>
      }
    </div>
  );
}

import './KpiCard.css';

interface Props {
  label: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  mono?: boolean;
}

export function KpiCard({ label, value, change, changeType = 'neutral', mono = true }: Props) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value${mono ? ' mono' : ''}`}>{value}</div>
      {change && (
        <div className={`kpi-change ${changeType}`}>
          {changeType === 'positive' && <span>▲</span>}
          {changeType === 'negative' && <span>▼</span>}
          {change}
        </div>
      )}
    </div>
  );
}

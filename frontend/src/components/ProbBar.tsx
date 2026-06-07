import './ProbBar.css';

interface Props {
  value: number; // 0–1
  rocAuc?: number;
  avgTrust?: boolean;
}

function barColor(value: number): string {
  if (value < 0.40) return '#ef4444';
  if (value < 0.55) return '#eab308';
  if (value < 0.70) return '#3b82f6';
  return '#16a34a';
}

function rocToDots(roc: number): number {
  if (roc < 0.55) return 1;
  if (roc < 0.60) return 2;
  if (roc < 0.65) return 3;
  if (roc < 0.70) return 4;
  return 5;
}

export function ProbBar({ value, rocAuc, avgTrust = false }: Props) {
  const pct = Math.round(value * 100);
  const filled = rocAuc !== undefined ? rocToDots(rocAuc) : 0;
  return (
    <div className="prob-cell">
      <div className="prob-main-row">
        <span className="prob-value">{pct}%</span>
        <div className="prob-bar-track">
          <div className="prob-bar-fill" style={{ width: `${pct}%`, background: barColor(value) }} />
        </div>
      </div>
      {rocAuc !== undefined && (
        <div className="prob-trust-row">
          <span className="prob-trust-label">Trust</span>
          {Array.from({ length: 5 }, (_, i) => (
            <span key={i} className={`prob-trust-dot${i < filled ? ' filled' : ''}`} />
          ))}
          {avgTrust && <span className="prob-trust-avg">(avg)</span>}
        </div>
      )}
    </div>
  );
}

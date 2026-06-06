import './ProbBar.css';

interface Props {
  value: number; // 0–1
}

export function ProbBar({ value }: Props) {
  const pct = Math.round(value * 100);
  const color = value < 0.4 ? 'var(--negative)' : value < 0.6 ? 'var(--neutral)' : 'var(--positive)';
  return (
    <div className="prob-cell">
      <div className="prob-bar-track">
        <div className="prob-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="prob-value">{pct.toFixed(1)}%</span>
    </div>
  );
}

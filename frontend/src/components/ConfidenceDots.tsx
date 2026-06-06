import './ConfidenceDots.css';

interface Props {
  value: number; // 0–1
}

const LABELS = ['', 'Very Low', 'Low', 'Moderate', 'High', 'Very High'];

export function ConfidenceDots({ value }: Props) {
  const filled = Math.round(value * 5);
  const label = LABELS[filled] ?? '';
  return (
    <div className="confidence-indicator">
      <div className="confidence-dots">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i} className={`confidence-dot${i < filled ? ' filled' : ''}`} />
        ))}
      </div>
      <span className="confidence-label">{label}</span>
    </div>
  );
}

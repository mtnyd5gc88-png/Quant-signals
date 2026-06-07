import './SignalBadge.css';

interface Props {
  signal: string;
}

export function SignalBadge({ signal }: Props) {
  const norm = signal === 'STAY IN CASH' ? 'CASH' : signal;
  const cls = norm === 'BUY' ? 'buy' : norm === 'SELL' ? 'sell' : norm === 'CASH' ? 'cash' : 'hold';
  return <span className={`signal-badge ${cls}`}>{norm}</span>;
}

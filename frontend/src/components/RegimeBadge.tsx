import './RegimeBadge.css';

interface Props {
  regime: string;
}

export function RegimeBadge({ regime }: Props) {
  const cls =
    regime === 'BULL' ? 'bull' :
    regime === 'RISK-OFF' ? 'risk-off' :
    regime === 'DEFENSIVE' ? 'defensive' : 'neutral';
  return <span className={`regime-badge ${cls}`}>{regime}</span>;
}

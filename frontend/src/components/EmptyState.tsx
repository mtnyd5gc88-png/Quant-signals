import type { ReactNode } from 'react';
import './EmptyState.css';

interface Props {
  icon?: ReactNode;
  message: string;
  sub?: string;
}

export function EmptyState({ icon, message, sub }: Props) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-state__icon">{icon}</div>}
      <p className="empty-state__message">{message}</p>
      {sub && <p className="empty-state__sub">{sub}</p>}
    </div>
  );
}

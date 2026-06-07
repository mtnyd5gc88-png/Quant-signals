import { Search, Bell } from 'lucide-react';
import { RegimeBadge } from '../components/RegimeBadge';
import { useRegime } from '../api/hooks';
import './TopBar.css';

interface Props {
  onSearch?: (q: string) => void;
}

function formatTimestamp(iso: string): string {
  try {
    return `UPDATED ${new Date(iso).toLocaleTimeString('en-US', { hour12: false, timeZone: 'UTC' })} UTC`;
  } catch {
    return 'UPDATED --:--:-- UTC';
  }
}

export function TopBar({ onSearch }: Props) {
  const { data: regime } = useRegime();

  return (
    <header className="topbar">
      <div className="topbar-search">
        <Search size={13} color="var(--text-tertiary)" strokeWidth={1.5} />
        <input
          type="text"
          placeholder="Search ticker, company... (Press /)"
          onChange={(e) => onSearch?.(e.target.value)}
        />
        <span className="topbar-search-kbd">/</span>
      </div>

      <div className="topbar-right">
        <RegimeBadge regime={regime.regime.toUpperCase()} />
        <span className="topbar-timestamp">{formatTimestamp(regime.last_updated)}</span>
        <button className="topbar-bell" aria-label="Notifications">
          <Bell size={15} strokeWidth={1.5} color="var(--text-secondary)" />
        </button>
        <div className="topbar-avatar">IJ</div>
      </div>
    </header>
  );
}

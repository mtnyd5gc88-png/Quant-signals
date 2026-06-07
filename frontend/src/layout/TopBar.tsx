import { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { RegimeBadge } from '../components/RegimeBadge';
import { useRegime } from '../api/hooks';
import { useAuth } from '../auth/AuthContext';
import './TopBar.css';

function formatTimestamp(iso: string): string {
  try {
    return `UPDATED ${new Date(iso).toLocaleTimeString('en-US', { hour12: false, timeZone: 'UTC' })} UTC`;
  } catch {
    return 'UPDATED --:--:-- UTC';
  }
}

function ProfileDropdown() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const keyHandler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    document.addEventListener('keydown', keyHandler);
    return () => {
      document.removeEventListener('mousedown', handler);
      document.removeEventListener('keydown', keyHandler);
    };
  }, []);

  return (
    <div ref={ref} className="avatar-wrap">
      <button
        className="topbar-avatar"
        onClick={() => setOpen((o) => !o)}
        aria-label="Profile menu"
        aria-expanded={open}
      >
        {user?.initials ?? 'U'}
      </button>

      {open && (
        <div className="avatar-dropdown dropdown-menu">
          <div className="avatar-dropdown-header">
            <div className="avatar-dropdown-name">{user?.name ?? 'User'}</div>
            <div className="avatar-dropdown-email">{user?.email ?? ''}</div>
          </div>
          <div className="avatar-dropdown-body">
            <button
              className="avatar-dropdown-item"
              onClick={() => { navigate('/settings'); setOpen(false); }}
            >
              Settings
            </button>
            <button
              className="avatar-dropdown-item danger"
              onClick={() => { logout(); setOpen(false); }}
            >
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

interface Props {
  onSearch?: (q: string) => void;
}

export function TopBar({ onSearch }: Props) {
  const { data: regime } = useRegime();

  return (
    <header className="topbar">
      {/* Logo */}
      <div className="topbar-logo">
        <div className="topbar-logo-mark">QS</div>
        <span className="topbar-logo-name">QUANT-SIGNALS</span>
      </div>

      {/* Search */}
      <div className="topbar-search">
        <Search size={13} color="var(--text-tertiary)" strokeWidth={1.5} />
        <input
          type="text"
          placeholder="Search ticker, company…"
          onChange={(e) => onSearch?.(e.target.value)}
        />
        <span className="topbar-search-kbd">/</span>
      </div>

      {/* Right */}
      <div className="topbar-right">
        <RegimeBadge regime={regime.regime.toUpperCase()} />
        <span className="topbar-timestamp">{formatTimestamp(regime.last_updated)}</span>
        <ProfileDropdown />
      </div>
    </header>
  );
}

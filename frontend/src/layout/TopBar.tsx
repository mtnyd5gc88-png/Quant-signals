import { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, RotateCw, CheckCircle, AlertCircle, Menu } from 'lucide-react';
import type { BackendStatus } from '../api/hooks';
import { RegimeBadge } from '../components/RegimeBadge';
import { useRegime } from '../api/hooks';
import { useAuth } from '../auth/AuthContext';
import { api } from '../api/client';
import { fmtRelTime } from '../utils/format';
import type { SignalItem } from '../api/types';
import './TopBar.css';

function formatLastUpdated(iso: string): string {
  try {
    return `Updated ${fmtRelTime(iso)}`;
  } catch {
    return 'Updated —';
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

function SearchBar() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<SignalItem | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [searching, setSearching] = useState(false);
  const navigate = useNavigate();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setQuery('');
        setResult(null);
        setNotFound(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleChange = (value: string) => {
    setQuery(value);
    setResult(null);
    setNotFound(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!value.trim()) { setSearching(false); return; }
    setSearching(true);
    timerRef.current = setTimeout(async () => {
      try {
        const data = await api.searchTicker(value.trim());
        setResult(data);
        setNotFound(false);
      } catch {
        setResult(null);
        setNotFound(true);
      } finally {
        setSearching(false);
      }
    }, 300);
  };

  const handleSelect = (ticker: string) => {
    navigate(`/signals?q=${encodeURIComponent(ticker)}`);
    setQuery('');
    setResult(null);
    setNotFound(false);
  };

  const showDropdown = query.trim().length > 0 && (searching || result !== null || notFound);

  return (
    <div ref={wrapRef} className="topbar-search-wrap">
      <div className="topbar-search">
        <Search size={13} color="var(--text-tertiary)" strokeWidth={1.5} />
        <input
          type="text"
          placeholder="Search ticker, company…"
          value={query}
          onChange={(e) => handleChange(e.target.value)}
        />
        <span className="topbar-search-kbd">/</span>
      </div>
      {showDropdown && (
        <div className="search-dropdown">
          {searching && (
            <div className="search-dropdown-item search-loading">Searching…</div>
          )}
          {!searching && result && (
            <button
              className="search-dropdown-item search-result"
              onClick={() => handleSelect(result.ticker)}
            >
              <span className="search-ticker">{result.ticker}</span>
              {result.company && <span className="search-company">{result.company}</span>}
              <span className={`search-signal signal-${result.signal.toLowerCase().replace(/ /g, '-')}`}>
                {result.signal}
              </span>
            </button>
          )}
          {!searching && notFound && (
            <div className="search-dropdown-item search-not-found">
              No signal data for <strong>{query.trim().toUpperCase()}</strong>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RefreshButton() {
  const [state, setState] = useState<'idle' | 'loading' | 'ok' | 'error'>('idle');
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleRefresh = async () => {
    if (state === 'loading') return;
    setState('loading');
    if (timerRef.current) clearTimeout(timerRef.current);
    try {
      await api.refresh();
      setState('ok');
    } catch {
      setState('error');
    }
    timerRef.current = setTimeout(() => setState('idle'), 2500);
  };

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  return (
    <button
      className="topbar-refresh"
      onClick={handleRefresh}
      disabled={state === 'loading'}
      title="Refresh data"
      aria-label="Refresh data"
    >
      {state === 'ok'
        ? <CheckCircle size={14} color="var(--positive)" />
        : state === 'error'
          ? <AlertCircle size={14} color="var(--negative)" />
          : <RotateCw size={14} strokeWidth={1.5} className={state === 'loading' ? 'topbar-refresh-spin' : ''} />}
    </button>
  );
}

const STATUS_CONFIG: Record<BackendStatus, { color: string; label: string }> = {
  connecting: { color: '#9ca3af', label: 'Connecting to server...' },
  online:     { color: '#16a34a', label: 'Live' },
  slow:       { color: '#d97706', label: 'Server starting up... (free tier)' },
  offline:    { color: '#dc2626', label: 'Server offline' },
};

function BackendStatusBadge({ status }: { status: BackendStatus }) {
  const { color, label } = STATUS_CONFIG[status];
  return (
    <div className="backend-status">
      <span className="backend-status-dot" style={{ background: color }} />
      <span className="backend-status-label" style={{ color: status === 'offline' ? '#dc2626' : undefined }}>
        {label}
      </span>
    </div>
  );
}

export function TopBar({
  onMenuClick,
  backendStatus = 'online',
}: {
  onMenuClick?: () => void;
  backendStatus?: BackendStatus;
}) {
  const { data: regime } = useRegime();

  return (
    <header className="topbar">
      {onMenuClick && (
        <button
          className="topbar-hamburger"
          onClick={onMenuClick}
          aria-label="Toggle navigation"
          type="button"
        >
          <Menu size={18} strokeWidth={1.5} />
        </button>
      )}
      <div className="topbar-logo">
        <div className="topbar-logo-mark">QS</div>
        <span className="topbar-logo-name">QUANT-SIGNALS</span>
      </div>

      <SearchBar />

      <div className="topbar-right">
        <BackendStatusBadge status={backendStatus} />
        <RegimeBadge regime={regime.regime.toUpperCase()} />
        <span className="topbar-timestamp">{formatLastUpdated(regime.last_updated)}</span>
        <RefreshButton />
        <ProfileDropdown />
      </div>
    </header>
  );
}

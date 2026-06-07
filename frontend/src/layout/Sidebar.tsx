import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Zap,
  PieChart,
  Activity,
  TrendingUp,
  FlaskConical,
  Settings,
  ChevronDown,
} from 'lucide-react';
import './Sidebar.css';

const NAV_GROUPS = [
  {
    section: 'NAVIGATION',
    items: [
      { to: '/',            label: 'Dashboard',   icon: LayoutDashboard },
      { to: '/signals',     label: 'Signals',     icon: Zap },
      { to: '/portfolio',   label: 'Portfolio',   icon: PieChart },
      { to: '/diagnostics', label: 'Diagnostics', icon: Activity },
      { to: '/backtests',   label: 'Backtests',   icon: TrendingUp },
      { to: '/research',    label: 'Research',    icon: FlaskConical },
    ],
  },
  {
    section: 'SYSTEM',
    items: [
      { to: '/settings', label: 'Settings', icon: Settings },
    ],
  },
];

export function Sidebar() {
  return (
    <nav className="sidebar">
      {/* Brand / Logo */}
      <div className="sidebar-brand">
        <div className="sidebar-logo-mark">QS</div>
        <div className="sidebar-brand-text">
          <div className="sidebar-brand-name">Quant Signals</div>
          <div className="sidebar-brand-sub">Research Platform</div>
        </div>
      </div>

      <div className="sidebar-divider" />

      {/* Nav groups */}
      {NAV_GROUPS.map(({ section, items }) => (
        <div key={section} className="nav-group">
          <div className="nav-section-label">
            <span>{section}</span>
            <ChevronDown size={10} strokeWidth={1.5} />
          </div>
          {items.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <Icon size={15} strokeWidth={1.5} />
              <span>{label}</span>
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}

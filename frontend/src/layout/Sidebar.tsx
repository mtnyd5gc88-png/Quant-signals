import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Zap,
  PieChart,
  Activity,
  TrendingUp,
  FlaskConical,
  Settings,
} from 'lucide-react';
import './Sidebar.css';

const NAV = [
  { to: '/',            label: 'Dashboard',   icon: LayoutDashboard, section: 'NAVIGATION' },
  { to: '/signals',     label: 'Signals',     icon: Zap },
  { to: '/portfolio',   label: 'Portfolio',   icon: PieChart },
  { to: '/diagnostics', label: 'Diagnostics', icon: Activity },
  { to: '/backtests',   label: 'Backtests',   icon: TrendingUp },
  { to: '/research',    label: 'Research',    icon: FlaskConical },
  { to: '/settings',    label: 'Settings',    icon: Settings, section: 'SYSTEM' },
];

export function Sidebar() {
  let lastSection = '';
  return (
    <nav className="sidebar">
      {NAV.map(({ to, label, icon: Icon, section }) => {
        const showSection = section && section !== lastSection;
        if (showSection) lastSection = section!;
        return (
          <div key={to}>
            {showSection && <div className="nav-section-label">{section}</div>}
            <NavLink
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <Icon size={16} strokeWidth={1.5} />
              {label}
            </NavLink>
          </div>
        );
      })}
    </nav>
  );
}

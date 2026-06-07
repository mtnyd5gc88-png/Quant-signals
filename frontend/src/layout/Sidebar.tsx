import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Zap,
  PieChart,
  Activity,
  TrendingUp,
  BarChart2,
  FlaskConical,
  Settings,
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
      { to: '/backtests',    label: 'Backtests',    icon: TrendingUp },
      { to: '/performance', label: 'Performance',  icon: BarChart2 },
      { to: '/research',    label: 'Research',     icon: FlaskConical },
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
      {NAV_GROUPS.map(({ section, items }) => (
        <div key={section} className="nav-group">
          <div className="nav-section-label">{section}</div>
          {items.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <Icon size={16} strokeWidth={1.5} />
              <span>{label}</span>
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}

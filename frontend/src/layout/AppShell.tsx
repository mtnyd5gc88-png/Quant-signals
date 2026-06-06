import { Outlet } from 'react-router-dom';
import { TopBar } from './TopBar';
import { Sidebar } from './Sidebar';
import './AppShell.css';

export function AppShell() {
  return (
    <div className="app-shell">
      <TopBar />
      <Sidebar />
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}

import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { TopBar } from './TopBar';
import { Sidebar } from './Sidebar';
import { useBackendStatus } from '../api/hooks';
import './AppShell.css';

export function AppShell() {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const backendStatus = useBackendStatus();

  return (
    <div className="app-shell">
      {(backendStatus === 'connecting' || backendStatus === 'slow') && (
        <div className="cold-start-bar" aria-hidden="true">
          <div className="cold-start-bar-fill" />
        </div>
      )}
      <TopBar
        onMenuClick={() => setMobileSidebarOpen((o) => !o)}
        backendStatus={backendStatus}
      />
      <Sidebar
        mobileOpen={mobileSidebarOpen}
        onClose={() => setMobileSidebarOpen(false)}
      />
      {mobileSidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}

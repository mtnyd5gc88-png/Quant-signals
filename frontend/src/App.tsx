import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './auth/AuthContext';
import { InvestorProfileProvider } from './context/InvestorProfile';
import { AppShell } from './layout/AppShell';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { Signals } from './pages/Signals';
import { Portfolio } from './pages/Portfolio';
import { Diagnostics } from './pages/Diagnostics';
import { Backtests } from './pages/Backtests';
import { Performance } from './pages/Performance';
import { Research } from './pages/Research';
import { Settings } from './pages/Settings';
import type { ReactNode } from 'react';

function ProtectedShell({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <InvestorProfileProvider>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            element={
              <ProtectedShell>
                <AppShell />
              </ProtectedShell>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="signals"     element={<Signals />} />
            <Route path="portfolio"   element={<Portfolio />} />
            <Route path="diagnostics" element={<Diagnostics />} />
            <Route path="backtests"    element={<Backtests />} />
            <Route path="performance"  element={<Performance />} />
            <Route path="research"    element={<Research />} />
            <Route path="settings"    element={<Settings />} />
            <Route path="*"           element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </AuthProvider>
      </InvestorProfileProvider>
    </BrowserRouter>
  );
}

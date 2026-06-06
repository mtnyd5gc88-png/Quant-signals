import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from './layout/AppShell';
import { Dashboard } from './pages/Dashboard';
import { Signals } from './pages/Signals';
import { Portfolio } from './pages/Portfolio';
import { Diagnostics } from './pages/Diagnostics';
import { Backtests } from './pages/Backtests';
import { Research } from './pages/Research';
import { Settings } from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Dashboard />} />
          <Route path="signals"     element={<Signals />} />
          <Route path="portfolio"   element={<Portfolio />} />
          <Route path="diagnostics" element={<Diagnostics />} />
          <Route path="backtests"   element={<Backtests />} />
          <Route path="research"    element={<Research />} />
          <Route path="settings"    element={<Settings />} />
          <Route path="*"           element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

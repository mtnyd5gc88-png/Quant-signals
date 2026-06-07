import { useState, type ReactNode } from 'react';
import { api } from '../api/client';
import './Settings.css';

function Toast({ message, type, onClose }: { message: string; type: 'success' | 'error' | 'info'; onClose: () => void }) {
  const cls = type === 'success' ? 'toast-success' : type === 'error' ? 'toast-error' : 'toast-info';
  return (
    <div className={`toast ${cls}`}>
      <span>{message}</span>
      <button className="toast-close" onClick={onClose}>✕</button>
    </div>
  );
}

function SettingRow({ label, description, control }: { label: string; description: string; control: ReactNode }) {
  return (
    <div className="setting-row">
      <div className="setting-info">
        <div className="setting-label">{label}</div>
        <div className="setting-desc">{description}</div>
      </div>
      <div className="setting-control">{control}</div>
    </div>
  );
}

function DataRefreshSection() {
  const [interval, setRefreshInterval] = useState(
    () => localStorage.getItem('qs_refresh_interval') || 'manual',
  );
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  const handleIntervalChange = (val: string) => {
    setRefreshInterval(val);
    localStorage.setItem('qs_refresh_interval', val);
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    setToast(null);
    try {
      await api.refresh();
      setLastRefresh(new Date().toISOString());
      setToast({ message: 'Data refreshed successfully', type: 'success' });
    } catch (e) {
      setToast({ message: `Refresh failed: ${e instanceof Error ? e.message : 'Unknown error'}`, type: 'error' });
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="settings-card">
      <SettingRow
        label="Refresh Interval"
        description="How often to poll the backend for new signals"
        control={
          <select
            className="settings-select"
            value={interval}
            onChange={(e) => handleIntervalChange(e.target.value)}
          >
            <option value="manual">Manual Only</option>
            <option value="5m">Every 5 minutes</option>
            <option value="15m">Every 15 minutes</option>
            <option value="1h">Every hour</option>
          </select>
        }
      />
      <div className="setting-divider" />
      <SettingRow
        label="Manual Refresh"
        description="Trigger an immediate data reload from backend"
        control={
          <button
            className={`settings-btn primary${refreshing ? ' loading' : ''}`}
            onClick={handleRefresh}
            disabled={refreshing}
          >
            {refreshing ? 'Refreshing…' : 'Refresh Now'}
          </button>
        }
      />
      {lastRefresh && (
        <div className="setting-meta">
          Last refreshed: {new Date(lastRefresh).toLocaleTimeString()}
        </div>
      )}
      {toast && (
        <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
      )}
    </div>
  );
}

function ApiConfigSection() {
  const [baseUrl, setBaseUrl] = useState(
    () => localStorage.getItem('qs_api_url') || 'http://localhost:8000',
  );
  const [connStatus, setConnStatus] = useState<'idle' | 'checking' | 'ok' | 'error'>('idle');

  const saveUrl = () => {
    localStorage.setItem('qs_api_url', baseUrl);
  };

  const testConnection = async () => {
    setConnStatus('checking');
    try {
      const res = await api.health(baseUrl);
      setConnStatus(res.ok ? 'ok' : 'error');
    } catch {
      setConnStatus('error');
    }
  };

  return (
    <div className="settings-card">
      <SettingRow
        label="Backend URL"
        description="FastAPI backend base URL"
        control={
          <div className="settings-url-row">
            <input
              className="settings-input"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              onBlur={saveUrl}
              placeholder="http://localhost:8000"
            />
            {connStatus === 'ok'       && <span className="conn-status ok">● Connected</span>}
            {connStatus === 'error'    && <span className="conn-status err">● Unreachable</span>}
            {connStatus === 'checking' && <span className="conn-status checking">● Checking…</span>}
          </div>
        }
      />
      <div className="setting-divider" />
      <SettingRow
        label="Test Connection"
        description="Ping the backend health endpoint"
        control={
          <button
            className="settings-btn secondary"
            onClick={testConnection}
            disabled={connStatus === 'checking'}
          >
            {connStatus === 'checking' ? 'Testing…' : 'Test Connection'}
          </button>
        }
      />
    </div>
  );
}

export function Settings() {
  const [rowsPerPage, setRowsPerPage] = useState('50');
  const [timezone, setTimezone] = useState('UTC');

  return (
    <div className="page-content">
      <div className="settings-sections">

        <div className="settings-section">
          <div className="settings-section-title">Data Refresh</div>
          <DataRefreshSection />
        </div>

        <div className="settings-section">
          <div className="settings-section-title">API Configuration</div>
          <ApiConfigSection />
        </div>

        <div className="settings-section">
          <div className="settings-section-title">Display Preferences</div>
          <div className="settings-card">
            <SettingRow
              label="Default Page"
              description="Landing page on application load"
              control={
                <select className="settings-select">
                  <option>Dashboard</option>
                  <option>Signals</option>
                  <option>Diagnostics</option>
                </select>
              }
            />
            <div className="setting-divider" />
            <SettingRow
              label="Rows Per Table"
              description="Default row count for all data tables"
              control={
                <select
                  className="settings-select"
                  value={rowsPerPage}
                  onChange={(e) => setRowsPerPage(e.target.value)}
                >
                  <option>25</option>
                  <option>50</option>
                  <option>100</option>
                </select>
              }
            />
            <div className="setting-divider" />
            <SettingRow
              label="Timezone"
              description="Display timezone for all timestamps"
              control={
                <select
                  className="settings-select"
                  value={timezone}
                  onChange={(e) => setTimezone(e.target.value)}
                >
                  <option>UTC</option>
                  <option>US/Eastern</option>
                  <option>US/Central</option>
                  <option>US/Pacific</option>
                </select>
              }
            />
          </div>
        </div>

        <div className="settings-section">
          <div className="settings-section-title">Version Information</div>
          <div className="settings-card">
            {[
              ['Platform Version', '2.0.0'],
              ['Model Version', 'RF + LR Ensemble v2'],
              ['Data Schema', '1.0.0'],
              ['Backend', 'FastAPI 0.115 + JSON-only mode'],
            ].map(([k, v]) => (
              <div key={k}>
                <SettingRow label={k} description="" control={
                  <span className="settings-version-value">{v}</span>
                } />
                <div className="setting-divider" />
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}

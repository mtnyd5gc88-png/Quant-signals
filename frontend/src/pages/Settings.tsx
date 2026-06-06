import { useState } from 'react';
import { SectionHeader } from '../components/SectionHeader';
import './Settings.css';

export function Settings() {
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [rowsPerPage, setRowsPerPage] = useState('50');
  const [timezone, setTimezone] = useState('UTC');

  return (
    <div className="page-content">
      <div className="settings-sections">

        <div className="settings-section">
          <SectionHeader title="Data Refresh" />
          <div className="settings-card">
            <div className="settings-row">
              <div className="settings-label">
                <span className="settings-field-label">Refresh Interval</span>
                <span className="settings-field-desc">How often to poll the backend for new signals</span>
              </div>
              <select className="settings-select">
                <option>Manual Only</option>
                <option>5 minutes</option>
                <option>15 minutes</option>
                <option>1 hour</option>
                <option>Daily (06:00 ET)</option>
              </select>
            </div>
            <div className="settings-divider" />
            <div className="settings-row">
              <div className="settings-label">
                <span className="settings-field-label">Manual Refresh</span>
                <span className="settings-field-desc">Trigger an immediate data reload from backend</span>
              </div>
              <button className="settings-btn">Refresh Now</button>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <SectionHeader title="API Configuration" />
          <div className="settings-card">
            <div className="settings-row">
              <div className="settings-label">
                <span className="settings-field-label">Backend URL</span>
                <span className="settings-field-desc">FastAPI endpoint base URL</span>
              </div>
              <input
                className="settings-input"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="http://localhost:8000"
              />
            </div>
            <div className="settings-divider" />
            <div className="settings-row">
              <div className="settings-label">
                <span className="settings-field-label">Connection Status</span>
                <span className="settings-field-desc">Last health check result</span>
              </div>
              <div className="settings-status-ok">
                <span className="status-dot" />
                <span>Connected (JSON-only mode)</span>
              </div>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <SectionHeader title="Display Preferences" />
          <div className="settings-card">
            <div className="settings-row">
              <div className="settings-label">
                <span className="settings-field-label">Default Page</span>
                <span className="settings-field-desc">Landing page on application load</span>
              </div>
              <select className="settings-select">
                <option>Dashboard</option>
                <option>Signals</option>
                <option>Diagnostics</option>
              </select>
            </div>
            <div className="settings-divider" />
            <div className="settings-row">
              <div className="settings-label">
                <span className="settings-field-label">Rows Per Table</span>
                <span className="settings-field-desc">Default row count for all tables</span>
              </div>
              <select className="settings-select" value={rowsPerPage} onChange={(e) => setRowsPerPage(e.target.value)}>
                <option>25</option>
                <option>50</option>
                <option>100</option>
              </select>
            </div>
            <div className="settings-divider" />
            <div className="settings-row">
              <div className="settings-label">
                <span className="settings-field-label">Timezone</span>
                <span className="settings-field-desc">Display timezone for all timestamps</span>
              </div>
              <select className="settings-select" value={timezone} onChange={(e) => setTimezone(e.target.value)}>
                <option>UTC</option>
                <option>US/Eastern</option>
                <option>US/Central</option>
                <option>US/Pacific</option>
              </select>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <SectionHeader title="Version Information" />
          <div className="settings-card">
            {[
              ['Platform Version', '1.0.0'],
              ['Model Version', 'RandomForest v2 + LR Ensemble'],
              ['Data Schema', 'v1.2.0'],
              ['Backend', 'FastAPI 0.110 + SQLAlchemy 2.0'],
            ].map(([k, v]) => (
              <div key={k}>
                <div className="settings-row">
                  <span className="settings-field-label">{k}</span>
                  <span className="settings-version-value">{v}</span>
                </div>
                <div className="settings-divider" />
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}

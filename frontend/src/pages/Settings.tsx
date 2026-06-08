import { useState, type ReactNode } from 'react';
import { Lock } from 'lucide-react';
import { api } from '../api/client';
import { humanizeError } from '../utils/format';
import { useInvestorProfile, isDefaultProfile, type InvestorProfile } from '../context/InvestorProfile';
import { RISK_NAMES } from '../utils/investorProfile';
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

function SettingRow({ label, description, control }: { label: ReactNode; description: string; control: ReactNode }) {
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
      setToast({ message: humanizeError(e), type: 'error' });
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
  const [baseUrl] = useState(
    () => (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000',
  );
  const [connStatus, setConnStatus] = useState<'idle' | 'checking' | 'ok' | 'error'>('idle');

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
        label={
          <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <Lock size={12} style={{ flexShrink: 0, opacity: 0.55 }} />
            Backend URL
          </span>
        }
        description="FastAPI backend base URL (set via VITE_API_URL at build time)"
        control={
          <div className="settings-url-row">
            <input
              className="settings-input"
              value={baseUrl}
              readOnly
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

function ProfileBtnGroup<T extends number>({
  value, options, onChange,
}: {
  value: T;
  options: readonly string[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="profile-btn-group">
      {options.map((label, i) => (
        <button
          key={i}
          className={`profile-btn${value === i + 1 ? ' active' : ''}`}
          onClick={() => onChange((i + 1) as T)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function ProfileEffectPreview({ profile }: { profile: InvestorProfile }) {
  const rt = profile.riskTolerance;
  const fitChangePct = (rt - 3) * 10;
  const regretChangePct = Math.round((3 - rt) * 15);

  const fitText = fitChangePct > 0 ? `+${fitChangePct}% for BUY`
    : fitChangePct < 0 ? `${fitChangePct}% for BUY`
    : 'Neutral';
  const regretText = regretChangePct > 0 ? `+${regretChangePct}% sensitivity`
    : regretChangePct < 0 ? `${regretChangePct}% sensitivity`
    : 'Neutral';
  const convText = rt <= 2 ? 'Stricter — higher evidence required'
    : rt >= 4 ? 'Looser — acts on weaker signals'
    : 'Standard thresholds';

  const thNote = profile.timeHorizon === 1
    ? 'Short horizon: near-threshold and deteriorating signals carry +10–15 regret risk'
    : profile.timeHorizon === 3 ? 'Long horizon: improving trend signals receive −10 regret risk'
    : null;

  return (
    <div className="profile-preview">
      <div className="profile-preview-title">Score adjustments for this profile</div>
      <div className="profile-preview-grid">
        <div className="pp-row">
          <span className="pp-label">Portfolio Fit</span>
          <span className={`pp-value ${fitChangePct > 0 ? 'positive' : fitChangePct < 0 ? 'negative' : 'neutral'}`}>
            {fitText}
          </span>
        </div>
        <div className="pp-row">
          <span className="pp-label">Regret Risk</span>
          <span className={`pp-value ${regretChangePct > 0 ? 'negative' : regretChangePct < 0 ? 'positive' : 'neutral'}`}>
            {regretText}
          </span>
        </div>
        <div className="pp-row">
          <span className="pp-label">Conviction</span>
          <span className="pp-value neutral">{convText}</span>
        </div>
      </div>
      {thNote && <div className="pp-note">{thNote}</div>}
    </div>
  );
}

function InvestorProfileSection() {
  const { profile, setProfile } = useInvestorProfile();

  const update = <K extends keyof InvestorProfile>(key: K, val: InvestorProfile[K]) => {
    setProfile({ ...profile, [key]: val });
  };

  return (
    <div className="settings-card">
      <SettingRow
        label="Risk Tolerance"
        description="How much investment risk are you comfortable taking?"
        control={
          <ProfileBtnGroup
            value={profile.riskTolerance}
            options={['Very Cons.', 'Conservative', 'Moderate', 'Aggressive', 'Very Aggr.']}
            onChange={v => update('riskTolerance', v)}
          />
        }
      />
      <div className="setting-divider" />
      <SettingRow
        label="Drawdown Tolerance"
        description="Maximum portfolio loss you can accept before feeling distress"
        control={
          <ProfileBtnGroup
            value={profile.drawdownTolerance}
            options={['< 10%', '< 20%', '< 30%', '> 30%']}
            onChange={v => update('drawdownTolerance', v)}
          />
        }
      />
      <div className="setting-divider" />
      <SettingRow
        label="Time Horizon"
        description="How long you plan to hold positions before re-evaluating"
        control={
          <ProfileBtnGroup
            value={profile.timeHorizon}
            options={['Short < 3mo', 'Medium 3-12mo', 'Long > 1yr']}
            onChange={v => update('timeHorizon', v)}
          />
        }
      />
      <div className="setting-divider" />
      <SettingRow
        label="Volatility Preference"
        description="Preferred signal certainty level — low prefers high-conviction signals"
        control={
          <ProfileBtnGroup
            value={profile.volatilityPreference}
            options={['Low', 'Medium', 'High']}
            onChange={v => update('volatilityPreference', v)}
          />
        }
      />
      <ProfileEffectPreview profile={profile} />
      {!isDefaultProfile(profile) && (
        <div className="profile-active-notice">
          Profile active: <strong>{RISK_NAMES[profile.riskTolerance - 1]}</strong> investor —
          scores in Idea Scorecard are personalized for your profile.
        </div>
      )}
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
          <div className="settings-section-title">Investor Profile</div>
          <div className="setting-desc" style={{ marginBottom: 12 }}>
            Personalizes Portfolio Fit, Regret Risk, and Conviction scores in the Idea Scorecard.
            A Conservative investor sees stricter thresholds; an Aggressive investor sees looser ones.
          </div>
          <InvestorProfileSection />
        </div>

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

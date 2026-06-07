import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import './Login.css';

const FEATURES = [
  'ML-driven signal generation',
  'Real-time drift monitoring',
  'Full calibration analytics',
  'Transparent model diagnostics',
];

export function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const canSubmit = email.trim() !== '' && password !== '' && !loading;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setError('');
    setLoading(true);
    try {
      await login(email.trim(), password);
      navigate('/', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-shell">
      {/* Left panel */}
      <div className="login-left">
        <div className="login-brand">
          <div className="login-brand-mark">QS</div>
          <span className="login-brand-name">QUANT-SIGNALS</span>
        </div>

        <div className="login-hero">
          <p className="login-tagline">
            Institutional-grade quantitative research, productized.
          </p>
          <p className="login-sub">
            Signal generation, diagnostics, calibration, and drift monitoring — unified.
          </p>
        </div>

        <ul className="login-features">
          {FEATURES.map((f) => (
            <li key={f} className="login-feature-item">
              <span className="login-feature-check">✓</span>
              <span>{f}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Right panel */}
      <div className="login-right">
        <div className="login-card">
          <h1 className="login-title">Welcome back</h1>
          <p className="login-subtitle">Sign in to continue to Quant-Signals</p>

          {error && (
            <div className="login-error">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate>
            <div className="login-field">
              <label className="login-label">Email address</label>
              <input
                type="email"
                className="login-input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                autoFocus
              />
            </div>

            <div className="login-field">
              <label className="login-label">Password</label>
              <div className="login-pw-wrap">
                <input
                  type={showPw ? 'text' : 'password'}
                  className="login-input"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="login-pw-toggle"
                  onClick={() => setShowPw((v) => !v)}
                  tabIndex={-1}
                >
                  {showPw ? '🙈' : '👁'}
                </button>
              </div>
            </div>

            <button
              type="submit"
              className={`login-btn${canSubmit ? '' : ' disabled'}`}
              disabled={!canSubmit}
            >
              {loading ? 'Signing in…' : 'Sign In'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

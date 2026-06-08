import type { PortfolioImpact, ValidationScorecard } from '../api/types';
import { useInvestorProfile, isDefaultProfile } from '../context/InvestorProfile';
import { applyProfile, RISK_NAMES } from '../utils/investorProfile';
import { ProbHistoryChart } from './ProbHistoryChart';
import './IdeaScorecard.css';

interface Props {
  data: ValidationScorecard;
  loading?: boolean;
  error?: string | null;
}

function ScoreGauge({ value, label, invert }: { value: number; label: string; invert?: boolean }) {
  const pct = Math.max(0, Math.min(100, value));
  const fill = invert
    ? pct <= 30 ? 'var(--positive)' : pct <= 55 ? 'var(--warn)' : 'var(--negative)'
    : pct >= 70 ? 'var(--positive)' : pct >= 45 ? 'var(--warn)' : 'var(--negative)';
  return (
    <div className="score-gauge">
      <div className="score-gauge-top">
        <span className="score-gauge-value" style={{ color: fill }}>{value}</span>
        <span className="score-gauge-denom">/100</span>
      </div>
      <div className="score-gauge-bar-track">
        <div className="score-gauge-bar-fill" style={{ width: `${pct}%`, background: fill }} />
      </div>
      <div className="score-gauge-label">{label}</div>
    </div>
  );
}

function ImpactStat({
  label, value, format, favorableUp, neutral,
}: {
  label: string;
  value: number;
  format: 'pct' | 'raw' | 'score';
  favorableUp: boolean;
  neutral?: boolean;
}) {
  let display: string;
  if (format === 'pct') {
    display = `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
  } else if (format === 'score') {
    display = `${Math.round(value)}/100`;
  } else {
    display = `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
  }
  const isNeutral = (neutral ?? false) || Math.abs(value) < 0.0005;
  const color = isNeutral
    ? 'var(--text-secondary)'
    : favorableUp
      ? (value > 0 ? 'var(--positive)' : 'var(--negative)')
      : (value > 0 ? 'var(--negative)' : 'var(--positive)');
  return (
    <div className="impact-stat">
      <span className="impact-stat-value" style={{ color }}>{display}</span>
      <span className="impact-stat-label">{label}</span>
    </div>
  );
}

function PortfolioImpactSection({ impact }: { impact: PortfolioImpact }) {
  return (
    <div className="scorecard-impact-section">
      <div className="scorecard-impact-header">
        <span className="scorecard-history-title">Portfolio Impact</span>
        {!impact.portfolio_data_available && (
          <span className="scorecard-impact-est">estimated</span>
        )}
      </div>
      <div className="impact-grid">
        <ImpactStat label="Expected Return" value={impact.expected_return_impact} format="pct" favorableUp />
        <ImpactStat label="Volatility Add" value={impact.volatility_impact} format="pct" favorableUp={false} neutral={impact.volatility_impact < 0.001} />
        <ImpactStat label="Max Drawdown" value={impact.max_drawdown_impact} format="pct" favorableUp />
        <ImpactStat label="Diversification" value={impact.diversification_change} format="raw" favorableUp />
        <ImpactStat label="Sector Weight Δ" value={impact.sector_concentration_change} format="pct" favorableUp={false} neutral={Math.abs(impact.sector_concentration_change) < 0.001} />
        <ImpactStat label="Portfolio Fit" value={impact.portfolio_fit_score} format="score" favorableUp />
      </div>
      {!impact.portfolio_data_available && (
        <p className="impact-data-notice">
          Estimates based on signal data. Add holdings for personalized analysis.
        </p>
      )}
    </div>
  );
}

const CONVICTION_COLOR: Record<string, string> = {
  'VERY HIGH': 'var(--positive)',
  HIGH: 'var(--accent)',
  MEDIUM: 'var(--warn)',
  LOW: 'var(--text-tertiary)',
};

const VERDICT_META: Record<string, { icon: string; color: string }> = {
  AGREE: { icon: '✓', color: 'var(--positive)' },
  'PARTIALLY AGREE': { icon: '~', color: 'var(--warn)' },
  DISAGREE: { icon: '✗', color: 'var(--negative)' },
  NEUTRAL: { icon: '—', color: 'var(--text-secondary)' },
};

const TREND_META: Record<string, { icon: string; color: string; label: string }> = {
  IMPROVING: { icon: '↑', color: 'var(--positive)', label: 'Improving' },
  STABLE: { icon: '→', color: 'var(--text-secondary)', label: 'Stable' },
  DETERIORATING: { icon: '↓', color: 'var(--negative)', label: 'Deteriorating' },
  'INSUFFICIENT DATA': { icon: '?', color: 'var(--text-tertiary)', label: 'No trend data' },
};

export function IdeaScorecard({ data, loading, error }: Props) {
  const { profile } = useInvestorProfile();
  const profileActive = !isDefaultProfile(profile);

  if (loading) {
    return (
      <div className="idea-scorecard idea-scorecard-loading">
        <div className="skel" style={{ width: '60%', height: 14, marginBottom: 12 }} />
        <div className="scorecard-grid">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="skel score-gauge-skel" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="idea-scorecard idea-scorecard-error">
        <span className="scorecard-error-text">Scorecard unavailable</span>
      </div>
    );
  }

  const adj = applyProfile(data, profile);

  const verdictMeta = VERDICT_META[data.verdict] ?? VERDICT_META.NEUTRAL;
  const trendMeta = TREND_META[data.signal_trend] ?? TREND_META['INSUFFICIENT DATA'];
  const convictionColor = CONVICTION_COLOR[adj.conviction] ?? 'var(--text-secondary)';

  return (
    <div className="idea-scorecard">
      {/* Idea Score header */}
      <div className="scorecard-header">
        <div className="scorecard-idea-score-wrap">
          <span className="scorecard-idea-score-num">{adj.idea_score}</span>
          <div className="scorecard-idea-score-meta">
            <span className="scorecard-idea-score-label">IDEA SCORE</span>
            <span className="scorecard-suggested-action">{adj.suggested_action}</span>
            {profileActive && (
              <span className="scorecard-profile-tag">
                {RISK_NAMES[profile.riskTolerance - 1]}
              </span>
            )}
          </div>
        </div>
        <div className="scorecard-conviction-badge" style={{ borderColor: convictionColor, color: convictionColor }}>
          {adj.conviction}
        </div>
      </div>

      {/* Score gauges grid — evidence & trust stay objective; fit & regret are personalized */}
      <div className="scorecard-grid">
        <ScoreGauge value={data.evidence_strength} label="Evidence Strength" />
        <ScoreGauge value={adj.portfolio_fit} label={profileActive ? 'Portfolio Fit ✦' : 'Portfolio Fit'} />
        <ScoreGauge value={adj.regret_risk} label={profileActive ? 'Regret Risk ✦' : 'Regret Risk'} invert />
        <ScoreGauge value={data.trust_score} label="Trust Score" />
      </div>

      {/* Signal history chart */}
      <div className="scorecard-history-section">
        <div className="scorecard-history-header">
          <span className="scorecard-history-title">Signal History</span>
          {data.signal_trend !== 'INSUFFICIENT DATA' && (
            <span className="scorecard-trend-pill" style={{ color: trendMeta.color }}>
              {trendMeta.icon} {trendMeta.label}
            </span>
          )}
        </div>
        <ProbHistoryChart history={data.history} currentSignal={data.signal} />
      </div>

      {/* Portfolio Impact section */}
      {data.portfolio_impact && <PortfolioImpactSection impact={data.portfolio_impact} />}

      {/* Second Opinion section */}
      <div className="second-opinion">
        <div className="second-opinion-header">
          <span className="second-opinion-title">Quant Second Opinion</span>
          <span
            className="second-opinion-verdict-badge"
            style={{ background: `color-mix(in srgb, ${verdictMeta.color} 12%, transparent)`, color: verdictMeta.color, borderColor: `color-mix(in srgb, ${verdictMeta.color} 30%, transparent)` }}
          >
            <span className="verdict-icon">{verdictMeta.icon}</span>
            {data.verdict}
          </span>
        </div>
        <ul className="second-opinion-reasons">
          {data.verdict_reasons.map((reason, i) => (
            <li key={i} className="second-opinion-reason">{reason}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

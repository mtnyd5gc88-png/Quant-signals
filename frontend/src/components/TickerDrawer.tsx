import { useEffect } from 'react';
import { X } from 'lucide-react';
import { SignalBadge } from './SignalBadge';
import { ProbBar } from './ProbBar';
import { IdeaScorecard } from './IdeaScorecard';
import { useValidation } from '../api/hooks';
import type { SignalItem, PortfolioHolding } from '../api/types';
import { fmtPrice, fmtPctSigned, colorClass } from '../utils/format';
import './TickerDrawer.css';

interface RecentValidation {
  ticker: string;
  company?: string;
  signal: string;
  prob_up: number;
  ts: number;
}

const MAX_WEIGHT = 0.15;

function signalStrength(prob: number): string {
  if (prob > 0.65) return 'strong';
  if (prob >= 0.55) return 'moderate';
  return 'weak';
}

function rocReliability(roc: number): string {
  if (roc > 0.65) return 'above-average';
  if (roc >= 0.55) return 'average';
  return 'below-average';
}

interface Props {
  item: SignalItem | null;
  onClose: () => void;
  rocAuc: number;
  regime: string;
  portfolioHolding?: PortfolioHolding;
}

export function TickerDrawer({ item, onClose, rocAuc, regime, portfolioHolding }: Props) {
  const { data: scorecard, loading: scorecardLoading, error: scorecardError } = useValidation(item?.ticker ?? null);

  useEffect(() => {
    if (!item) return;
    try {
      const key = 'qs_recent_validations';
      const prev: RecentValidation[] = JSON.parse(localStorage.getItem(key) ?? '[]');
      const entry: RecentValidation = { ticker: item.ticker, company: item.company, signal: item.signal, prob_up: item.prob_up, ts: Date.now() };
      localStorage.setItem(key, JSON.stringify([entry, ...prev.filter(r => r.ticker !== item.ticker)].slice(0, 20)));
      window.dispatchEvent(new Event('qs:recent-updated'));
    } catch (_) { /* ignore */ }
  }, [item?.ticker]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!item) return null;

  const retVal = item.target_return ?? item.expected_return ?? 0;
  const retClass = colorClass(retVal);
  const pct = Math.round(item.prob_up * 100);

  const warnings: string[] = [];
  if (item.prob_up >= 0.55 && item.prob_up < 0.60) {
    warnings.push('Probability near threshold (55–60%) — signal is weak');
  }
  if (regime === 'RISK-OFF') {
    warnings.push('Market in risk-off regime — consider reducing position size');
  }
  if (item.last_prediction) {
    const age = Date.now() - new Date(item.last_prediction).getTime();
    if (age > 24 * 3600 * 1000) {
      warnings.push('Data last updated >24h ago — signal may be stale');
    }
  }

  const direction = item.signal === 'SELL' ? 'bearish' : 'bullish';
  const strength = signalStrength(item.prob_up);
  const reliability = rocReliability(rocAuc);

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="ticker-drawer" role="dialog" aria-label={`${item.ticker} detail`}>
        <div className="drawer-header">
          <div className="drawer-header-left">
            <span className="drawer-ticker">{item.ticker}</span>
            {item.company && <span className="drawer-company">{item.company}</span>}
          </div>
          <div className="drawer-header-right">
            <SignalBadge signal={item.signal} />
            <button className="drawer-close" onClick={onClose} aria-label="Close">
              <X size={16} strokeWidth={2} />
            </button>
          </div>
        </div>

        <div className="drawer-body">
          {/* Key Numbers */}
          <section className="drawer-section">
            <div className="drawer-kpi-grid">
              <div className="drawer-kpi">
                <span className="drawer-kpi-label">Current Price</span>
                <span className="drawer-kpi-value">${fmtPrice(item.price)}</span>
              </div>
              <div className="drawer-kpi">
                <span className="drawer-kpi-label">P(Up)</span>
                <span className="drawer-kpi-value">{pct}%</span>
              </div>
              <div className="drawer-kpi">
                <span className="drawer-kpi-label">Est. Return (5d)</span>
                <span className={`drawer-kpi-value ${retClass}`}>{fmtPctSigned(retVal)}</span>
              </div>
              {item.target_price && (
                <div className="drawer-kpi">
                  <span className="drawer-kpi-label">Target</span>
                  <span className="drawer-kpi-value">${fmtPrice(item.target_price)}</span>
                </div>
              )}
              <div className="drawer-kpi">
                <span className="drawer-kpi-label">Model</span>
                <span className="drawer-kpi-value drawer-kpi-model">RF</span>
              </div>
            </div>
          </section>

          <div className="drawer-divider" />

          {/* Signal Explanation */}
          <section className="drawer-section">
            <h4 className="drawer-section-title">Signal Explanation</h4>
            <p className="drawer-explanation">
              <strong>{item.ticker}</strong> shows a <strong>{strength}</strong> {direction} signal with{' '}
              <strong>{pct}%</strong> probability of upward movement. The model&apos;s mean ROC-AUC
              across the universe is <strong>{rocAuc.toFixed(2)}</strong>, indicating{' '}
              <strong>{reliability}</strong> predictive reliability. Current market regime:{' '}
              <strong>{regime}</strong>.
            </p>
          </section>

          <div className="drawer-divider" />

          {/* Idea Scorecard + Second Opinion */}
          <section className="drawer-section">
            <h4 className="drawer-section-title">Idea Scorecard</h4>
            <IdeaScorecard
              data={scorecard!}
              loading={scorecardLoading || !scorecard}
              error={scorecardError}
            />
          </section>

          <div className="drawer-divider" />

          {/* Trust Breakdown */}
          <section className="drawer-section">
            <h4 className="drawer-section-title">Model Reliability</h4>
            <div className="drawer-trust-block">
              <ProbBar value={item.prob_up} rocAuc={rocAuc} avgTrust />
              <p className="drawer-trust-sub">Based on walk-forward cross-validated ROC-AUC</p>
              <span className="drawer-roc-link-wrap">
                <span className="drawer-roc-link">
                  What is this?
                  <span className="drawer-roc-tooltip" role="tooltip">
                    ROC-AUC measures how well the model distinguishes up-moves from down-moves.
                    0.5 = random, 1.0 = perfect. Above 0.60 is considered useful.
                  </span>
                </span>
              </span>
            </div>
          </section>

          {/* Portfolio Implication */}
          {portfolioHolding && (
            <>
              <div className="drawer-divider" />
              <section className="drawer-section">
                <h4 className="drawer-section-title">Portfolio Implication</h4>
                <div className="drawer-alloc">
                  <div className="drawer-alloc-row">
                    <span className="drawer-alloc-label">Currently allocated</span>
                    <span className="drawer-alloc-val">{(portfolioHolding.weight * 100).toFixed(1)}% of portfolio</span>
                  </div>
                  <div className="drawer-alloc-row">
                    <span className="drawer-alloc-label">Max allowed</span>
                    <span className={`drawer-alloc-val${portfolioHolding.weight > MAX_WEIGHT ? ' over-cap' : ''}`}>
                      {(MAX_WEIGHT * 100).toFixed(1)}% (hard cap)
                    </span>
                  </div>
                  <div className="drawer-alloc-bar-track">
                    <div
                      className={`drawer-alloc-bar-fill${portfolioHolding.weight > MAX_WEIGHT ? ' over' : ''}`}
                      style={{ width: `${Math.min((portfolioHolding.weight / MAX_WEIGHT) * 100, 100)}%` }}
                    />
                  </div>
                </div>
              </section>
            </>
          )}

          {/* Warning Flags */}
          {warnings.length > 0 && (
            <>
              <div className="drawer-divider" />
              <section className="drawer-section">
                {warnings.map((w, i) => (
                  <div key={i} className="drawer-warning">
                    <span className="drawer-warning-icon">⚠️</span>
                    <span className="drawer-warning-text">{w}</span>
                  </div>
                ))}
              </section>
            </>
          )}
        </div>
      </div>
    </>
  );
}

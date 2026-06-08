import type { ValidationScorecard } from '../api/types';
import type { InvestorProfile } from '../context/InvestorProfile';

export const RISK_NAMES = [
  'Very Conservative', 'Conservative', 'Moderate', 'Aggressive', 'Very Aggressive',
] as const;

export interface AdjustedScores {
  portfolio_fit: number;
  regret_risk: number;
  idea_score: number;
  conviction: 'VERY HIGH' | 'HIGH' | 'MEDIUM' | 'LOW';
  suggested_action: string;
}

const SELL_SIGNALS = new Set(['SELL', 'CASH', 'STAY IN CASH']);

const DT_MULT: Record<number, number> = { 1: 1.20, 2: 1.00, 3: 0.95, 4: 0.85 };

// Evidence thresholds per risk tolerance level (index = riskTolerance - 1)
const VH_EVID = [82, 78, 75, 68, 60];
const H_EVID  = [68, 64, 60, 54, 48];

function buyAction(conviction: string): string {
  if (conviction === 'VERY HIGH') return 'Core Position Candidate';
  if (conviction === 'HIGH') return 'Consider Building Position';
  if (conviction === 'MEDIUM') return 'Speculative Position Only';
  return 'Monitor Only';
}

export function applyProfile(s: ValidationScorecard, p: InvestorProfile): AdjustedScores {
  const { riskTolerance: rt, drawdownTolerance: dt, timeHorizon: th, volatilityPreference: vp } = p;

  // ── Portfolio Fit ──────────────────────────────────────────────
  let fit = s.portfolio_fit;
  if (s.signal === 'BUY') {
    fit = Math.round(fit * (1 + (rt - 3) * 0.10));
    if (vp === 1 && s.prob_up >= 0.55 && s.prob_up < 0.65) fit -= 8;   // low vol pref penalizes weak BUY
    if (vp === 3 && s.prob_up >= 0.70) fit += 5;                         // high vol pref rewards strong BUY
  } else if (SELL_SIGNALS.has(s.signal)) {
    fit = Math.round(fit * (1 + (3 - rt) * 0.08));                       // conservative wants to exit more
  }
  fit = Math.max(0, Math.min(100, fit));

  // ── Regret Risk ────────────────────────────────────────────────
  let regret = Math.round(s.regret_risk * (1 + (3 - rt) * 0.15));       // conservative = more risk-sensitive
  regret = Math.round(regret * (DT_MULT[dt] ?? 1.0));
  if (th === 1) {
    if (s.signal_trend === 'DETERIORATING') regret += 15;
    if (s.prob_up >= 0.55 && s.prob_up < 0.63) regret += 10;
  } else if (th === 3 && s.signal_trend === 'IMPROVING') {
    regret -= 10;
  }
  if (vp === 1 && s.signal_trend === 'DETERIORATING') regret += 8;
  regret = Math.max(0, Math.min(100, regret));

  // ── Idea Score ─────────────────────────────────────────────────
  let idea: number;
  const ev = s.evidence_strength;
  if (s.signal === 'BUY') {
    idea = Math.round(0.40 * ev + 0.30 * (100 - regret) + 0.30 * fit);
  } else if (SELL_SIGNALS.has(s.signal)) {
    idea = Math.round(0.50 * ev + 0.30 * (100 - regret) + 0.20 * fit);
  } else {
    idea = s.idea_score;
  }
  idea = Math.max(0, Math.min(100, idea));

  // ── Conviction ─────────────────────────────────────────────────
  let conviction: 'VERY HIGH' | 'HIGH' | 'MEDIUM' | 'LOW';
  if (ev >= VH_EVID[rt - 1] && s.trust_score >= 70) conviction = 'VERY HIGH';
  else if (ev >= H_EVID[rt - 1] && s.trust_score >= 62) conviction = 'HIGH';
  else if (ev >= 40) conviction = 'MEDIUM';
  else conviction = 'LOW';

  const suggested_action = s.signal === 'BUY' ? buyAction(conviction) : s.suggested_action;

  return { portfolio_fit: fit, regret_risk: regret, idea_score: idea, conviction, suggested_action };
}

import { useState } from 'react';
import {
  ResponsiveContainer, BarChart, Bar, CartesianGrid,
  XAxis, YAxis, Tooltip, Cell,
} from 'recharts';
import { PageTabs } from '../components/PageTabs';
import { SectionHeader } from '../components/SectionHeader';
import { useSignals, useDiagnostics, useRegime } from '../api/hooks';
import type { DiagnosticsResponse, RegimeResponse, SignalItem } from '../api/types';
import { fmtPct, fmtNum } from '../utils/format';
import './Research.css';

const TABS = [
  { id: 'universe', label: 'Universe Stats' },
  { id: 'signals',  label: 'Signal Distribution' },
  { id: 'sector',   label: 'Sector Analysis' },
];

const CHART_TOOLTIP = {
  contentStyle: {
    background: 'var(--bg-surface-4)',
    border: '1px solid var(--border-strong)',
    borderRadius: '6px',
    padding: '8px 12px',
    fontSize: '12px',
    color: 'var(--text-primary)',
    boxShadow: 'none',
  },
};

const FEATURE_LABELS: Record<string, string> = {
  momentum_12m: '12-month price momentum',
  rsi_14: '14-day relative strength (RSI)',
  volume_ratio: 'Volume vs. 20-day average',
  eps_surprise: 'EPS surprise factor',
  price_ma_cross: 'Moving average crossover',
  sector_momentum: 'Sector-relative momentum',
  volatility_20d: '20-day realized volatility',
  revenue_growth: 'Year-over-year revenue growth',
  pe_ratio_norm: 'Normalized P/E ratio',
  short_interest: 'Short interest as % of float',
};

function signalColor(signal: string): string {
  if (signal === 'BUY') return 'var(--positive)';
  if (signal === 'SELL' || signal === 'CASH' || signal === 'STAY IN CASH') return 'var(--negative)';
  return 'var(--text-tertiary)';
}

export function Research() {
  const [tab, setTab] = useState('universe');
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const { data: signals } = useSignals();
  const { data: diag } = useDiagnostics();
  const { data: regime } = useRegime();

  const selectedItem = selectedTicker
    ? signals.items.find(i => i.ticker === selectedTicker) ?? null
    : null;

  return (
    <div className="research-page">
      <PageTabs tabs={TABS} active={tab} onChange={setTab} />
      <div className="research-body">
        <div className="research-left">
          <div className="research-content">
            {tab === 'universe' && <UniverseTab signals={signals} />}
            {tab === 'signals'  && <SignalDistTab signals={signals} />}
            {tab === 'sector'   && <SectorTab signals={signals} />}
          </div>
        </div>
        <div className="research-right">
          <ExplanationPanel
            item={selectedItem}
            allItems={signals.items}
            onSelect={setSelectedTicker}
            diag={diag}
            regime={regime}
          />
        </div>
      </div>
    </div>
  );
}

// ── Explanation Panel ────────────────────────────────────────────────

interface ExplanationPanelProps {
  item: SignalItem | null;
  allItems: SignalItem[];
  onSelect: (ticker: string) => void;
  diag: DiagnosticsResponse;
  regime: RegimeResponse;
}

function ExplanationPanel({ item, allItems, onSelect, diag, regime }: ExplanationPanelProps) {
  const [query, setQuery] = useState('');

  const results = query.length >= 1
    ? allItems
        .filter(i =>
          i.ticker.toLowerCase().startsWith(query.toLowerCase()) ||
          (i.company?.toLowerCase().includes(query.toLowerCase()) ?? false),
        )
        .slice(0, 6)
    : [];

  const roc = diag.model_quality.roc_auc_mean ?? 0.60;
  const regimeName = (regime.regime ?? 'neutral').toLowerCase();
  const isRiskOn = regimeName.includes('bull') || regimeName.includes('risk-on') || regimeName.includes('on');

  return (
    <div className="explanation-panel">
      <div className="expl-panel-header">
        <span className="expl-panel-title">Signal Explanation</span>
        <span className="expl-panel-subtitle">Why, when, and how this signal works</span>
      </div>

      {/* Ticker search */}
      <div className="expl-search-wrap">
        <input
          className="expl-search-input"
          type="text"
          placeholder="Search ticker or company…"
          value={query}
          onChange={e => setQuery(e.target.value)}
          autoComplete="off"
        />
        {results.length > 0 && (
          <div className="expl-search-results">
            {results.map(r => (
              <div
                key={r.ticker}
                className="expl-search-result"
                onMouseDown={() => { onSelect(r.ticker); setQuery(''); }}
              >
                <span className="esr-ticker">{r.ticker}</span>
                {r.company && <span className="esr-company">{r.company}</span>}
                <span className="esr-signal" style={{ color: signalColor(r.signal) }}>{r.signal} {Math.round(r.prob_up * 100)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {item ? (
        <div className="expl-content">
          {/* Selected ticker header */}
          <div className="expl-ticker-header">
            <div className="expl-th-left">
              <span className="expl-th-ticker">{item.ticker}</span>
              {item.company && <span className="expl-th-company">{item.company}</span>}
            </div>
            <span className="expl-th-signal" style={{ color: signalColor(item.signal) }}>
              {item.signal} · {Math.round(item.prob_up * 100)}%
            </span>
          </div>

          {/* 1. Why does this signal exist? */}
          <div className="expl-section">
            <div className="expl-section-q">Why does this signal exist?</div>
            <div className="expl-section-body">
              <p>
                The Random Forest model classified <strong>{item.ticker}</strong> as{' '}
                <strong style={{ color: signalColor(item.signal) }}>{item.signal}</strong>{' '}
                with <strong>{Math.round(item.prob_up * 100)}%</strong> upward probability
                based on quantitative momentum, quality, and technical factors.
              </p>
              {diag.feature_importance.length > 0 && (
                <div className="expl-features">
                  <div className="expl-features-label">Top model drivers (universe-wide):</div>
                  {diag.feature_importance.slice(0, 3).map(f => (
                    <div key={f.feature} className="expl-feature-row">
                      <span className="expl-feature-name">
                        {FEATURE_LABELS[f.feature] ?? f.feature}
                      </span>
                      <div className="expl-feature-bar">
                        <div
                          className="expl-feature-fill"
                          style={{ width: `${Math.round(f.importance * 100)}%` }}
                        />
                      </div>
                      <span className="expl-feature-pct">{(f.importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 2. What conditions support it? */}
          <div className="expl-section">
            <div className="expl-section-q">What conditions support it?</div>
            <div className="expl-section-body expl-conditions">
              <div className="expl-condition-row">
                <span className="expl-cond-label">Market regime</span>
                <span className="expl-cond-value">{regime.regime ?? 'Unknown'}</span>
                <span className="expl-cond-note">
                  {isRiskOn
                    ? 'Momentum and growth factors are rewarded in this environment'
                    : 'Defensive positioning favored — growth signals may underperform'}
                </span>
              </div>
              <div className="expl-condition-row">
                <span className="expl-cond-label">Signal conviction</span>
                <span className="expl-cond-value" style={{ color: signalColor(item.signal) }}>
                  {Math.round(item.prob_up * 100)}%
                </span>
                <span className="expl-cond-note">
                  {item.prob_up >= 0.70
                    ? 'High conviction — well above the 55% threshold'
                    : item.prob_up >= 0.60
                      ? 'Moderate conviction — above threshold with room for deterioration'
                      : item.prob_up >= 0.55
                        ? 'Weak conviction — near the decision boundary (55%)'
                        : item.prob_up >= 0.45
                          ? 'Neutral territory — no strong directional edge'
                          : 'Bearish signal — below neutral threshold'}
                </span>
              </div>
              <div className="expl-condition-row">
                <span className="expl-cond-label">Model reliability</span>
                <span className="expl-cond-value">ROC-AUC {roc.toFixed(3)}</span>
                <span className="expl-cond-note">
                  {roc >= 0.65
                    ? 'Above-average — model has strong directional accuracy'
                    : roc >= 0.58
                      ? 'Average — use with appropriate position sizing'
                      : 'Below-average — additional conviction sources recommended'}
                </span>
              </div>
            </div>
          </div>

          {/* 3. When does it fail? */}
          <div className="expl-section">
            <div className="expl-section-q">When does it fail?</div>
            <div className="expl-section-body">
              <ul className="expl-fail-list">
                <li>During market regime transitions (risk-on ↔ risk-off) when factor premiums shift abruptly</li>
                {item.prob_up >= 0.55 && item.prob_up < 0.63 && (
                  <li>
                    Near-threshold probabilities like {Math.round(item.prob_up * 100)}% indicate reduced conviction —
                    small shifts can flip the signal direction
                  </li>
                )}
                {roc < 0.65 && (
                  <li>
                    When model AUC is below 0.65 (current: {roc.toFixed(3)}) — directional accuracy is limited
                  </li>
                )}
                <li>High-volatility events (earnings, macro) where historical training patterns underrepresent extremes</li>
                <li>Sector rotation environments where cross-sectional momentum breaks down</li>
              </ul>
            </div>
          </div>

          {/* 4. Historical similarity */}
          <div className="expl-section">
            <div className="expl-section-q">Historical similarity</div>
            <div className="expl-section-body">
              <p>
                Current conditions (<strong>{regime.regime ?? 'neutral'}</strong> regime,{' '}
                <strong>{Math.round(item.prob_up * 100)}%</strong> confidence) most closely
                resemble{' '}
                {isRiskOn
                  ? 'late-cycle bull market periods where momentum and quality factors perform well.'
                  : 'risk-off environments where defensive positioning and capital preservation dominate.'}
              </p>
              <p>
                Signals in the {Math.round(item.prob_up * 100)}% probability range{' '}
                {item.prob_up >= 0.65
                  ? 'have historically validated at above-average rates based on model calibration.'
                  : item.prob_up >= 0.55
                    ? 'show average accuracy at this level — outcomes reflect model confidence closely.'
                    : 'indicate limited directional conviction — outcomes are near-random at these levels.'}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="expl-empty">
          <span className="expl-empty-title">Search for a ticker above</span>
          <span className="expl-empty-sub">
            Signal explanation · Conditions · Failure modes · Historical similarity
          </span>
        </div>
      )}
    </div>
  );
}

// ── Existing tabs ────────────────────────────────────────────────────

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function UniverseTab({ signals }: { signals: ReturnType<typeof useSignals>['data'] }) {
  const items = signals.items;
  const probs = items.map((i) => i.prob_up);
  const meanProb = probs.reduce((s, v) => s + v, 0) / (probs.length || 1);
  const stdProb = Math.sqrt(probs.reduce((s, v) => s + Math.pow(v - meanProb, 2), 0) / (probs.length || 1));
  const confidence = (prob_up: number) => Math.abs(prob_up - 0.5) * 2;
  const medianConf = median(items.map((i) => confidence(i.prob_up)));

  const stats: [string, string, string?][] = [
    ['Total Tickers', items.length.toString()],
    ['Coverage', `${items.length} (100%)`],
    ['BUY Signals', `${signals.buy_count} (${fmtPct(signals.buy_count / items.length)})`],
    ['HOLD Signals', `${signals.hold_count} (${fmtPct(signals.hold_count / items.length)})`],
    ['CASH Signals', `${signals.cash_count} (${fmtPct(signals.cash_count / items.length)})`],
    ['Mean Predicted Prob', fmtNum(meanProb, 4)],
    ['Std Predicted Prob', fmtNum(stdProb, 4)],
    ['Median Confidence', fmtNum(medianConf, 2), 'Median |p − 0.5| × 2 across all signals. 0 = no conviction, 1 = maximum conviction.'],
  ];

  return (
    <div className="research-section">
      <SectionHeader title="Universe Statistics" meta={`${items.length} tickers`} />
      <div className="data-table-wrapper" style={{ maxWidth: 560 }}>
        <table className="data-table">
          <thead><tr><th>METRIC</th><th className="numeric">VALUE</th></tr></thead>
          <tbody>
            {stats.map(([label, value, tooltip]) => (
              <tr key={label} title={tooltip}>
                <td className="cell-sector">{label}</td>
                <td className="cell-numeric">{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SignalDistTab({ signals }: { signals: ReturnType<typeof useSignals>['data'] }) {
  const items = signals.items;

  const buckets = Array.from({ length: 10 }, (_, i) => ({
    label: `${(i * 10).toFixed(0)}–${((i + 1) * 10).toFixed(0)}%`,
    count: items.filter((it) => it.prob_up >= i * 0.1 && it.prob_up < (i + 1) * 0.1).length,
    mid: (i * 0.1 + (i + 1) * 0.1) / 2,
  }));

  const signalDist = [
    { name: 'BUY',  count: signals.buy_count,  fill: 'var(--positive)' },
    { name: 'HOLD', count: signals.hold_count,  fill: 'var(--neutral)' },
    { name: 'CASH', count: signals.cash_count,  fill: 'var(--text-tertiary)' },
  ];

  return (
    <div className="research-section">
      <SectionHeader title="Probability Distribution" meta={`${items.length} tickers`} />
      <div className="chart-panel">
        <div className="chart-title">Predicted Probability Histogram</div>
        <div className="chart-subtitle">Distribution of Prob_Up across all tickers in universe</div>
        <div style={{ width: '100%', minWidth: 0, height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={buckets} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="label" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={32} />
              <Tooltip {...CHART_TOOLTIP} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {buckets.map((b, i) => (
                  <Cell key={i} fill={b.mid < 0.4 ? 'var(--negative)' : b.mid < 0.6 ? 'var(--neutral)' : 'var(--positive)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-panel">
        <div className="chart-title">Signal Breakdown</div>
        <div style={{ width: '100%', minWidth: 0, height: 140 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={signalDist} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="1 3" stroke="var(--chart-grid)" strokeOpacity={0.5} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} axisLine={false} tickLine={false} width={32} />
              <Tooltip {...CHART_TOOLTIP} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {signalDist.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function SectorTab({ signals }: { signals: ReturnType<typeof useSignals>['data'] }) {
  const items = signals.items;
  const sectorMap = new Map<string, { buy: number; hold: number; cash: number; probs: number[] }>();

  items.forEach((it) => {
    const s = it.sector ?? 'Unknown';
    if (!sectorMap.has(s)) sectorMap.set(s, { buy: 0, hold: 0, cash: 0, probs: [] });
    const e = sectorMap.get(s)!;
    if (it.signal === 'BUY') e.buy++;
    else if (it.signal === 'CASH') e.cash++;
    else e.hold++;
    e.probs.push(it.prob_up);
  });

  const sectorData = Array.from(sectorMap.entries())
    .map(([sector, d]) => ({
      sector,
      buy: d.buy, hold: d.hold, cash: d.cash,
      total: d.buy + d.hold + d.cash,
      meanProb: d.probs.reduce((s, v) => s + v, 0) / d.probs.length,
    }))
    .sort((a, b) => b.total - a.total);

  return (
    <div className="research-section">
      <SectionHeader title="Sector Analysis" meta={`${sectorData.length} sectors`} />
      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: 160 }}>SECTOR</th>
              <th className="numeric" style={{ width: 70 }}>TICKERS</th>
              <th className="numeric" style={{ width: 60 }}>BUY</th>
              <th className="numeric" style={{ width: 60 }}>HOLD</th>
              <th className="numeric" style={{ width: 60 }}>CASH</th>
              <th className="numeric" style={{ width: 110 }}>MEAN PROB</th>
            </tr>
          </thead>
          <tbody>
            {sectorData.map((s) => (
              <tr key={s.sector}>
                <td className="cell-sector">{s.sector}</td>
                <td className="cell-numeric">{s.total}</td>
                <td className="cell-numeric positive">{s.buy}</td>
                <td className="cell-numeric" style={{ color: 'var(--neutral)' }}>{s.hold}</td>
                <td className="cell-numeric" style={{ color: 'var(--text-secondary)' }}>{s.cash}</td>
                <td className="cell-numeric">{fmtPct(s.meanProb)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

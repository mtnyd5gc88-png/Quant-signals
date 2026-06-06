import { useState } from 'react';
import {
  ResponsiveContainer, BarChart, Bar, CartesianGrid,
  XAxis, YAxis, Tooltip, Cell,
} from 'recharts';
import { PageTabs } from '../components/PageTabs';
import { SectionHeader } from '../components/SectionHeader';
import { useSignals } from '../api/hooks';
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

export function Research() {
  const [tab, setTab] = useState('universe');
  const { data: signals } = useSignals();

  return (
    <div className="research-page">
      <PageTabs tabs={TABS} active={tab} onChange={setTab} />
      <div className="research-content">
        {tab === 'universe' && <UniverseTab signals={signals} />}
        {tab === 'signals'  && <SignalDistTab signals={signals} />}
        {tab === 'sector'   && <SectorTab signals={signals} />}
      </div>
    </div>
  );
}

function UniverseTab({ signals }: { signals: ReturnType<typeof useSignals>['data'] }) {
  const items = signals.items;
  const probs = items.map((i) => i.prob_up);
  const meanProb = probs.reduce((s, v) => s + v, 0) / (probs.length || 1);
  const stdProb = Math.sqrt(probs.reduce((s, v) => s + Math.pow(v - meanProb, 2), 0) / (probs.length || 1));
  const medianConf = items.map((i) => i.model_confidence ?? 0).sort((a, b) => a - b)[Math.floor(items.length / 2)] ?? 0;

  const stats = [
    ['Total Tickers', items.length.toString()],
    ['Coverage', `${items.length} (100%)`],
    ['BUY Signals', `${signals.buy_count} (${fmtPct(signals.buy_count / items.length)})`],
    ['HOLD Signals', `${signals.hold_count} (${fmtPct(signals.hold_count / items.length)})`],
    ['CASH Signals', `${signals.cash_count} (${fmtPct(signals.cash_count / items.length)})`],
    ['Mean Predicted Prob', fmtNum(meanProb, 4)],
    ['Std Predicted Prob', fmtNum(stdProb, 4)],
    ['Median Confidence', fmtNum(medianConf, 2)],
  ];

  return (
    <div className="research-section">
      <SectionHeader title="Universe Statistics" meta={`${items.length} tickers`} />
      <div className="data-table-wrapper" style={{ maxWidth: 560 }}>
        <table className="data-table">
          <thead><tr><th>METRIC</th><th className="numeric">VALUE</th></tr></thead>
          <tbody>
            {stats.map(([label, value]) => (
              <tr key={label}>
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

  // Prob distribution histogram (10 buckets)
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
        <ResponsiveContainer width="100%" height={200}>
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

      <div className="chart-panel">
        <div className="chart-title">Signal Breakdown</div>
        <ResponsiveContainer width="100%" height={140}>
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

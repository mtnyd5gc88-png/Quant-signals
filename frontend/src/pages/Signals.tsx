import { useState } from 'react';
import { SignalBadge } from '../components/SignalBadge';
import { ProbBar } from '../components/ProbBar';
import { ConfidenceDots } from '../components/ConfidenceDots';
import { Sparkline } from '../components/Sparkline';
import { useSignals } from '../api/hooks';
import { fmtPctSigned, fmtPrice, fmtRelTime, colorClass } from '../utils/format';
import type { SignalItem } from '../api/types';
import './Signals.css';

const FILTERS = ['ALL', 'BUY', 'HOLD', 'CASH'] as const;

export function Signals() {
  const [filter, setFilter] = useState('ALL');
  const [sortBy, setSortBy] = useState('prob_up');
  const [order, setOrder] = useState<'asc' | 'desc'>('desc');
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data, loading } = useSignals(filter, sortBy, order, search || undefined);

  function toggleSort(col: string) {
    if (sortBy === col) setOrder((o) => o === 'desc' ? 'asc' : 'desc');
    else { setSortBy(col); setOrder('desc'); }
  }

  function SortIcon({ col }: { col: string }) {
    if (sortBy !== col) return <span className="sort-icon">↕</span>;
    return <span className="sort-icon">{order === 'desc' ? '▼' : '▲'}</span>;
  }

  return (
    <div className="signals-page">
      {/* Filter Bar */}
      <div className="filter-bar">
        <div className="filter-pills">
          {FILTERS.map((f) => (
            <button
              key={f}
              className={`filter-pill${filter === f ? ' active' : ''}`}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>

        <div className="filter-divider" />

        <div className="filter-bar-search">
          <input
            type="text"
            placeholder="Filter ticker..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <select
          className="sort-dropdown"
          value={`${sortBy}:${order}`}
          onChange={(e) => {
            const [col, ord] = e.target.value.split(':');
            setSortBy(col);
            setOrder(ord as 'asc' | 'desc');
          }}
        >
          <option value="prob_up:desc">Sort: Prob ↓</option>
          <option value="prob_up:asc">Sort: Prob ↑</option>
          <option value="target_return:desc">Sort: Return ↓</option>
          <option value="target_return:asc">Sort: Return ↑</option>
          <option value="ticker:asc">Sort: Ticker A→Z</option>
        </select>

        <span className="filter-bar-right">
          {data.items.length} of {data.total} &nbsp;·&nbsp;
          <span className="positive">{data.buy_count} BUY</span>
          &nbsp;·&nbsp;
          <span style={{ color: 'var(--neutral)' }}>{data.hold_count} HOLD</span>
          &nbsp;·&nbsp;
          {data.cash_count} CASH
        </span>
      </div>

      {/* Table */}
      <div className="table-scroll-wrapper">
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 200 }}>TICKER</th>
                <th style={{ width: 80 }} className="th-center">SIGNAL</th>
                <th style={{ width: 130 }} className="numeric sortable" onClick={() => toggleSort('prob_up')}>
                  PROB UP <SortIcon col="prob_up" />
                </th>
                <th style={{ width: 110 }} className="numeric sortable" onClick={() => toggleSort('target_return')}>
                  EXP RETURN <SortIcon col="target_return" />
                </th>
                <th style={{ width: 110 }} className="numeric">TARGET PRICE</th>
                <th style={{ width: 110 }} className="numeric">CURRENT PRICE</th>
                <th style={{ width: 90 }} className="numeric">POS WT</th>
                <th style={{ width: 120 }}>SECTOR</th>
                <th style={{ width: 120 }} className="numeric">CONFIDENCE</th>
                <th style={{ width: 100 }} className="numeric">LAST PRED</th>
                <th style={{ width: 80 }} className="th-center">TREND</th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 15 }, (_, i) => (
                    <tr key={i}>
                      {Array.from({ length: 11 }, (__, j) => (
                        <td key={j}><div className="skeleton skeleton-text" /></td>
                      ))}
                    </tr>
                  ))
                : data.items.map((item) => (
                    <SignalRow
                      key={item.ticker}
                      item={item}
                      expanded={expanded === item.ticker}
                      onToggle={() => setExpanded(expanded === item.ticker ? null : item.ticker)}
                    />
                  ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SignalRow({ item, expanded, onToggle }: { item: SignalItem; expanded: boolean; onToggle: () => void }) {
  const retVal = item.target_return ?? item.expected_return ?? 0;
  const retClass = colorClass(retVal);

  return (
    <>
      <tr className={`signal-row${expanded ? ' expanded' : ''}`} onClick={onToggle}>
        <td>
          <div className="ticker-cell">
            <div className="company-logo-circle">{item.ticker.slice(0, 2)}</div>
            <div>
              <div className="ticker-symbol">{item.ticker}</div>
              {item.company && <div className="company-name">{item.company}</div>}
            </div>
          </div>
        </td>
        <td className="td-center"><SignalBadge signal={item.signal} /></td>
        <td><ProbBar value={item.prob_up} /></td>
        <td className={`cell-numeric ${retClass}`}>
          {fmtPctSigned(retVal)}
        </td>
        <td className="cell-numeric">{item.target_price ? `$${fmtPrice(item.target_price)}` : '—'}</td>
        <td className="cell-numeric">${fmtPrice(item.price)}</td>
        <td className="cell-numeric">
          {item.position_weight ? `${(item.position_weight * 100).toFixed(2)}%` : '—'}
        </td>
        <td className="cell-sector">{item.sector ?? '—'}</td>
        <td><ConfidenceDots value={item.model_confidence ?? 0.5} /></td>
        <td className="cell-numeric cell-sm">
          {item.last_prediction ? fmtRelTime(item.last_prediction) : '—'}
        </td>
        <td className="td-center">
          <Sparkline data={item.trend ?? [item.price]} />
        </td>
      </tr>
      {expanded && (
        <tr className="expanded-detail-row">
          <td colSpan={11}>
            <div className="expanded-detail">
              <div className="expanded-stat">
                <span className="expanded-label">Alpha Score</span>
                <span className="expanded-value">{item.alpha_score?.toFixed(4) ?? '—'}</span>
              </div>
              <div className="expanded-stat">
                <span className="expanded-label">Probability Up</span>
                <span className="expanded-value">{(item.prob_up * 100).toFixed(2)}%</span>
              </div>
              <div className="expanded-stat">
                <span className="expanded-label">Model Confidence</span>
                <span className="expanded-value">{((item.model_confidence ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="expanded-stat">
                <span className="expanded-label">Expected Return</span>
                <span className={`expanded-value ${colorClass(retVal)}`}>{fmtPctSigned(retVal)}</span>
              </div>
              <div className="expanded-stat">
                <span className="expanded-label">Signal</span>
                <SignalBadge signal={item.signal} />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

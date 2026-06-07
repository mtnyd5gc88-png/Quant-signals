import { useState, useMemo } from 'react';
import { SignalBadge } from '../components/SignalBadge';
import { ProbBar } from '../components/ProbBar';
import { ConfidenceDots } from '../components/ConfidenceDots';
import { Sparkline } from '../components/Sparkline';
import { useSignals } from '../api/hooks';
import { fmtPctSigned, fmtPrice, fmtRelTime, colorClass } from '../utils/format';
import type { SignalItem } from '../api/types';
import './Signals.css';

const FILTERS = ['ALL', 'BUY', 'HOLD', 'CASH'] as const;

type SortKey = 'prob_up' | 'target_return' | 'ticker' | 'model_confidence' | 'position_weight';

export function Signals() {
  const [filter, setFilter] = useState('ALL');
  const [sortKey, setSortKey] = useState<SortKey>('prob_up');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data, loading } = useSignals();

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  }

  const displayRows = useMemo<SignalItem[]>(() => {
    let rows = [...data.items];

    if (filter !== 'ALL') {
      rows = rows.filter((r) => r.signal === filter);
    }

    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter(
        (r) =>
          r.ticker.toLowerCase().includes(q) ||
          (r.company ?? '').toLowerCase().includes(q),
      );
    }

    rows.sort((a, b) => {
      const av = a[sortKey] as number | string | undefined;
      const bv = b[sortKey] as number | string | undefined;
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'desc' ? bv - av : av - bv;
      }
      const as = String(av ?? '');
      const bs = String(bv ?? '');
      return sortDir === 'desc'
        ? bs.localeCompare(as)
        : as.localeCompare(bs);
    });

    return rows;
  }, [data.items, filter, search, sortKey, sortDir]);

  function SortTH({
    label,
    col,
    width,
    right,
    center,
  }: {
    label: string;
    col: SortKey;
    width: number;
    right?: boolean;
    center?: boolean;
  }) {
    const active = sortKey === col;
    return (
      <th
        style={{ width }}
        className={`sortable${active ? ' sorted' : ''}${right ? ' numeric' : ''}${center ? ' th-center' : ''}`}
        onClick={() => handleSort(col)}
      >
        {label}
        {active ? (
          <span className="sort-icon">{sortDir === 'desc' ? '▼' : '▲'}</span>
        ) : (
          <span className="sort-icon sort-icon-inactive">↕</span>
        )}
      </th>
    );
  }

  const shown = displayRows.length;
  const buyCount = displayRows.filter((r) => r.signal === 'BUY').length;
  const holdCount = displayRows.filter((r) => r.signal === 'HOLD').length;
  const cashCount = displayRows.filter((r) => r.signal === 'CASH').length;

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
            placeholder="Filter ticker…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <span className="filter-bar-right">
          {shown} signals&nbsp;·&nbsp;
          <span className="count-buy">{buyCount} BUY</span>
          &nbsp;·&nbsp;
          <span className="count-hold">{holdCount} HOLD</span>
          &nbsp;·&nbsp;
          {cashCount} CASH
        </span>
      </div>

      {/* Table */}
      <div className="table-scroll-wrapper">
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 160 }}>TICKER</th>
                <th style={{ width: 80 }} className="th-center">SIGNAL</th>
                <SortTH label="PROB UP"    col="prob_up"          width={130} right />
                <SortTH label="EXP RETURN" col="target_return"    width={100} right />
                <th style={{ width: 96 }} className="numeric">TARGET PRICE</th>
                <th style={{ width: 96 }} className="numeric">CURRENT PRICE</th>
                <SortTH label="POS WT"     col="position_weight"  width={72}  right />
                <th style={{ width: 110 }}>SECTOR</th>
                <SortTH label="CONFIDENCE" col="model_confidence" width={120} right />
                <th style={{ width: 80 }} className="numeric">LAST PRED</th>
                <th style={{ width: 76 }} className="th-center">TREND</th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 15 }, (_, i) => (
                    <tr key={i}>
                      {Array.from({ length: 11 }, (__, j) => (
                        <td key={j}>
                          <div className="skeleton skeleton-text" />
                        </td>
                      ))}
                    </tr>
                  ))
                : displayRows.map((item) => (
                    <SignalRow
                      key={item.ticker}
                      item={item}
                      expanded={expanded === item.ticker}
                      onToggle={() =>
                        setExpanded(expanded === item.ticker ? null : item.ticker)
                      }
                    />
                  ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SignalRow({
  item,
  expanded,
  onToggle,
}: {
  item: SignalItem;
  expanded: boolean;
  onToggle: () => void;
}) {
  const retVal = item.target_return ?? item.expected_return ?? 0;
  const retClass = colorClass(retVal);

  return (
    <>
      <tr
        className={`signal-row${expanded ? ' expanded' : ''}`}
        onClick={onToggle}
      >
        <td>
          <div className="ticker-cell">
            <div className="company-logo-circle">{item.ticker.slice(0, 2)}</div>
            <div>
              <div className="ticker-symbol">{item.ticker}</div>
              {item.company && (
                <div className="company-name">{item.company}</div>
              )}
            </div>
          </div>
        </td>
        <td className="td-center">
          <SignalBadge signal={item.signal} />
        </td>
        <td>
          <ProbBar value={item.prob_up} />
        </td>
        <td className={`cell-numeric ${retClass}`}>{fmtPctSigned(retVal)}</td>
        <td className="cell-numeric">
          {item.target_price ? `$${fmtPrice(item.target_price)}` : '—'}
        </td>
        <td className="cell-numeric">${fmtPrice(item.price)}</td>
        <td className="cell-numeric">
          {item.position_weight
            ? `${(item.position_weight * 100).toFixed(2)}%`
            : '—'}
        </td>
        <td className="cell-sector">{item.sector ?? '—'}</td>
        <td>
          <ConfidenceDots value={item.model_confidence ?? 0.5} />
        </td>
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
                <span className="expanded-value">
                  {item.alpha_score?.toFixed(4) ?? '—'}
                </span>
              </div>
              <div className="expanded-stat">
                <span className="expanded-label">Probability Up</span>
                <span className="expanded-value">
                  {(item.prob_up * 100).toFixed(2)}%
                </span>
              </div>
              <div className="expanded-stat">
                <span className="expanded-label">Model Confidence</span>
                <span className="expanded-value">
                  {((item.model_confidence ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="expanded-stat">
                <span className="expanded-label">Expected Return</span>
                <span className={`expanded-value ${colorClass(retVal)}`}>
                  {fmtPctSigned(retVal)}
                </span>
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

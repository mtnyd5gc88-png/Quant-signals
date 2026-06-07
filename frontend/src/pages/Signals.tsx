import { useState, useMemo, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Search } from 'lucide-react';
import { SignalBadge } from '../components/SignalBadge';
import { ProbBar } from '../components/ProbBar';
import { useSignals, useDiagnostics } from '../api/hooks';
import { fmtPctSigned, fmtPrice, colorClass } from '../utils/format';
import type { SignalItem } from '../api/types';
import './Signals.css';

const FILTERS = ['ALL', 'BUY', 'HOLD', 'SELL'] as const;
type Filter = typeof FILTERS[number];

type SortKey = 'prob_up' | 'target_return' | 'ticker' | 'price' | 'signal';

export function Signals() {
  const [searchParams] = useSearchParams();
  const [filter, setFilter] = useState<Filter>('ALL');
  const [sortKey, setSortKey] = useState<SortKey>('prob_up');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [search, setSearch] = useState(() => searchParams.get('q') ?? '');
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data, loading } = useSignals();
  const { data: diag } = useDiagnostics();
  const rocAuc = diag.model_quality.roc_auc_mean;

  useEffect(() => {
    const q = searchParams.get('q');
    if (!q || !data.items.length) return;
    const exact = data.items.find((i) => i.ticker === q.toUpperCase());
    if (exact) setExpanded(exact.ticker);
  }, [data.items, searchParams]);

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
      return sortDir === 'desc' ? bs.localeCompare(as) : as.localeCompare(bs);
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

  const isEmpty = !loading && shown === 0;

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
              {f === 'ALL' ? 'All' : `${f} only`}
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

        {!loading && (
          <span className="filter-bar-right">
            {shown} signals&nbsp;·&nbsp;
            <span className="count-buy">{buyCount} BUY</span>
            &nbsp;·&nbsp;
            <span className="count-hold">{holdCount} HOLD</span>
          </span>
        )}
      </div>

      {/* Table Area */}
      <div className="table-scroll-wrapper">
        <div className="data-table-wrapper">
          {loading ? (
            <div className="table-skeleton-wrapper">
              {Array.from({ length: 8 }, (_, i) => (
                <div key={i} className="table-skeleton-row">
                  <div className="skel" style={{ width: 160 }} />
                  <div className="skel" style={{ width: 64 }} />
                  <div className="skel" style={{ width: 128 }} />
                  <div className="skel" style={{ width: 72 }} />
                  <div className="skel" style={{ width: 72 }} />
                  <div className="skel" style={{ width: 48 }} />
                </div>
              ))}
            </div>
          ) : isEmpty ? (
            <div className="table-empty">
              <Search size={48} color="#9ca3af" strokeWidth={1.5} />
              <p className="table-empty-text">
                No signals found{search.trim() ? ` for "${search.trim()}"` : ''}
              </p>
              <p className="table-empty-sub">Try a different ticker symbol</p>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <SortTH label="TICKER"      col="ticker"        width={200} />
                  <SortTH label="SIGNAL"      col="signal"        width={90}  center />
                  <SortTH label="PROBABILITY" col="prob_up"       width={180} />
                  <SortTH label="EST. RETURN" col="target_return" width={110} right />
                  <SortTH label="PRICE"       col="price"         width={100} right />
                  <th style={{ width: 80 }} className="th-center">ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {displayRows.map((item) => (
                  <SignalRow
                    key={item.ticker}
                    item={item}
                    expanded={expanded === item.ticker}
                    onToggle={() =>
                      setExpanded(expanded === item.ticker ? null : item.ticker)
                    }
                    rocAuc={rocAuc}
                  />
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

function SignalRow({
  item,
  expanded,
  onToggle,
  rocAuc,
}: {
  item: SignalItem;
  expanded: boolean;
  onToggle: () => void;
  rocAuc?: number;
}) {
  const retVal = item.target_return ?? item.expected_return ?? 0;
  const retClass = colorClass(retVal);

  return (
    <>
      <tr className={`signal-row${expanded ? ' expanded' : ''}`}>
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
          <ProbBar value={item.prob_up} rocAuc={rocAuc} avgTrust={rocAuc !== undefined} />
        </td>
        <td className={`cell-numeric ${retClass}`}>{fmtPctSigned(retVal)}</td>
        <td className="cell-numeric">${fmtPrice(item.price)}</td>
        <td className="td-center">
          <button className="detail-btn" onClick={onToggle}>
            {expanded ? '▲' : 'Detail'}
          </button>
        </td>
      </tr>
      {expanded && (
        <tr className="expanded-detail-row">
          <td colSpan={6}>
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
                <span className="expanded-label">Target Price</span>
                <span className="expanded-value">
                  {item.target_price ? `$${fmtPrice(item.target_price)}` : '—'}
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

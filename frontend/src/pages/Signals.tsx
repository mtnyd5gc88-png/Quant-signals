import { useState, useMemo, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Activity } from 'lucide-react';
import { SignalBadge } from '../components/SignalBadge';
import { ProbBar } from '../components/ProbBar';
import { ConfidenceDots } from '../components/ConfidenceDots';
import { TickerDrawer } from '../components/TickerDrawer';
import { FreshnessTag } from '../components/FreshnessTag';
import { SectionError } from '../components/SectionError';
import { EmptyState } from '../components/EmptyState';
import { useSignals, useDiagnostics, usePortfolio, useRegime } from '../api/hooks';
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
  const [drawerItem, setDrawerItem] = useState<SignalItem | null>(null);

  const { data, loading, error, refetch: signalsRefetch, fetchedAt } = useSignals();
  const { data: diag } = useDiagnostics();
  const { data: portfolio } = usePortfolio();
  const { data: regimeData } = useRegime();
  const rocAuc = diag.model_quality.roc_auc_mean;
  const regime = regimeData.regime;

  useEffect(() => {
    const q = searchParams.get('q');
    if (!q || !data.items.length) return;
    const exact = data.items.find((i) => i.ticker === q.toUpperCase());
    if (exact) setDrawerItem(exact);
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

  const isEmpty = !loading && !error && shown === 0;

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
            placeholder="Ticker…"
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
            &nbsp;·&nbsp;
            <FreshnessTag lastUpdated={fetchedAt} />
          </span>
        )}
      </div>

      {/* Desktop table */}
      <div className="signals-table-desktop">
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
            ) : error ? (
              <div style={{ padding: 20 }}>
                <SectionError message="Failed to load signals." onRetry={signalsRefetch} />
              </div>
            ) : isEmpty ? (
              <EmptyState
                icon={<Activity size={40} strokeWidth={1.25} />}
                message="No signals available. Run the quant engine to generate signals."
                sub={search.trim() ? `No results for "${search.trim()}"` : undefined}
              />
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
                      onOpen={() => setDrawerItem(item)}
                      rocAuc={rocAuc}
                    />
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {/* Mobile card list */}
      <div className="signals-mobile-list">
        {displayRows.map((item) => {
          const retVal = item.target_return ?? item.expected_return ?? 0;
          return (
            <div
              key={item.ticker}
              className="signal-card-mobile"
              onClick={() => setDrawerItem(item)}
            >
              <div className="scm-row-1">
                <div className="scm-left">
                  <span className="scm-avatar">{item.ticker.slice(0, 2)}</span>
                  <span className="scm-ticker">{item.ticker}</span>
                  <SignalBadge signal={item.signal} />
                </div>
                <div className="scm-right">
                  <ProbBar value={item.prob_up} />
                </div>
              </div>
              <div className="scm-row-2">
                <span className="scm-trust">
                  Trust <ConfidenceDots value={Math.abs(item.prob_up - 0.5) * 2} />
                </span>
                <span className={`scm-return ${retVal > 0 ? 'pos' : 'neg'}`}>
                  {retVal > 0 ? '+' : ''}{(retVal * 100).toFixed(2)}%
                </span>
                <span className="scm-price">${item.price?.toFixed(2) ?? '—'}</span>
              </div>
            </div>
          );
        })}
      </div>
      <TickerDrawer
        item={drawerItem}
        onClose={() => setDrawerItem(null)}
        rocAuc={rocAuc ?? 0}
        regime={regime}
        portfolioHolding={
          drawerItem
            ? portfolio.holdings.find((h) => h.ticker === drawerItem.ticker)
            : undefined
        }
      />
    </div>
  );
}

function SignalRow({
  item,
  onOpen,
  rocAuc,
}: {
  item: SignalItem;
  onOpen: () => void;
  rocAuc?: number;
}) {
  const retVal = item.target_return ?? item.expected_return ?? 0;
  const retClass = colorClass(retVal);

  return (
    <tr className="signal-row" onClick={onOpen} style={{ cursor: 'pointer' }}>
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
        <button className="detail-btn" onClick={(e) => { e.stopPropagation(); onOpen(); }}>
          Detail
        </button>
      </td>
    </tr>
  );
}

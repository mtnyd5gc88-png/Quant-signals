from __future__ import annotations

import base64
import os
from pathlib import Path

# Sandboxed environments may have unwritable default MPL config path
_default_cache = Path(__file__).resolve().parent / "outputs" / "mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_default_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Static chart generators (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str) -> None:
    """Apply a clean dark-ish matplotlib style to an axis."""
    ax.set_facecolor("#0f1620")
    ax.grid(True, color="#1e2d42", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.tick_params(colors="#8a9bb0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d42")
    ax.title.set_color("#cdd9e8")
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_color("#8a9bb0")
    ax.yaxis.label.set_color("#8a9bb0")
    ax.set_title(title)


def plot_equity_curve(equity: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#090d14")
    ax.plot(equity.index, equity.values, color="#00d4a0", linewidth=1.5, label="Strategy")
    ax.fill_between(equity.index, equity.values, equity.iloc[0], alpha=0.12, color="#00d4a0")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend(facecolor="#0f1620", edgecolor="#1e2d42", labelcolor="#cdd9e8", fontsize=9)
    _style_ax(ax, "Portfolio Equity Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_strategy_vs_benchmark(equity: pd.Series, benchmark: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#090d14")
    strat_norm = equity.values / equity.iloc[0]
    bench_norm = benchmark.values / benchmark.iloc[0]
    ax.plot(equity.index, strat_norm, color="#00d4a0", linewidth=1.5, label="Strategy")
    ax.plot(benchmark.index, bench_norm, color="#6c8ebf", linewidth=1.5,
            linestyle="--", label="SPY Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(facecolor="#0f1620", edgecolor="#1e2d42", labelcolor="#cdd9e8", fontsize=9)
    _style_ax(ax, "Strategy vs SPY (Normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_drawdown(drawdown: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#090d14")
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#ef4444", alpha=0.35)
    ax.plot(drawdown.index, drawdown.values, color="#ef4444", linewidth=1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    _style_ax(ax, "Drawdown")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_feature_importance(
    importances: pd.Series,
    out_path: Path,
    title: str = "Random Forest Feature Importance",
    top_k: int = 15,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imp = importances.sort_values(ascending=False).head(top_k)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#090d14")
    bars = ax.barh(imp.index, imp.values, color="#00d4a0", alpha=0.8)
    ax.set_xlabel("Importance")
    _style_ax(ax, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML Dashboard generator
# ─────────────────────────────────────────────────────────────────────────────

def _embed_png(path: Path) -> str:
    """Return a base64 data URI for a PNG, or empty string if missing."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"


def generate_html_dashboard(
    predictions: list[dict],
    report,                         # PerformanceReport dataclass
    plots_dir: Path,
    out_path: Path,
    regime: str | None = None,
    run_timestamp: str | None = None,
    n_tickers_trained: int = 0,
    backtest_start: str = "N/A",
    backtest_end: str = "N/A",
) -> None:
    """
    Generate a self-contained HTML dashboard.
    All chart images are embedded as base64 — no external file deps.
    """

    # ── embed charts
    img_vs_spy  = _embed_png(plots_dir / "strategy_vs_spy.png")
    img_dd      = _embed_png(plots_dir / "drawdown.png")
    img_equity  = _embed_png(plots_dir / "equity_curve.png")
    img_imp     = _embed_png(plots_dir / "feature_importance_random_forest.png")

    # ── metrics formatting helpers
    def fmt_pct(v: float) -> str:
        return f"{v:+.2%}" if v == v else "N/A"  # NaN guard

    def fmt_f(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}" if v == v else "N/A"

    def color_cls(v: float, flip: bool = False) -> str:
        if v != v:
            return "neutral"
        pos = v >= 0
        if flip:
            pos = not pos
        return "pos" if pos else "neg"

    # ── signal table rows
    rows_html: list[str] = []
    buy_count = hold_count = stay_count = 0

    for p in predictions:
        sig = p["signal"]
        prob = p["prob_up"]
        price = p.get("price")
        ret = p.get("target_return")

        if sig == "BUY":
            buy_count += 1
            sig_cls = "buy"
        elif sig == "HOLD":
            hold_count += 1
            sig_cls = "hold"
        else:
            stay_count += 1
            sig_cls = "stay"

        price_str = f"${price:.2f}" if price is not None else "—"
        if ret is not None and price is not None:
            tp = price * (1 + ret)
            tp_str = f"${tp:.2f}"
            ret_str = f"{ret*100:+.2f}%"
            ret_color = "pos" if ret > 0 else "neg"
        else:
            tp_str = ret_str = "—"
            ret_color = "neutral"

        rows_html.append(f"""
        <tr class="{sig_cls}-row" data-signal="{sig}">
          <td class="td-ticker">{p['ticker']}</td>
          <td><span class="badge {sig_cls}">{sig}</span></td>
          <td class="td-num">{prob:.1%}</td>
          <td class="td-num">{price_str}</td>
          <td class="td-num">{tp_str}</td>
          <td class="td-num {ret_color}">{ret_str}</td>
        </tr>""")

    rows_joined = "\n".join(rows_html)

    regime_label = regime or "UNKNOWN"
    regime_cls = {
        "bull": "regime-bull",
        "neutral": "regime-neutral",
        "risk_off": "regime-risk-off",
    }.get(regime or "", "regime-neutral")

    timestamp = run_timestamp or "N/A"

    # ── chart blocks
    def chart_block(label: str, img_src: str, wide: bool = False) -> str:
        if not img_src:
            return ""
        extra = ' style="grid-column: 1 / -1;"' if wide else ""
        return f"""
        <div class="chart-card"{extra}>
          <div class="chart-header">{label}</div>
          <img src="{img_src}" alt="{label}" loading="lazy" />
        </div>"""

    importance_block = chart_block(
        "Random Forest — Feature Importance", img_imp, wide=True
    ) if img_imp else ""

    # ── full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QUANT/SYS — Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet" />
  <style>
    /* ── Reset & tokens */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:         #090d14;
      --surface:    #0f1620;
      --surface2:   #0a1422;
      --border:     #1a2840;
      --border2:    rgba(26,40,64,.5);
      --accent:     #00d4a0;
      --accent-dim: rgba(0,212,160,.12);
      --warn:       #f59e0b;
      --warn-dim:   rgba(245,158,11,.12);
      --danger:     #ef4444;
      --danger-dim: rgba(239,68,68,.12);
      --hold-c:     #6c8ebf;
      --text:       #cdd9e8;
      --text-dim:   #5a6e85;
      --mono:       'IBM Plex Mono', monospace;
      --sans:       'IBM Plex Sans', sans-serif;
    }}

    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.6;
      min-height: 100vh;
    }}

    /* ── Header */
    header {{
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 16px 32px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      flex-wrap: wrap;
    }}

    .logo {{
      font-size: 17px;
      font-weight: 600;
      letter-spacing: 3px;
      color: var(--accent);
      margin-right: auto;
    }}
    .logo span {{ color: var(--text-dim); font-weight: 300; }}

    .header-meta {{
      font-size: 11px;
      color: var(--text-dim);
      letter-spacing: .5px;
    }}

    .regime-badge {{
      padding: 4px 12px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }}
    .regime-bull     {{ background: var(--accent-dim); color: var(--accent); border: 1px solid var(--accent); }}
    .regime-neutral  {{ background: var(--warn-dim);   color: var(--warn);   border: 1px solid var(--warn);   }}
    .regime-risk-off {{ background: var(--danger-dim); color: var(--danger); border: 1px solid var(--danger); }}

    /* ── Main */
    main {{ padding: 24px 32px; max-width: 1600px; margin: 0 auto; }}

    /* ── Section title */
    .section-label {{
      font-size: 10px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--text-dim);
      margin: 28px 0 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .section-label::before {{
      content: '';
      display: inline-block;
      width: 3px;
      height: 12px;
      background: var(--accent);
      border-radius: 2px;
    }}

    /* ── Metric cards */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 4px;
    }}

    .metric-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px 18px;
      transition: border-color .2s;
    }}
    .metric-card:hover {{ border-color: var(--accent); }}

    .metric-label {{
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 6px;
    }}

    .metric-val {{
      font-size: 22px;
      font-weight: 600;
      letter-spacing: -.5px;
    }}
    .metric-val.pos     {{ color: var(--accent); }}
    .metric-val.neg     {{ color: var(--danger); }}
    .metric-val.neutral {{ color: var(--text); }}

    .metric-sub {{ font-size: 11px; color: var(--text-dim); margin-top: 2px; }}

    /* ── Summary strip */
    .summary-strip {{
      display: flex;
      gap: 6px;
      margin-bottom: 4px;
      flex-wrap: wrap;
    }}
    .strip-pill {{
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: .5px;
    }}
    .pill-buy  {{ background: var(--accent-dim); color: var(--accent); }}
    .pill-hold {{ background: var(--warn-dim);   color: var(--warn);   }}
    .pill-stay {{ background: var(--danger-dim); color: var(--danger); }}
    .pill-info {{ background: rgba(108,142,191,.12); color: var(--hold-c); }}

    /* ── Charts grid */
    .charts-grid {{
      display: grid;
      grid-template-columns: 3fr 2fr;
      gap: 12px;
    }}
    @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}

    .chart-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }}

    .chart-header {{
      padding: 9px 14px;
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-dim);
      border-bottom: 1px solid var(--border);
    }}

    .chart-card img {{
      width: 100%;
      display: block;
    }}

    /* ── Filter bar */
    .filter-bar {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}

    .filter-btn {{
      padding: 5px 14px;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text-dim);
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 1px;
      cursor: pointer;
      text-transform: uppercase;
      transition: border-color .15s, color .15s;
    }}
    .filter-btn:hover,
    .filter-btn.active {{ border-color: var(--accent); color: var(--accent); }}
    .filter-btn[data-f="BUY"].active     {{ border-color: var(--accent); color: var(--accent); }}
    .filter-btn[data-f="HOLD"].active    {{ border-color: var(--warn);   color: var(--warn);   }}
    .filter-btn[data-f="STAY IN CASH"].active {{ border-color: var(--danger); color: var(--danger); }}

    /* ── Table */
    .table-wrap {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      overflow-x: auto;
    }}

    table {{ width: 100%; border-collapse: collapse; }}

    thead {{ background: var(--surface2); }}

    th {{
      padding: 9px 14px;
      text-align: left;
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-dim);
      font-weight: 400;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
    }}
    th:hover {{ color: var(--accent); }}
    th.td-num, td.td-num {{ text-align: right; }}

    td {{
      padding: 8px 14px;
      border-bottom: 1px solid var(--border2);
      font-size: 12px;
      white-space: nowrap;
    }}

    tr:last-child td {{ border-bottom: none; }}

    tbody tr {{ transition: background .1s; }}
    tbody tr:hover td {{ background: rgba(255,255,255,.025); }}

    .td-ticker {{
      font-weight: 600;
      letter-spacing: .5px;
      color: var(--text);
    }}

    .badge {{
      font-weight: 600;
      font-size: 10px;
      letter-spacing: 1px;
      padding: 2px 8px;
      border-radius: 3px;
      text-transform: uppercase;
      display: inline-block;
    }}
    .badge.buy  {{ color: var(--accent); background: var(--accent-dim); }}
    .badge.hold {{ color: var(--warn);   background: var(--warn-dim);   }}
    .badge.stay {{ color: var(--danger); background: var(--danger-dim); }}

    .buy-row  td:first-child {{ border-left: 2px solid var(--accent); }}
    .hold-row td:first-child {{ border-left: 2px solid var(--warn);   }}
    .stay-row td:first-child {{ border-left: 2px solid transparent;   }}

    td.pos     {{ color: var(--accent); }}
    td.neg     {{ color: var(--danger); }}
    td.neutral {{ color: var(--text);   }}

    /* ── Footer */
    footer {{
      text-align: center;
      padding: 20px;
      color: var(--text-dim);
      font-size: 11px;
      border-top: 1px solid var(--border);
      margin-top: 40px;
      letter-spacing: .5px;
    }}

    /* ── Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  </style>
</head>
<body>

<!-- ── Header ──────────────────────────────────────────────────────────────── -->
<header>
  <div class="logo">QUANT<span>/SYS</span></div>
  <div class="header-meta">LAST RUN: {timestamp}</div>
  <div class="header-meta">BACKTEST: {backtest_start} → {backtest_end}</div>
  <div class="header-meta">{n_tickers_trained} TICKERS TRAINED</div>
  <div class="regime-badge {regime_cls}">{regime_label}</div>
</header>

<main>

  <!-- ── Performance Metrics ────────────────────────────────────────────────── -->
  <div class="section-label">Performance Metrics</div>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Total Return</div>
      <div class="metric-val {color_cls(report.cumulative_return)}">{fmt_pct(report.cumulative_return)}</div>
      <div class="metric-sub">since inception</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Ann. Return</div>
      <div class="metric-val {color_cls(report.annualized_return)}">{fmt_pct(report.annualized_return)}</div>
      <div class="metric-sub">annualised CAGR</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sharpe Ratio</div>
      <div class="metric-val {color_cls(report.sharpe_ratio)}">{fmt_f(report.sharpe_ratio)}</div>
      <div class="metric-sub">daily, ann. √252</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Max Drawdown</div>
      <div class="metric-val neg">{fmt_pct(report.max_drawdown)}</div>
      <div class="metric-sub">peak-to-trough</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">SPY Buy & Hold</div>
      <div class="metric-val {color_cls(report.buy_and_hold_return)}">{fmt_pct(report.buy_and_hold_return)}</div>
      <div class="metric-sub">benchmark</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Win Rate</div>
      <div class="metric-val neutral">{fmt_pct(report.win_rate)}</div>
      <div class="metric-sub">{report.number_of_trades} trades</div>
    </div>
  </div>

  <!-- ── Charts ─────────────────────────────────────────────────────────────── -->
  <div class="section-label">Charts</div>
  <div class="charts-grid">
    {chart_block("Strategy vs SPY (Normalised)", img_vs_spy)}
    {chart_block("Drawdown", img_dd)}
    {chart_block("Equity Curve", img_equity, wide=True)}
  </div>

  <!-- ── Signal Output ──────────────────────────────────────────────────────── -->
  <div class="section-label">Signal Output</div>

  <div class="summary-strip">
    <div class="strip-pill pill-buy">BUY: {buy_count}</div>
    <div class="strip-pill pill-hold">HOLD: {hold_count}</div>
    <div class="strip-pill pill-stay">STAY IN CASH: {stay_count}</div>
    <div class="strip-pill pill-info">TOTAL: {len(predictions)}</div>
  </div>

  <div class="filter-bar">
    <button class="filter-btn active" data-f="ALL" onclick="filterSignals(this)">All</button>
    <button class="filter-btn" data-f="BUY" onclick="filterSignals(this)">BUY only</button>
    <button class="filter-btn" data-f="HOLD" onclick="filterSignals(this)">HOLD only</button>
    <button class="filter-btn" data-f="STAY IN CASH" onclick="filterSignals(this)">Stay only</button>
  </div>

  <div class="table-wrap">
    <table id="signals-table">
      <thead>
        <tr>
          <th onclick="sortTable(0)">Ticker ↕</th>
          <th onclick="sortTable(1)">Signal ↕</th>
          <th class="td-num" onclick="sortTable(2)">P(Up) ↕</th>
          <th class="td-num" onclick="sortTable(3)">Price ↕</th>
          <th class="td-num" onclick="sortTable(4)">5d Target ↕</th>
          <th class="td-num" onclick="sortTable(5)">Exp. Return ↕</th>
        </tr>
      </thead>
      <tbody>
        {rows_joined}
      </tbody>
    </table>
  </div>

  <!-- ── Feature Importance ─────────────────────────────────────────────────── -->
  {f'<div class="section-label">Feature Importance</div><div class="charts-grid">{importance_block}</div>' if importance_block else ''}

</main>

<footer>
  QUANT/SYS — ML-based quantitative trading system
  — Walk-forward out-of-sample signals — No lookahead bias
</footer>

<script>
  // ── Signal filter
  function filterSignals(btn) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.f;
    document.querySelectorAll('#signals-table tbody tr').forEach(row => {{
      const sig = row.dataset.signal;
      row.style.display = (filter === 'ALL' || sig === filter) ? '' : 'none';
    }});
  }}

  // ── Column sort
  let _sortDir = {{}};
  function sortTable(col) {{
    const tbody = document.querySelector('#signals-table tbody');
    const rows = [...tbody.querySelectorAll('tr')];
    _sortDir[col] = !_sortDir[col];
    const dir = _sortDir[col] ? 1 : -1;

    rows.sort((a, b) => {{
      const aText = a.querySelectorAll('td')[col]?.textContent.trim() ?? '';
      const bText = b.querySelectorAll('td')[col]?.textContent.trim() ?? '';

      // Try numeric parse (strip $, %, +)
      const aNum = parseFloat(aText.replace(/[$%+,]/g, ''));
      const bNum = parseFloat(bText.replace(/[$%+,]/g, ''));
      if (!isNaN(aNum) && !isNaN(bNum)) return dir * (aNum - bNum);
      return dir * aText.localeCompare(bText);
    }});

    rows.forEach(r => tbody.appendChild(r));
  }}
</script>

</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard: {out_path}")

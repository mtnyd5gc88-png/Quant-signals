import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import type { SignalHistoryPoint } from '../api/types';
import './ProbHistoryChart.css';

interface Props {
  history: SignalHistoryPoint[];
  currentSignal: string;
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function signalColor(signal: string): string {
  if (signal === 'BUY') return 'var(--positive)';
  if (signal === 'SELL' || signal === 'CASH' || signal === 'STAY IN CASH') return 'var(--negative)';
  return 'var(--text-tertiary)';
}

interface TooltipPayload {
  payload?: { run_at: string; prob_up: number; signal: string };
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  if (!d) return null;
  return (
    <div className="prob-chart-tooltip">
      <div className="prob-chart-tt-date">{fmtDate(d.run_at)}</div>
      <div className="prob-chart-tt-prob" style={{ color: signalColor(d.signal) }}>
        {Math.round(d.prob_up * 100)}%
      </div>
      <div className="prob-chart-tt-signal">{d.signal}</div>
    </div>
  );
}

function CustomDot(props: Record<string, unknown>) {
  const { cx, cy, payload } = props as { cx: number; cy: number; payload: SignalHistoryPoint };
  const color = signalColor(payload.signal);
  return <circle cx={cx} cy={cy} r={3} fill={color} stroke="var(--bg-surface-1)" strokeWidth={1.5} />;
}

export function ProbHistoryChart({ history, currentSignal }: Props) {
  if (history.length < 2) {
    return (
      <div className="prob-chart-empty">
        <span>No signal history available</span>
        <span className="prob-chart-empty-sub">Requires database connection with multiple runs</span>
      </div>
    );
  }

  // Use last 15 points at most
  const points = history.slice(-15);
  const strokeColor = signalColor(currentSignal);
  const gradientId = `probFill_${currentSignal.replace(/\s+/g, '')}`;

  return (
    <div className="prob-history-chart">
      <div style={{ width: '100%', minWidth: 0, height: 120 }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={points} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={strokeColor} stopOpacity={0.18} />
              <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="1 3"
            stroke="var(--chart-grid)"
            strokeOpacity={0.4}
            vertical={false}
          />
          <ReferenceLine
            y={0.65}
            stroke="var(--positive)"
            strokeDasharray="3 3"
            strokeOpacity={0.4}
            strokeWidth={1}
          />
          <ReferenceLine
            y={0.55}
            stroke="var(--warn)"
            strokeDasharray="3 3"
            strokeOpacity={0.4}
            strokeWidth={1}
          />
          <XAxis
            dataKey="run_at"
            tick={{ fill: 'var(--text-tertiary)', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={fmtDate}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0.3, 1.0]}
            tick={{ fill: 'var(--text-tertiary)', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            width={28}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <Tooltip
            content={<ChartTooltip />}
            cursor={{ stroke: 'var(--border-default)', strokeWidth: 1 }}
          />
          <Area
            type="monotone"
            dataKey="prob_up"
            stroke={strokeColor}
            strokeWidth={1.5}
            fill={`url(#${gradientId})`}
            dot={<CustomDot />}
            activeDot={{ r: 4, fill: strokeColor, stroke: 'var(--bg-surface-1)', strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
      </div>
      <div className="prob-chart-legend">
        <span className="prob-chart-ref buy">— 65% strong</span>
        <span className="prob-chart-ref hold">— 55% threshold</span>
        <span className="prob-chart-count">{history.length} runs</span>
      </div>
    </div>
  );
}

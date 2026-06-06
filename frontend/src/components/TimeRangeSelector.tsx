import './TimeRangeSelector.css';

const RANGES = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', 'ALL'] as const;
export type TimeRange = typeof RANGES[number];

interface Props {
  value: TimeRange;
  onChange: (v: TimeRange) => void;
}

export function TimeRangeSelector({ value, onChange }: Props) {
  return (
    <div className="time-range-selector">
      {RANGES.map((r) => (
        <button
          key={r}
          className={`time-range-btn${value === r ? ' active' : ''}`}
          onClick={() => onChange(r)}
        >
          {r}
        </button>
      ))}
    </div>
  );
}

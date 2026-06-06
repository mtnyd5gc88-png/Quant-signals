interface Props {
  data: number[];
  width?: number;
  height?: number;
}

export function Sparkline({ data, width = 64, height = 24 }: Props) {
  if (!data || data.length < 2) return <span style={{ width, height, display: 'inline-block' }} />;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 2;
  const w = width - pad * 2;
  const h = height - pad * 2;

  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * w;
    const y = pad + (1 - (v - min) / range) * h;
    return `${x},${y}`;
  });

  const last = data[data.length - 1];
  const first = data[0];
  const color = last > first ? 'var(--positive)' : last < first ? 'var(--negative)' : 'var(--chart-1)';

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

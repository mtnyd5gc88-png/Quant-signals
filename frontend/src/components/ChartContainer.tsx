import { useState, useEffect, type ReactNode } from 'react';

interface Props {
  height: number;
  children: ReactNode;
}

export function ChartContainer({ height, children }: Props) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setReady(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
    <div style={{ width: '100%', minWidth: 0, height }}>
      {ready ? children : null}
    </div>
  );
}

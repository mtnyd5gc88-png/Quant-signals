import { AlertTriangle } from 'lucide-react';
import './FreshnessTag.css';

function formatFreshness(iso: string): string {
  const now = Date.now();
  const then = new Date(iso).getTime();
  const diffMs = now - then;
  const diffMin = Math.floor(diffMs / 60_000);

  if (diffMin < 1) return 'Updated just now';
  if (diffMin < 60) return `Updated ${diffMin} minute${diffMin === 1 ? '' : 's'} ago`;

  const d = new Date(iso);
  const today = new Date();
  if (
    d.getFullYear() === today.getFullYear() &&
    d.getMonth() === today.getMonth() &&
    d.getDate() === today.getDate()
  ) {
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return `Updated today at ${hh}:${mm}`;
  }

  const diffH = Math.floor(diffMin / 60);
  if (diffH < 48) return `Updated ${diffH}h ago`;
  return `Updated ${Math.floor(diffH / 24)}d ago`;
}

export function FreshnessTag({ lastUpdated }: { lastUpdated?: string }) {
  if (!lastUpdated) return null;
  const stale = Date.now() - new Date(lastUpdated).getTime() > 6 * 3_600_000;
  return (
    <span className={`freshness-tag${stale ? ' freshness-tag--stale' : ''}`}>
      {stale && <AlertTriangle size={10} strokeWidth={2.5} />}
      {formatFreshness(lastUpdated)}
    </span>
  );
}

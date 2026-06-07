export function fmtPct(v: number, decimals = 2): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function fmtPctSigned(v: number, decimals = 2): string {
  const s = (v * 100).toFixed(decimals);
  return v >= 0 ? `+${s}%` : `${s}%`;
}

export function fmtNum(v: number, decimals = 3): string {
  return v.toFixed(decimals);
}

export function fmtPrice(v: number): string {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function fmtRelTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const hrs = diff / 3600000;
  if (hrs < 1) return `${Math.round(diff / 60000)}m ago`;
  if (hrs < 24) return `${Math.round(hrs)}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

export function colorClass(v: number): 'positive' | 'negative' | 'neutral' {
  if (v > 0) return 'positive';
  if (v < 0) return 'negative';
  return 'neutral';
}

export function humanizeError(err: unknown): string {
  if (err instanceof TypeError && /fetch|network|Failed to fetch/i.test(err.message)) {
    return '네트워크 연결을 확인해주세요.';
  }
  if (err instanceof Error) {
    if (/405|Method Not Allowed/i.test(err.message)) {
      return '서버 요청 방식이 맞지 않습니다. 잠시 후 다시 시도해주세요.';
    }
    if (/401|Unauthorized/i.test(err.message)) {
      return '인증이 필요합니다. 다시 로그인해주세요.';
    }
    if (/already_running/i.test(err.message)) {
      return '이미 데이터를 새로고침 중입니다. 잠시 후 확인해주세요.';
    }
  }
  return '데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.';
}

export function filterByRange<T extends { date: string }>(
  data: T[],
  range: string,
): T[] {
  if (!data.length || range === 'ALL') return data;
  const now = new Date();
  const from = new Date(now);
  if (range === '1M') from.setMonth(now.getMonth() - 1);
  else if (range === '3M') from.setMonth(now.getMonth() - 3);
  else if (range === '6M') from.setMonth(now.getMonth() - 6);
  else if (range === 'YTD') from.setMonth(0, 1);
  else if (range === '1Y') from.setFullYear(now.getFullYear() - 1);
  else if (range === '3Y') from.setFullYear(now.getFullYear() - 3);
  return data.filter((d) => new Date(d.date) >= from);
}

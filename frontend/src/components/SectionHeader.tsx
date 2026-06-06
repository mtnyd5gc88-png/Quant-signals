import './SectionHeader.css';

interface Props {
  title: string;
  meta?: string;
  children?: React.ReactNode;
}

export function SectionHeader({ title, meta, children }: Props) {
  return (
    <div className="section-header">
      <span className="section-title">{title}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
        {children}
        {meta && <span className="section-meta">{meta}</span>}
      </div>
    </div>
  );
}

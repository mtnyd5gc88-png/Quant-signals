import './PageTabs.css';

interface Tab {
  id: string;
  label: string;
}

interface Props {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
}

export function PageTabs({ tabs, active, onChange }: Props) {
  return (
    <div className="page-tab-bar">
      {tabs.map((t) => (
        <button
          key={t.id}
          className={`tab-item${active === t.id ? ' active' : ''}`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

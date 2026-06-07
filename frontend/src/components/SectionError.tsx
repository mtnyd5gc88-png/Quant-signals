import { AlertCircle } from 'lucide-react';
import './SectionError.css';

interface Props {
  message: string;
  onRetry?: () => void;
}

export function SectionError({ message, onRetry }: Props) {
  return (
    <div className="section-error" role="alert">
      <AlertCircle size={16} strokeWidth={2} className="section-error__icon" />
      <span className="section-error__message">{message}</span>
      {onRetry && (
        <button className="section-error__retry" onClick={onRetry} type="button">
          Try Again
        </button>
      )}
    </div>
  );
}

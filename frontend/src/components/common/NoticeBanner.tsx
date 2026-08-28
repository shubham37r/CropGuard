import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

interface NoticeBannerProps {
  type?: 'warning' | 'info';
  message?: string;
}

export const NoticeBanner: React.FC<NoticeBannerProps> = ({ type = 'info', message }) => {
  const defaultNote =
    'PROTOTYPE EARLY WARNING SYSTEM: Contextual risk scores and mock analysis results are for prototype demonstration only and are NOT scientifically validated agricultural predictions.';

  return (
    <div className={`border-l-4 p-3 rounded-r text-xs md:text-sm flex items-start gap-2.5 ${
      type === 'warning'
        ? 'bg-amber-50 border-amber-500 text-amber-900'
        : 'bg-emerald-50 border-emerald-600 text-emerald-950'
    }`}>
      {type === 'warning' ? (
        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
      ) : (
        <Info className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
      )}
      <div>
        <span className="font-bold uppercase tracking-wider text-[11px] block text-emerald-800">
          Prototype Notice
        </span>
        <span>{message || defaultNote}</span>
      </div>
    </div>
  );
};

import React from 'react';

interface RiskBadgeProps {
  level?: 'LOW' | 'MEDIUM' | 'HIGH' | string;
  size?: 'sm' | 'md' | 'lg';
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({ level = 'MEDIUM', size = 'md' }) => {
  const lvl = level.toUpperCase();
  
  let colorClasses = 'bg-slate-100 text-slate-700 border-slate-300';
  if (lvl === 'LOW') {
    colorClasses = 'bg-emerald-100 text-emerald-800 border-emerald-300 font-semibold';
  } else if (lvl === 'MEDIUM') {
    colorClasses = 'bg-amber-100 text-amber-800 border-amber-300 font-semibold';
  } else if (lvl === 'HIGH') {
    colorClasses = 'bg-rose-100 text-rose-800 border-rose-300 font-bold animate-pulse';
  }

  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : (size === 'lg' ? 'px-3 py-1 text-base' : 'px-2.5 py-1 text-sm');

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border ${colorClasses} ${sizeClasses}`}>
      <span className={`w-2 h-2 rounded-full ${
        lvl === 'LOW' ? 'bg-emerald-500' : (lvl === 'MEDIUM' ? 'bg-amber-500' : 'bg-rose-600')
      }`} />
      <span>{lvl} RISK</span>
    </span>
  );
};

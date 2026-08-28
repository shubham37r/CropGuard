import React from 'react';

interface StatusBadgeProps {
  status: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  const s = status.toUpperCase();

  let badgeStyle = 'bg-slate-100 text-slate-700 border-slate-300';
  let label = s;

  if (s === 'SUBMITTED') {
    badgeStyle = 'bg-blue-50 text-blue-700 border-blue-200';
    label = 'Submitted';
  } else if (s === 'ANALYZED') {
    badgeStyle = 'bg-cyan-50 text-cyan-800 border-cyan-300';
    label = 'Analyzed';
  } else if (s === 'PENDING_VERIFICATION') {
    badgeStyle = 'bg-amber-100 text-amber-900 border-amber-300 font-medium';
    label = 'Pending Officer Review';
  } else if (s === 'CONFIRMED') {
    badgeStyle = 'bg-emerald-100 text-emerald-800 border-emerald-400 font-bold';
    label = 'Officer Confirmed';
  } else if (s === 'REJECTED') {
    badgeStyle = 'bg-rose-100 text-rose-800 border-rose-300 font-medium';
    label = 'Officer Rejected';
  } else if (s === 'NEEDS_MORE_INFO') {
    badgeStyle = 'bg-purple-100 text-purple-800 border-purple-300 font-medium';
    label = 'Needs More Info';
  }

  return (
    <span className={`inline-block px-2.5 py-0.5 text-xs rounded-full border ${badgeStyle}`}>
      {label}
    </span>
  );
};

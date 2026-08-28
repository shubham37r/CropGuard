import React from 'react';
import { Thermometer, Droplets, CloudRain, ShieldAlert, Info } from 'lucide-react';

interface WeatherContextCardProps {
  district?: string;
  className?: string;
}

export const WeatherContextCard: React.FC<WeatherContextCardProps> = ({ district = 'Nagpur', className = '' }) => {
  return (
    <div className={`bg-gradient-to-br from-slate-900 to-slate-800 text-white p-5 rounded-xl border border-slate-700 shadow-sm space-y-3.5 ${className}`}>
      <div className="flex items-center justify-between border-b border-slate-700/80 pb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-emerald-500/20 text-emerald-400 rounded-md border border-emerald-500/30">
            <ShieldAlert className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Field & Weather Context
            </h4>
            <span className="text-[10px] text-slate-400 block -mt-0.5">
              Regional Environmental Risk Signals ({district} Sector)
            </span>
          </div>
        </div>
        <span className="text-[10px] bg-slate-800 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded font-mono uppercase">
          PROTOTYPE CONTEXT
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
        <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700/60">
          <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
            <Thermometer className="w-3.5 h-3.5 text-amber-400" />
            <span>Temperature</span>
          </div>
          <div className="text-base font-extrabold text-slate-100">28°C</div>
        </div>

        <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700/60">
          <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
            <Droplets className="w-3.5 h-3.5 text-blue-400" />
            <span>Humidity</span>
          </div>
          <div className="text-base font-extrabold text-slate-100">72%</div>
        </div>

        <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700/60">
          <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
            <CloudRain className="w-3.5 h-3.5 text-sky-400" />
            <span>Rainfall</span>
          </div>
          <div className="text-base font-extrabold text-slate-100">Moderate</div>
        </div>

        <div className="bg-slate-800/80 p-2.5 rounded-lg border border-slate-700/60">
          <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
            <ShieldAlert className="w-3.5 h-3.5 text-emerald-400" />
            <span>Env Risk</span>
          </div>
          <div className="text-base font-extrabold text-emerald-400">Moderate</div>
        </div>
      </div>

      <div className="text-[11px] text-slate-400 flex items-start gap-1.5 bg-slate-950/60 p-2 rounded border border-slate-800">
        <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
        <span>
          Contextual prototype environmental data based on Nagpur region seasonal averages. Evaluated alongside ML visual indicators for risk scoring.
        </span>
      </div>
    </div>
  );
};

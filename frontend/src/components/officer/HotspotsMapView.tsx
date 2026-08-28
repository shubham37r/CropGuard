import React, { useEffect, useState } from 'react';
import { Flame, RefreshCw, Layers } from 'lucide-react';
import type { HotspotResponse } from '../../types';
import { api } from '../../api/client';
import { LeafletMap } from '../maps/LeafletMap';
import { NoticeBanner } from '../common/NoticeBanner';

interface HotspotsMapViewProps {
  onSelectReport: (reportId: number) => void;
}

export const HotspotsMapView: React.FC<HotspotsMapViewProps> = ({ onSelectReport }) => {
  const [hotspotsData, setHotspotsData] = useState<HotspotResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const res = await api.getHotspots();
      setHotspotsData(res);
    } catch (err) {
      console.error('Failed to load hotspot data', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs uppercase tracking-wider text-rose-700 font-bold">Geospatial Surveillance</span>
          <h1 className="text-2xl font-extrabold text-slate-900 flex items-center gap-2">
            <Flame className="w-6 h-6 text-rose-600" />
            <span>Emerging Hotspot & Outbreak Map</span>
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">Radius-based report clustering for Nagpur agricultural blocks</p>
        </div>

        <button
          onClick={loadData}
          className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold px-4 py-2 rounded-lg text-xs border border-slate-300 flex items-center gap-1.5 transition self-start sm:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Map Data</span>
        </button>
      </div>

      <NoticeBanner message="GEOSPATIAL HOTSPOT CLUSTERING: Groups reports occurring within a 15km radius to highlight emerging clusters. This is a rule-based prototype spatial signal, not an advanced predictive epidemiological model." />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3 h-[520px] rounded-xl overflow-hidden border border-slate-300 shadow-md">
          {loading ? (
            <div className="w-full h-full bg-slate-100 flex items-center justify-center text-slate-400 text-sm">
              Loading geospatial map layers...
            </div>
          ) : (
            <LeafletMap
              mode="hotspots"
              center={[21.1458, 79.0882]}
              zoom={10}
              points={hotspotsData?.points || []}
              clusters={hotspotsData?.clusters || []}
              onSelectPoint={(reportId) => onSelectReport(reportId)}
            />
          )}
        </div>

        <div className="space-y-4">
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <h3 className="font-bold text-slate-900 text-sm flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-blue-600" />
              <span>Map Marker Legend</span>
            </h3>
            
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-rose-600 shrink-0" />
                <span className="font-semibold text-slate-800">HIGH Risk Report</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-amber-500 shrink-0" />
                <span className="font-semibold text-slate-800">MEDIUM Risk Report</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-emerald-500 shrink-0" />
                <span className="font-semibold text-slate-800">LOW Risk Report</span>
              </div>
              <div className="flex items-center gap-2 pt-1 border-t border-slate-100">
                <span className="w-4 h-4 rounded-full border-2 border-dashed border-rose-600 bg-rose-200/50 shrink-0" />
                <span className="font-semibold text-slate-800">Emerging Hotspot Cluster (15km)</span>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <h3 className="font-bold text-slate-900 text-sm">
              Detected Clusters ({hotspotsData?.clusters.length || 0})
            </h3>

            {hotspotsData?.clusters.length === 0 ? (
              <div className="text-xs text-slate-500">No active high-risk report clusters detected in 15km radius.</div>
            ) : (
              <div className="space-y-2 max-h-72 overflow-y-auto">
                {hotspotsData?.clusters.map((c) => (
                  <div key={c.cluster_id} className="p-3 bg-rose-50 rounded-lg border border-rose-200 space-y-1">
                    <div className="font-bold text-rose-950 text-xs">{c.title}</div>
                    <div className="text-[11px] text-rose-800">{c.description}</div>
                    <div className="text-[10px] text-rose-700 font-medium">
                      Dominant: {c.dominant_condition} ({c.dominant_crop})
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

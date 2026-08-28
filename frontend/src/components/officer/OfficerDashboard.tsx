import React, { useEffect, useState } from 'react';
import { ShieldAlert, Flame, ArrowRight, MapPin } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import type { CropReport, HotspotResponse } from '../../types';
import { api } from '../../api/client';
import { RiskBadge } from '../common/RiskBadge';
import { NoticeBanner } from '../common/NoticeBanner';

interface OfficerDashboardProps {
  onNavigate: (tab: string) => void;
  onOpenReport: (report: CropReport) => void;
}

export const OfficerDashboard: React.FC<OfficerDashboardProps> = ({ onNavigate, onOpenReport }) => {
  const { currentUser } = useAuth();
  const [reports, setReports] = useState<CropReport[]>([]);
  const [hotspotsData, setHotspotsData] = useState<HotspotResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [repRes, hotRes] = await Promise.all([api.getReports(), api.getHotspots()]);
        setReports(repRes);
        setHotspotsData(hotRes);
      } catch (err) {
        console.error('Error loading officer dashboard data', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const totalCount = reports.length;
  const pendingCount = reports.filter((r) => r.status === 'PENDING_VERIFICATION').length;
  const highRiskCount = reports.filter((r) => r.risk_assessment?.risk_level === 'HIGH').length;
  const confirmedCount = reports.filter((r) => r.status === 'CONFIRMED').length;
  const hotspotCount = hotspotsData?.clusters.length || 0;

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <span className="text-xs uppercase tracking-wider text-blue-700 font-bold">Extension Officer Command Center</span>
          <h1 className="text-2xl font-extrabold text-slate-900">Welcome, {currentUser?.name}</h1>
          <p className="text-sm text-slate-600 flex items-center gap-1.5 mt-1">
            <MapPin className="w-4 h-4 text-blue-600" />
            <span>{currentUser?.region || 'Nagpur Division Agriculture Office'}</span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('verification-table')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold px-4 py-2.5 rounded-lg shadow text-sm flex items-center gap-2 transition"
          >
            <ShieldAlert className="w-4 h-4" />
            <span>Review Pending ({pendingCount})</span>
          </button>
          <button
            onClick={() => onNavigate('hotspot-map')}
            className="bg-slate-900 hover:bg-slate-800 text-white font-bold px-4 py-2.5 rounded-lg shadow text-sm flex items-center gap-2 transition"
          >
            <Flame className="w-4 h-4 text-rose-400" />
            <span>Hotspot Map</span>
          </button>
        </div>
      </div>

      <NoticeBanner message="EXTENDER OFFICER WORKFLOW: Verify incoming crop disease & pest reports, confirm diagnoses, and monitor radius-based spatial outbreak clusters." />

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-xs font-semibold text-slate-500 uppercase">Total Reports</div>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">{totalCount}</div>
          <div className="text-[11px] text-slate-400 mt-0.5">Nagpur Region</div>
        </div>

        <div className="bg-amber-50 p-4 rounded-xl border border-amber-200 shadow-sm">
          <div className="text-xs font-semibold text-amber-800 uppercase">Pending Review</div>
          <div className="text-2xl font-extrabold text-amber-900 mt-1">{pendingCount}</div>
          <div className="text-[11px] text-amber-700 mt-0.5">Needs Verification</div>
        </div>

        <div className="bg-rose-50 p-4 rounded-xl border border-rose-200 shadow-sm">
          <div className="text-xs font-semibold text-rose-800 uppercase">High Risk Cases</div>
          <div className="text-2xl font-extrabold text-rose-900 mt-1">{highRiskCount}</div>
          <div className="text-[11px] text-rose-700 mt-0.5">Attention Required</div>
        </div>

        <div className="bg-emerald-50 p-4 rounded-xl border border-emerald-200 shadow-sm">
          <div className="text-xs font-semibold text-emerald-800 uppercase">Confirmed Cases</div>
          <div className="text-2xl font-extrabold text-emerald-900 mt-1">{confirmedCount}</div>
          <div className="text-[11px] text-emerald-700 mt-0.5">Officer Verified</div>
        </div>

        <div className="bg-purple-50 p-4 rounded-xl border border-purple-200 shadow-sm col-span-2 lg:col-span-1">
          <div className="text-xs font-semibold text-purple-800 uppercase">Emerging Hotspots</div>
          <div className="text-2xl font-extrabold text-purple-900 mt-1">{hotspotCount} Clusters</div>
          <div className="text-[11px] text-purple-700 mt-0.5">15km Radius Clusters</div>
        </div>
      </div>

      {hotspotsData && hotspotsData.clusters.length > 0 && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-bold text-slate-900 text-lg flex items-center gap-2">
              <Flame className="w-5 h-5 text-rose-600" />
              Active Emerging Hotspot Alerts ({hotspotsData.clusters.length})
            </h2>
            <button
              onClick={() => onNavigate('hotspot-map')}
              className="text-xs font-bold text-blue-700 hover:text-blue-800 flex items-center gap-1"
            >
              <span>Inspect on Map</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {hotspotsData.clusters.map((cluster) => (
              <div key={cluster.cluster_id} className="bg-rose-50/70 border border-rose-200 p-4 rounded-xl space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-rose-950 text-sm">{cluster.title}</span>
                  <span className="bg-rose-600 text-white text-[10px] font-extrabold px-2 py-0.5 rounded-full uppercase">
                    {cluster.high_risk_count} High Risk Cases
                  </span>
                </div>
                <p className="text-xs text-rose-900">{cluster.description}</p>
                <div className="text-xs text-rose-800 font-medium flex items-center gap-3 pt-1 border-t border-rose-200/80">
                  <span>Dominant Crop: <strong>{cluster.dominant_crop}</strong></span>
                  <span>District: <strong>{cluster.district}</strong></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <h2 className="font-bold text-slate-900 text-lg flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-600" />
            Pending Verification Cases
          </h2>
          <button
            onClick={() => onNavigate('verification-table')}
            className="text-xs font-bold text-blue-700 hover:text-blue-800 flex items-center gap-1"
          >
            <span>View All Verification Queue</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {loading ? (
          <div className="p-8 text-center text-slate-400 text-sm">Loading verification queue...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 text-xs uppercase font-bold tracking-wider">
                <tr>
                  <th className="py-3.5 px-4">Report ID & Farmer</th>
                  <th className="py-3.5 px-4">Crop</th>
                  <th className="py-3.5 px-4">Suspected Condition</th>
                  <th className="py-3.5 px-4">Risk Level</th>
                  <th className="py-3.5 px-4">Location</th>
                  <th className="py-3.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {reports
                  .filter((r) => r.status === 'PENDING_VERIFICATION' || r.risk_assessment?.risk_level === 'HIGH')
                  .slice(0, 5)
                  .map((report) => (
                    <tr key={report.id} className="hover:bg-slate-50 transition">
                      <td className="py-3.5 px-4">
                        <div className="font-bold text-slate-900">Report #{report.id}</div>
                        <div className="text-xs text-slate-500">{report.farmer_name || 'Farmer'}</div>
                      </td>

                      <td className="py-3.5 px-4 font-semibold text-slate-800">{report.crop}</td>

                      <td className="py-3.5 px-4 font-medium text-slate-800">
                        {report.analysis?.condition?.name || 'Pending'}
                      </td>

                      <td className="py-3.5 px-4">
                        <RiskBadge level={report.risk_assessment?.risk_level} size="sm" />
                      </td>

                      <td className="py-3.5 px-4 text-xs text-slate-600">
                        {report.location.address || report.location.district}
                      </td>

                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={() => onOpenReport(report)}
                          className="bg-blue-600 hover:bg-blue-700 text-white font-bold px-3 py-1.5 rounded text-xs shadow transition"
                        >
                          Review Case
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

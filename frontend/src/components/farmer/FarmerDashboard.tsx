import React, { useEffect, useState } from 'react';
import { Camera, FileText, AlertTriangle, ShieldCheck, ArrowRight, Activity, MapPin } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import type { CropReport } from '../../types';
import { api } from '../../api/client';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { NoticeBanner } from '../common/NoticeBanner';
import { WeatherContextCard } from '../common/WeatherContextCard';


interface FarmerDashboardProps {
  onNavigate: (tab: string) => void;
  onSelectReport: (report: CropReport) => void;
}

export const FarmerDashboard: React.FC<FarmerDashboardProps> = ({ onNavigate, onSelectReport }) => {
  const { currentUser } = useAuth();
  const [reports, setReports] = useState<CropReport[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const loadReports = async () => {
      try {
        if (currentUser) {
          const data = await api.getReports({ farmer_id: currentUser.id });
          setReports(data);
        }
      } catch (err) {
        console.error('Failed to load farmer reports', err);
      } finally {
        setLoading(false);
      }
    };
    loadReports();
  }, [currentUser]);

  const highRiskCount = reports.filter((r) => r.risk_assessment?.risk_level === 'HIGH').length;
  const pendingCount = reports.filter((r) => r.status === 'PENDING_VERIFICATION').length;

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <span className="text-xs uppercase tracking-wider text-emerald-700 font-bold">Farmer Portal</span>
          <h1 className="text-2xl font-extrabold text-slate-900">Welcome, {currentUser?.name}</h1>
          <p className="text-sm text-slate-600 flex items-center gap-1.5 mt-1">
            <MapPin className="w-4 h-4 text-emerald-600" />
            <span>{currentUser?.region || 'Nagpur District, Maharashtra'}</span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('check-crop')}
            className="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold px-4 py-2.5 rounded-lg shadow flex items-center gap-2 transition"
          >
            <Camera className="w-4 h-4" />
            <span>Check Crop Health</span>
          </button>
          <button
            onClick={() => onNavigate('my-reports')}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium px-4 py-2.5 rounded-lg border border-slate-300 flex items-center gap-2 transition"
          >
            <FileText className="w-4 h-4" />
            <span>View My Reports</span>
          </button>
        </div>
      </div>

      <NoticeBanner />

      <WeatherContextCard district="Nagpur" />


      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
          <div className="p-3 bg-emerald-50 rounded-lg text-emerald-600">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase">Overall Crop Health Status</div>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">
              {highRiskCount > 0 ? 'ATTENTION REQUIRED' : 'STABLE'}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">Based on your recent reports</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
          <div className="p-3 bg-rose-50 rounded-lg text-rose-600">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase">High Risk Alerts</div>
            <div className="text-xl font-extrabold text-rose-700 mt-0.5">{highRiskCount} Active</div>
            <div className="text-xs text-slate-500 mt-0.5">Requires immediate IPM attention</div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
          <div className="p-3 bg-amber-50 rounded-lg text-amber-600">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xs font-medium text-slate-500 uppercase">Pending Officer Reviews</div>
            <div className="text-xl font-extrabold text-amber-700 mt-0.5">{pendingCount} Pending</div>
            <div className="text-xs text-slate-500 mt-0.5">Referred for Extension Officer verification</div>
          </div>
        </div>
      </div>

      <div className="bg-amber-50 border border-amber-300 rounded-xl p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
          <div>
            <h3 className="font-bold text-amber-950 text-base">Regional Advisory Alert: Nagpur District</h3>
            <p className="text-xs text-amber-900 mt-1 leading-relaxed">
              High humidity ({'>'}80%) in Katol and Saoner blocks has increased environmental suitability for 
              <strong> Pink Bollworm in Cotton</strong> and <strong>Early Blight in Tomato</strong>. Scout fields daily.
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <h2 className="font-bold text-slate-900 text-lg flex items-center gap-2">
            <FileText className="w-5 h-5 text-emerald-600" />
            Recent Crop Health Reports
          </h2>
          <button
            onClick={() => onNavigate('my-reports')}
            className="text-xs font-bold text-emerald-700 hover:text-emerald-800 flex items-center gap-1"
          >
            <span>View All</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {loading ? (
          <div className="p-8 text-center text-slate-400 text-sm">Loading recent reports...</div>
        ) : reports.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">
            No crop health reports submitted yet.{' '}
            <button onClick={() => onNavigate('check-crop')} className="text-emerald-700 underline font-semibold">
              Submit your first crop check
            </button>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {reports.slice(0, 4).map((report) => (
              <div
                key={report.id}
                onClick={() => onSelectReport(report)}
                className="p-4 hover:bg-slate-50 transition cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-3"
              >
                <div className="flex items-center gap-3">
                  <img
                    src={report.image_url}
                    alt={report.crop}
                    className="w-12 h-12 rounded-lg object-cover border border-slate-200 shrink-0"
                    onError={(e: any) => {
                      e.target.src = 'https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=100';
                    }}
                  />
                  <div>
                    <div className="font-bold text-slate-900 text-sm flex items-center gap-2">
                      <span>{report.crop}</span>
                      <span className="text-xs font-normal text-slate-500">({report.growth_stage})</span>
                    </div>
                    <div className="text-xs text-slate-600 mt-0.5">
                      Suspected Condition:{' '}
                      <strong className="text-slate-800">
                        {report.analysis?.condition?.name || 'Under Analysis'}
                      </strong>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {new Date(report.created_at).toLocaleDateString()} • {report.location.address || report.location.district}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 self-end sm:self-center">
                  <RiskBadge level={report.risk_assessment?.risk_level} size="sm" />
                  <StatusBadge status={report.status} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

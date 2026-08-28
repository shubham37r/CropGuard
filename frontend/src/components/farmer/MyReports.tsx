import React, { useEffect, useState } from 'react';
import { Search, Filter, Calendar, Eye } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import type { CropReport } from '../../types';
import { api } from '../../api/client';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { NoticeBanner } from '../common/NoticeBanner';

interface MyReportsProps {
  onSelectReport: (report: CropReport) => void;
  onNavigateToCheckCrop: () => void;
}

export const MyReports: React.FC<MyReportsProps> = ({ onSelectReport, onNavigateToCheckCrop }) => {
  const { currentUser } = useAuth();
  const [reports, setReports] = useState<CropReport[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [search, setSearch] = useState<string>('');
  const [cropFilter, setCropFilter] = useState<string>('ALL');

  useEffect(() => {
    const fetchReports = async () => {
      try {
        if (currentUser) {
          const data = await api.getReports({ farmer_id: currentUser.id });
          setReports(data);
        }
      } catch (err) {
        console.error('Error fetching my reports', err);
      } finally {
        setLoading(false);
      }
    };
    fetchReports();
  }, [currentUser]);

  const filteredReports = reports.filter((r) => {
    const matchesCrop = cropFilter === 'ALL' || r.crop === cropFilter;
    const condName = r.analysis?.condition?.name || '';
    const matchesSearch =
      r.crop.toLowerCase().includes(search.toLowerCase()) ||
      condName.toLowerCase().includes(search.toLowerCase()) ||
      r.location.address?.toLowerCase().includes(search.toLowerCase());
    return matchesCrop && matchesSearch;
  });

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs uppercase tracking-wider text-emerald-700 font-bold">Farmer History</span>
          <h1 className="text-2xl font-extrabold text-slate-900">My Crop Health Reports</h1>
          <p className="text-xs text-slate-500 mt-0.5">Track submitted reports, risk scores, and officer verification statuses</p>
        </div>
        <button
          onClick={onNavigateToCheckCrop}
          className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-4 py-2.5 rounded-lg text-sm shadow transition self-start sm:self-auto"
        >
          + New Crop Check
        </button>
      </div>

      <NoticeBanner />

      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            placeholder="Search by crop, disease name, or location..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500 shrink-0" />
          <select
            value={cropFilter}
            onChange={(e) => setCropFilter(e.target.value)}
            className="bg-slate-50 border border-slate-300 rounded-lg py-2 px-3 text-sm font-medium focus:ring-2 focus:ring-emerald-500 outline-none"
          >
            <option value="ALL">All Crops</option>
            <option value="Tomato">Tomato</option>
            <option value="Cotton">Cotton</option>
            <option value="Soybean">Soybean</option>
          </select>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-slate-400 text-sm">Loading report history...</div>
        ) : filteredReports.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">
            No crop reports found matching filter parameters.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 text-xs uppercase font-bold tracking-wider">
                <tr>
                  <th className="py-3.5 px-4">Date & ID</th>
                  <th className="py-3.5 px-4">Crop & Stage</th>
                  <th className="py-3.5 px-4">Suspected Condition</th>
                  <th className="py-3.5 px-4">Confidence</th>
                  <th className="py-3.5 px-4">Contextual Risk</th>
                  <th className="py-3.5 px-4">Verification Status</th>
                  <th className="py-3.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredReports.map((report) => (
                  <tr key={report.id} className="hover:bg-slate-50 transition">
                    <td className="py-3.5 px-4">
                      <div className="font-bold text-slate-900">#{report.id}</div>
                      <div className="text-xs text-slate-500 flex items-center gap-1">
                        <Calendar className="w-3 h-3 text-slate-400" />
                        {new Date(report.created_at).toLocaleDateString()}
                      </div>
                    </td>

                    <td className="py-3.5 px-4">
                      <div className="font-bold text-slate-800">{report.crop}</div>
                      <div className="text-xs text-slate-500">{report.growth_stage}</div>
                    </td>

                    <td className="py-3.5 px-4 font-semibold text-slate-800">
                      {report.analysis?.condition?.name || 'Pending Analysis'}
                    </td>

                    <td className="py-3.5 px-4 text-slate-700 font-bold">
                      {report.analysis?.confidence?.toFixed(0)}%
                    </td>

                    <td className="py-3.5 px-4">
                      <RiskBadge level={report.risk_assessment?.risk_level} size="sm" />
                    </td>

                    <td className="py-3.5 px-4">
                      <StatusBadge status={report.status} />
                    </td>

                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => onSelectReport(report)}
                        className="bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold px-3 py-1.5 rounded text-xs border border-slate-300 inline-flex items-center gap-1 transition"
                      >
                        <Eye className="w-3.5 h-3.5 text-slate-600" />
                        <span>Details</span>
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

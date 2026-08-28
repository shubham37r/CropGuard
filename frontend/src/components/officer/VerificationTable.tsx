import React, { useEffect, useState } from 'react';
import { Search, Eye } from 'lucide-react';
import type { CropReport } from '../../types';
import { api } from '../../api/client';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { NoticeBanner } from '../common/NoticeBanner';

interface VerificationTableProps {
  onOpenReport: (report: CropReport) => void;
}

export const VerificationTable: React.FC<VerificationTableProps> = ({ onOpenReport }) => {
  const [reports, setReports] = useState<CropReport[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [search, setSearch] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [riskFilter, setRiskFilter] = useState<string>('ALL');

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    setLoading(true);
    try {
      const data = await api.getReports();
      setReports(data);
    } catch (err) {
      console.error('Failed to load officer verification reports', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredReports = reports.filter((r) => {
    const matchesStatus = statusFilter === 'ALL' || r.status === statusFilter;
    const matchesRisk = riskFilter === 'ALL' || r.risk_assessment?.risk_level === riskFilter;
    const condName = r.analysis?.condition?.name || '';
    const farmerName = r.farmer_name || '';
    const matchesSearch =
      r.crop.toLowerCase().includes(search.toLowerCase()) ||
      condName.toLowerCase().includes(search.toLowerCase()) ||
      farmerName.toLowerCase().includes(search.toLowerCase()) ||
      r.location.address?.toLowerCase().includes(search.toLowerCase());

    return matchesStatus && matchesRisk && matchesSearch;
  });

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs uppercase tracking-wider text-blue-700 font-bold">Extension Officer Verification</span>
          <h1 className="text-2xl font-extrabold text-slate-900">Crop Health Case Registry</h1>
          <p className="text-xs text-slate-500 mt-0.5">Review, verify, or reject crop health reports across Nagpur region</p>
        </div>
        <button
          onClick={fetchReports}
          className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold px-4 py-2 rounded-lg text-xs border border-slate-300 transition"
        >
          Refresh Registry
        </button>
      </div>

      <NoticeBanner />

      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            placeholder="Search by farmer name, report ID, crop, condition, location..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none"
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-slate-500">Status:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-50 border border-slate-300 rounded-lg py-2 px-3 text-xs font-semibold focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option value="ALL">All Statuses</option>
              <option value="PENDING_VERIFICATION">Pending Verification</option>
              <option value="CONFIRMED">Confirmed</option>
              <option value="REJECTED">Rejected</option>
              <option value="NEEDS_MORE_INFO">Needs More Info</option>
              <option value="ANALYZED">Analyzed (Auto)</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-slate-500">Risk:</span>
            <select
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value)}
              className="bg-slate-50 border border-slate-300 rounded-lg py-2 px-3 text-xs font-semibold focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option value="ALL">All Risks</option>
              <option value="HIGH">HIGH Risk</option>
              <option value="MEDIUM">MEDIUM Risk</option>
              <option value="LOW">LOW Risk</option>
            </select>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-slate-400 text-sm">Loading registry...</div>
        ) : filteredReports.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">No reports match selected filters.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 text-xs uppercase font-bold tracking-wider">
                <tr>
                  <th className="py-3.5 px-4">Report ID & Farmer</th>
                  <th className="py-3.5 px-4">Crop & Stage</th>
                  <th className="py-3.5 px-4">Suspected Condition</th>
                  <th className="py-3.5 px-4">Confidence</th>
                  <th className="py-3.5 px-4">Contextual Risk</th>
                  <th className="py-3.5 px-4">Location</th>
                  <th className="py-3.5 px-4">Date</th>
                  <th className="py-3.5 px-4">Status</th>
                  <th className="py-3.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredReports.map((report) => (
                  <tr key={report.id} className="hover:bg-slate-50 transition">
                    <td className="py-3.5 px-4">
                      <div className="font-bold text-slate-900">#{report.id}</div>
                      <div className="text-xs text-slate-500">{report.farmer_name || 'Farmer'}</div>
                    </td>

                    <td className="py-3.5 px-4">
                      <div className="font-bold text-slate-800">{report.crop}</div>
                      <div className="text-xs text-slate-500">{report.growth_stage}</div>
                    </td>

                    <td className="py-3.5 px-4 font-semibold text-slate-800">
                      {report.analysis?.condition?.name || 'Unspecified'}
                    </td>

                    <td className="py-3.5 px-4 font-bold text-slate-700">
                      {report.analysis?.confidence?.toFixed(0)}%
                    </td>

                    <td className="py-3.5 px-4">
                      <RiskBadge level={report.risk_assessment?.risk_level} size="sm" />
                    </td>

                    <td className="py-3.5 px-4 text-xs text-slate-600">
                      {report.location.address || report.location.district}
                    </td>

                    <td className="py-3.5 px-4 text-xs text-slate-500 whitespace-nowrap">
                      {new Date(report.created_at).toLocaleDateString()}
                    </td>

                    <td className="py-3.5 px-4">
                      <StatusBadge status={report.status} />
                    </td>

                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => onOpenReport(report)}
                        className="bg-blue-600 hover:bg-blue-700 text-white font-bold px-3 py-1.5 rounded text-xs shadow transition inline-flex items-center gap-1"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Review</span>
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

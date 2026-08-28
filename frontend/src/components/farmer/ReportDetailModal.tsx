import React from 'react';
import { X, Calendar, MapPin, CheckCircle, UserCheck } from 'lucide-react';
import type { CropReport } from '../../types';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { NoticeBanner } from '../common/NoticeBanner';

interface ReportDetailModalProps {
  report: CropReport | null;
  onClose: () => void;
}

export const ReportDetailModal: React.FC<ReportDetailModalProps> = ({ report, onClose }) => {
  if (!report) return null;

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white max-w-3xl w-full rounded-2xl border border-slate-200 shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">
        <div className="px-6 py-4 bg-slate-900 text-white flex items-center justify-between">
          <div>
            <span className="text-xs uppercase tracking-wider text-emerald-400 font-bold">Crop Health Report Details</span>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span>{report.crop} Report #{report.id}</span>
              <span className="text-xs font-normal text-slate-300">({report.growth_stage})</span>
            </h2>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6 overflow-y-auto flex-1 text-sm">
          <NoticeBanner message={report.risk_assessment?.methodology_note} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <img
                src={report.image_url}
                alt={report.crop}
                className="w-full h-56 object-cover rounded-xl border border-slate-200 shadow-sm"
                onError={(e: any) => {
                  e.target.src = 'https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600';
                }}
              />
              <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-1.5 text-xs">
                <div className="flex items-center gap-1.5 text-slate-700">
                  <Calendar className="w-3.5 h-3.5 text-slate-500" />
                  <span>Submitted: {new Date(report.created_at).toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-700">
                  <MapPin className="w-3.5 h-3.5 text-slate-500" />
                  <span>Location: {report.location.address || report.location.district} ({report.location.latitude.toFixed(4)}, {report.location.longitude.toFixed(4)})</span>
                </div>
                {report.symptoms_description && (
                  <div className="text-slate-600 border-t border-slate-200 pt-1.5 mt-1.5">
                    <strong>Farmer Symptoms Note:</strong> "{report.symptoms_description}"
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase text-slate-500">Analysis Result</span>
                  <StatusBadge status={report.status} />
                </div>
                <div>
                  <div className="text-xs text-slate-500">
                    {(report.analysis?.confidence || 0) < 70 ? 'Possible Condition:' : 'Likely Condition:'}
                  </div>
                  <div className="text-xl font-extrabold text-slate-900">
                    {report.analysis?.condition?.name || 'Pending Diagnosis'}
                  </div>
                  <div className="text-xs text-slate-500 mt-0.5">
                    Confidence: <strong>{report.analysis?.confidence?.toFixed(0)}%</strong> • Type: <strong className="uppercase">{report.analysis?.condition?.type}</strong>
                  </div>
                </div>

                <div className="border-t border-slate-200 pt-3 flex items-center justify-between">
                  <span className="text-xs font-bold uppercase text-slate-500">Prototype Contextual Risk</span>
                  <RiskBadge level={report.risk_assessment?.risk_level} />
                </div>
              </div>

              {report.verification && report.verification.officer_notes && (
                <div className="bg-blue-50 border border-blue-200 p-4 rounded-xl space-y-2">
                  <div className="flex items-center gap-2 text-blue-900 font-bold text-xs uppercase">
                    <UserCheck className="w-4 h-4 text-blue-700" />
                    <span>Officer Verification Note ({report.verification.officer_name || 'Extension Officer'})</span>
                  </div>
                  <p className="text-xs text-blue-950 leading-relaxed font-medium">
                    "{report.verification.officer_notes}"
                  </p>
                  <div className="text-[11px] text-blue-700">
                    Verified on: {report.verification.verified_at ? new Date(report.verification.verified_at).toLocaleDateString() : 'Recent'}
                  </div>
                </div>
              )}
            </div>
          </div>

          {report.risk_assessment?.contributing_factors && (
            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-sm">Risk Contributing Factors:</h4>
              <ul className="space-y-1.5">
                {report.risk_assessment.contributing_factors.map((factor, idx) => (
                  <li key={idx} className="text-xs text-slate-700 bg-slate-50 p-2 rounded border border-slate-200 flex items-center gap-2">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {report.advisory && (
            <div className="bg-slate-900 text-white p-5 rounded-xl space-y-3">
              <h4 className="font-bold text-emerald-300 text-sm">Recommended IPM Guidance Actions:</h4>
              <ul className="space-y-1.5 text-xs text-slate-200">
                {report.advisory.actions.map((act, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-emerald-400 font-bold">•</span>
                    <span>{act}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="px-6 py-3 bg-slate-50 border-t border-slate-200 text-right">
          <button
            onClick={onClose}
            className="bg-slate-800 hover:bg-slate-900 text-white font-bold px-4 py-2 rounded-lg text-xs transition"
          >
            Close Window
          </button>
        </div>
      </div>
    </div>
  );
};

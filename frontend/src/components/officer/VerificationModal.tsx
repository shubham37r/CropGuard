import React, { useState } from 'react';
import { X, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import type { CropReport } from '../../types';
import { useAuth } from '../../context/AuthContext';
import { api } from '../../api/client';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { NoticeBanner } from '../common/NoticeBanner';
import { LeafletMap } from '../maps/LeafletMap';

interface VerificationModalProps {
  report: CropReport | null;
  onClose: () => void;
  onUpdated: (updatedReport: CropReport) => void;
}

export const VerificationModal: React.FC<VerificationModalProps> = ({ report, onClose, onUpdated }) => {
  const { currentUser } = useAuth();
  const [officerNotes, setOfficerNotes] = useState<string>(report?.verification?.officer_notes || '');
  const [submitting, setSubmitting] = useState<boolean>(false);

  if (!report) return null;

  const handleAction = async (status: 'CONFIRMED' | 'REJECTED' | 'NEEDS_MORE_INFO') => {
    setSubmitting(true);
    try {
      const officerId = currentUser?.id || 4;
      const updated = await api.verifyReport(report.id, officerId, status, officerNotes);
      onUpdated(updated);
      onClose();
    } catch (err) {
      console.error('Failed to submit verification action', err);
      alert('Failed to update verification status.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white max-w-4xl w-full rounded-2xl border border-slate-200 shadow-2xl overflow-hidden max-h-[92vh] flex flex-col">
        <div className="px-6 py-4 bg-slate-900 text-white flex items-center justify-between">
          <div>
            <span className="text-xs uppercase tracking-wider text-blue-400 font-bold">Extension Officer Review</span>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span>Report #{report.id} - {report.crop}</span>
              <StatusBadge status={report.status} />
            </h2>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6 overflow-y-auto flex-1 text-sm">
          <NoticeBanner message={report.risk_assessment?.methodology_note} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <span className="text-xs font-bold uppercase text-slate-500 block mb-1">Uploaded Crop Image</span>
                <img
                  src={report.image_url}
                  alt={report.crop}
                  className="w-full h-52 object-cover rounded-xl border border-slate-200 shadow-sm"
                  onError={(e: any) => {
                    e.target.src = 'https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600';
                  }}
                />
              </div>

              <div className="space-y-1">
                <span className="text-xs font-bold uppercase text-slate-500 block">Field Location</span>
                <div className="h-44 rounded-xl overflow-hidden border border-slate-300">
                  <LeafletMap
                    mode="picker"
                    center={[report.location.latitude, report.location.longitude]}
                    selectedLat={report.location.latitude}
                    selectedLng={report.location.longitude}
                  />
                </div>
                <div className="text-[11px] text-slate-500 mt-1">
                  {report.location.address || report.location.district} ({report.location.latitude.toFixed(4)}, {report.location.longitude.toFixed(4)})
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase text-slate-500">Submitted Diagnosis</span>
                  <RiskBadge level={report.risk_assessment?.risk_level} />
                </div>
                <div>
                  <div className="text-2xl font-extrabold text-slate-900">
                    {report.analysis?.condition?.name || 'Unspecified'}
                  </div>
                  <div className="text-xs text-slate-600 mt-0.5">
                    Confidence: <strong>{report.analysis?.confidence?.toFixed(0)}%</strong> • Type: <strong className="uppercase">{report.analysis?.condition?.type}</strong>
                  </div>
                </div>

                <div className="border-t border-slate-200 pt-2 text-xs space-y-1 text-slate-700">
                  <div>Farmer: <strong>{report.farmer_name || 'Rajesh Patel'}</strong></div>
                  <div>Crop & Stage: <strong>{report.crop} ({report.growth_stage})</strong></div>
                  <div>Submitted Date: <strong>{new Date(report.created_at).toLocaleString()}</strong></div>
                  {report.symptoms_description && (
                    <div className="text-slate-600 italic bg-white p-2 rounded border border-slate-200 mt-2">
                      Farmer Note: "{report.symptoms_description}"
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-xs font-bold uppercase text-slate-700">
                  Officer Verification Notes *
                </label>
                <textarea
                  rows={3}
                  placeholder="Enter expert inspection notes, field advice, or confirmation rationale..."
                  value={officerNotes}
                  onChange={(e) => setOfficerNotes(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-xs focus:ring-2 focus:ring-blue-500 outline-none"
                />
              </div>

              <div className="space-y-2 border-t border-slate-200 pt-4">
                <span className="text-xs font-bold uppercase text-slate-700 block">Take Official Case Action:</span>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => handleAction('CONFIRMED')}
                    disabled={submitting}
                    className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-2 rounded-lg text-xs shadow flex items-center justify-center gap-1 transition disabled:opacity-50"
                  >
                    <CheckCircle className="w-4 h-4" />
                    <span>Confirm Case</span>
                  </button>

                  <button
                    onClick={() => handleAction('REJECTED')}
                    disabled={submitting}
                    className="bg-rose-600 hover:bg-rose-700 text-white font-bold py-2 px-2 rounded-lg text-xs shadow flex items-center justify-center gap-1 transition disabled:opacity-50"
                  >
                    <XCircle className="w-4 h-4" />
                    <span>Reject Case</span>
                  </button>

                  <button
                    onClick={() => handleAction('NEEDS_MORE_INFO')}
                    disabled={submitting}
                    className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-2 rounded-lg text-xs shadow flex items-center justify-center gap-1 transition disabled:opacity-50"
                  >
                    <AlertCircle className="w-4 h-4" />
                    <span>Request Info</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

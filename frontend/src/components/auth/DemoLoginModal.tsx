import React from 'react';
import { Sprout, User, UserCheck, Shield, Sparkles } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import type { UserRole } from '../../types';

interface DemoLoginModalProps {
  onSelectRole: (role: UserRole) => void;
  onClose?: () => void;
}

export const DemoLoginModal: React.FC<DemoLoginModalProps> = ({ onSelectRole, onClose }) => {
  const { switchUserRole } = useAuth();

  const handleSelect = async (role: UserRole) => {
    await switchUserRole(role);
    onSelectRole(role);
    if (onClose) onClose();
  };

  return (
    <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white max-w-md w-full rounded-2xl shadow-2xl border border-slate-200 overflow-hidden">
        
        {/* Header Banner */}
        <div className="bg-gradient-to-br from-slate-900 via-emerald-950 to-slate-900 text-white p-8 text-center relative">
          <div className="w-14 h-14 bg-emerald-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-emerald-900/50">
            <Sprout className="w-8 h-8 text-white" />
          </div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-900/80 border border-emerald-700/60 rounded-full text-[11px] font-mono text-emerald-300 mb-2">
            <Sparkles className="w-3 h-3 text-emerald-400" />
            <span>SIH DEMO PROTOTYPE</span>
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight">CropGuard</h1>
          <p className="text-xs text-slate-300 mt-1 max-w-xs mx-auto">
            AI-Powered Localized Crop Health & Risk Intelligence
          </p>
        </div>

        {/* Content & Role Buttons */}
        <div className="p-6 space-y-5">
          <div className="text-center">
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">Select Demo User Role</h2>
            <p className="text-xs text-slate-500 mt-0.5">Choose a portal persona to explore the interactive workflows</p>
          </div>

          <div className="space-y-3">
            <button
              onClick={() => handleSelect('FARMER')}
              className="w-full bg-slate-50 hover:bg-emerald-50/80 border-2 border-slate-200 hover:border-emerald-500 rounded-xl p-4 text-left transition group shadow-sm flex items-center justify-between"
            >
              <div className="flex items-center gap-3.5">
                <div className="p-3 bg-emerald-100 group-hover:bg-emerald-600 text-emerald-700 group-hover:text-white rounded-lg transition">
                  <User className="w-6 h-6" />
                </div>
                <div>
                  <div className="font-extrabold text-slate-900 group-hover:text-emerald-900 text-base">
                    Login as Farmer
                  </div>
                  <div className="text-xs text-slate-500 mt-0.5">
                    Rajesh Patel • Katol Sector, Nagpur
                  </div>
                  <div className="text-[11px] text-emerald-700 font-medium mt-1">
                    Upload crop photos • Get ML diagnosis & IPM advisory
                  </div>
                </div>
              </div>
            </button>

            <button
              onClick={() => handleSelect('EXTENSION_OFFICER')}
              className="w-full bg-slate-50 hover:bg-blue-50/80 border-2 border-slate-200 hover:border-blue-500 rounded-xl p-4 text-left transition group shadow-sm flex items-center justify-between"
            >
              <div className="flex items-center gap-3.5">
                <div className="p-3 bg-blue-100 group-hover:bg-blue-600 text-blue-700 group-hover:text-white rounded-lg transition">
                  <UserCheck className="w-6 h-6" />
                </div>
                <div>
                  <div className="font-extrabold text-slate-900 group-hover:text-blue-900 text-base">
                    Login as Extension Officer
                  </div>
                  <div className="text-xs text-slate-500 mt-0.5">
                    Dr. Anish Sharma • Nagpur Agriculture Division
                  </div>
                  <div className="text-[11px] text-blue-700 font-medium mt-1">
                    Review case queues • Verify reports • Track spatial hotspots
                  </div>
                </div>
              </div>
            </button>
          </div>

          <div className="bg-slate-100 p-3 rounded-lg text-center border border-slate-200">
            <div className="text-[11px] text-slate-500 flex items-center justify-center gap-1">
              <Shield className="w-3.5 h-3.5 text-slate-400" />
              <span>Interactive Role Selector — No password required for demo</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

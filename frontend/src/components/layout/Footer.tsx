import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-900 text-slate-400 text-xs py-6 border-t border-slate-800 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-2">
        <p className="font-semibold text-slate-300">
          CropGuard MVP — Localized Crop Health Early Warning System for SIH
        </p>
        <p className="max-w-3xl mx-auto text-slate-400 leading-relaxed text-[11px]">
          <strong className="text-amber-400">Prototype Disclaimer:</strong> Prototype risk scores and mock analysis results are not scientifically validated agricultural predictions and should not be used for real-world treatment decisions.
        </p>
        <p className="text-slate-500 text-[10px]">
          Demo region pre-populated for Nagpur District, Maharashtra (Tomato, Cotton, Soybean).
        </p>
      </div>
    </footer>
  );
};

import React, { useState } from 'react';
import { Upload, X, MapPin, CheckCircle, AlertCircle, ArrowLeft, ArrowRight, FileSearch, Sparkles, ShieldAlert } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import type { CropReport } from '../../types';
import { api, getBackendOrigin } from '../../api/client';

import { LeafletMap } from '../maps/LeafletMap';
import { RiskBadge } from '../common/RiskBadge';
import { NoticeBanner } from '../common/NoticeBanner';
import { WeatherContextCard } from '../common/WeatherContextCard';


interface CheckCropWizardProps {
  onComplete: (report: CropReport) => void;
  onCancel: () => void;
}

export const CheckCropWizard: React.FC<CheckCropWizardProps> = ({ onComplete, onCancel }) => {
  const { currentUser } = useAuth();
  const [step, setStep] = useState<number>(1);

  // Step 1: Image state
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);

  // Step 2: Context form state
  const [crop, setCrop] = useState<string>('Tomato');
  const [variety, setVariety] = useState<string>('');
  const [growthStage, setGrowthStage] = useState<string>('Flowering');
  const [symptoms, setSymptoms] = useState<string>('');
  const [district] = useState<string>('Nagpur');
  const [address, setAddress] = useState<string>('Katol Sector, Nagpur');
  const [lat, setLat] = useState<number>(21.2825);
  const [lng, setLng] = useState<number>(78.5840);

  // Step 3: Submitting state
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  
  // Step 4: Result state
  const [resultReport, setResultReport] = useState<CropReport | null>(null);
  const [referring, setReferring] = useState<boolean>(false);
  const [referredSuccess, setReferredSuccess] = useState<boolean>(false);

  // Drag & drop handlers
  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setImageError('Please select a valid image file (JPG, PNG, WEBP).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setImageError('Image file size exceeds 10MB limit.');
      return;
    }
    setImageError(null);
    setImageFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setImageFile(null);
    setImagePreview(null);
    setImageError(null);
  };

  // Submit Handler
  const handleSubmitReport = async () => {
    setIsSubmitting(true);
    try {
      let finalImageUrl = 'https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600&auto=format&fit=crop';

      if (imageFile) {
        try {
          const uploadRes = await api.uploadImage(imageFile);
          const backendOrigin = getBackendOrigin();
          const relUrl = uploadRes.image_url.startsWith('/') ? uploadRes.image_url : `/${uploadRes.image_url}`;
          finalImageUrl = `${backendOrigin}${relUrl}`;

        } catch (e) {
          console.warn('Image upload endpoint fallback used', e);
        }
      }

      const payload = {
        farmer_id: currentUser?.id || 1,
        crop,
        variety: variety || undefined,
        growth_stage: growthStage,
        symptoms_description: symptoms || undefined,
        image_url: finalImageUrl,
        location: {
          latitude: lat,
          longitude: lng,
          district,
          address,
        },
      };

      const report = await api.createReport(payload);
      setResultReport(report);
      setStep(4);
    } catch (err) {
      console.error('Failed to submit crop report', err);
      alert('Failed to process report. Please ensure the backend is running.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReferToOfficer = async () => {
    if (!resultReport) return;
    setReferring(true);
    try {
      const updated = await api.referToExpert(resultReport.id);
      setResultReport(updated);
      setReferredSuccess(true);
    } catch (err) {
      console.error('Failed to refer to officer', err);
    } finally {
      setReferring(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs uppercase tracking-wider font-bold text-emerald-700">Check Crop Health</span>
            <h1 className="text-xl font-bold text-slate-900">
              {step === 1 && 'Step 1: Upload Crop Image'}
              {step === 2 && 'Step 2: Collect Crop Context & Location'}
              {step === 3 && 'Step 3: Processing Prototype Analysis'}
              {step === 4 && 'Step 4: Prototype Analysis & IPM Guidance'}
            </h1>
          </div>
          <button onClick={onCancel} className="text-xs font-semibold text-slate-500 hover:text-slate-700">
            Cancel
          </button>
        </div>

        <div className="grid grid-cols-4 gap-2 text-center text-xs font-medium border-t border-slate-100 pt-3">
          <div className={step >= 1 ? 'text-emerald-700 font-bold border-b-2 border-emerald-600 pb-1' : 'text-slate-400'}>
            1. Image
          </div>
          <div className={step >= 2 ? 'text-emerald-700 font-bold border-b-2 border-emerald-600 pb-1' : 'text-slate-400'}>
            2. Context
          </div>
          <div className={step >= 3 ? 'text-emerald-700 font-bold border-b-2 border-emerald-600 pb-1' : 'text-slate-400'}>
            3. Processing
          </div>
          <div className={step >= 4 ? 'text-emerald-700 font-bold border-b-2 border-emerald-600 pb-1' : 'text-slate-400'}>
            4. Result
          </div>
        </div>
      </div>

      {step === 1 && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-5">
          <div className="text-sm text-slate-600">
            Take a clear photo of the affected crop leaf, fruit, or plant area showing symptoms.
          </div>

          {!imagePreview ? (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              className="border-2 border-dashed border-slate-300 hover:border-emerald-500 rounded-xl p-8 text-center transition cursor-pointer bg-slate-50 hover:bg-emerald-50/50"
            >
              <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
              <div className="font-bold text-slate-800 text-base">Drag & drop your crop image here</div>
              <div className="text-xs text-slate-500 mt-1 mb-4">Supports JPG, PNG, WEBP up to 10MB</div>
              
              <label className="inline-block bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold px-4 py-2 rounded-lg cursor-pointer shadow transition">
                Browse Files
                <input type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
              </label>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative max-w-md mx-auto rounded-xl overflow-hidden border border-slate-200 shadow-md">
                <img src={imagePreview} alt="Preview" className="w-full h-64 object-cover" />
                <button
                  onClick={removeImage}
                  className="absolute top-3 right-3 bg-slate-900/80 hover:bg-slate-900 text-white p-1.5 rounded-full shadow"
                  title="Remove Image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="text-center text-xs text-emerald-700 font-medium flex items-center justify-center gap-1">
                <CheckCircle className="w-4 h-4 text-emerald-600" />
                <span>Image selected successfully. Proceed to collect context.</span>
              </div>
            </div>
          )}

          {imageError && (
            <div className="p-3 bg-rose-50 border border-rose-200 text-rose-800 text-xs rounded-lg flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{imageError}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-3 border-t border-slate-100 pt-4">
            <button
              onClick={() => {
                if (!imagePreview) {
                  setImagePreview('https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600&auto=format&fit=crop');
                }
                setStep(2);
              }}
              className="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold px-5 py-2 rounded-lg shadow text-sm flex items-center gap-1.5 transition"
            >
              <span>Next: Crop Context</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold uppercase text-slate-700 mb-1">Crop Type *</label>
                <select
                  value={crop}
                  onChange={(e) => setCrop(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-sm font-medium focus:ring-2 focus:ring-emerald-500 outline-none"
                >
                  <option value="Tomato">Tomato</option>
                  <option value="Cotton">Cotton</option>
                  <option value="Soybean">Soybean</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold uppercase text-slate-700 mb-1">Variety (Optional)</label>
                <input
                  type="text"
                  placeholder="e.g. Abhinav, Bt Cotton RCH 659, JS 335"
                  value={variety}
                  onChange={(e) => setVariety(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase text-slate-700 mb-1">Growth Stage *</label>
                <select
                  value={growthStage}
                  onChange={(e) => setGrowthStage(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-sm font-medium focus:ring-2 focus:ring-emerald-500 outline-none"
                >
                  <option value="Seedling">Seedling</option>
                  <option value="Vegetative">Vegetative</option>
                  <option value="Flowering">Flowering</option>
                  <option value="Fruiting">Fruiting</option>
                  <option value="Maturity">Maturity</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold uppercase text-slate-700 mb-1">District / Location Name</label>
                <input
                  type="text"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-bold uppercase text-slate-700 mb-1">Symptom Description (Optional)</label>
                <textarea
                  rows={3}
                  placeholder="Describe visible spots, leaf curling, larvae, pest presence, or yellowing..."
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="block text-xs font-bold uppercase text-slate-700 flex items-center justify-between">
                <span>Select Field Location on Map</span>
                <span className="text-[11px] font-normal text-slate-500">Nagpur District</span>
              </label>
              <div className="h-64 rounded-lg overflow-hidden border border-slate-300">
                <LeafletMap
                  mode="picker"
                  center={[lat, lng]}
                  selectedLat={lat}
                  selectedLng={lng}
                  onLocationSelect={(newLat, newLng) => {
                    setLat(newLat);
                    setLng(newLng);
                  }}
                />
              </div>
              <div className="text-[11px] text-slate-500 flex items-center gap-1">
                <MapPin className="w-3.5 h-3.5 text-emerald-600" />
                <span>Selected Coordinates: {lat.toFixed(4)}, {lng.toFixed(4)}</span>
              </div>
            </div>
          </div>

          <WeatherContextCard district={district} />

          <div className="flex items-center justify-between border-t border-slate-100 pt-4">

            <button
              onClick={() => setStep(1)}
              className="px-4 py-2 rounded-lg text-sm font-semibold border border-slate-300 text-slate-700 hover:bg-slate-50 flex items-center gap-1 transition"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back</span>
            </button>

            <button
              onClick={handleSubmitReport}
              disabled={isSubmitting}
              className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-6 py-2.5 rounded-lg shadow text-sm flex items-center gap-2 transition disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>Submit for Prototype Analysis</span>
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="bg-white p-12 rounded-xl border border-slate-200 shadow-sm text-center space-y-4">
          <div className="w-12 h-12 border-4 border-emerald-600 border-t-transparent rounded-full animate-spin mx-auto" />
          <h2 className="text-xl font-bold text-slate-900">Evaluating Prototype Risk Engine...</h2>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Combining visual image indicators, growth stage vulnerability, local weather suitability, and regional report signals.
          </p>
        </div>
      )}

      {step === 4 && resultReport && (
        <div className="space-y-6">
          <NoticeBanner message={resultReport.risk_assessment?.methodology_note} />

          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-5">
              <div>
                <div className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Prototype Analysis Result
                </div>
                <h2 className="text-2xl font-extrabold text-slate-900 mt-1">
                  {(resultReport.analysis?.confidence || 0) < 70 ? 'Possible Condition:' : 'Likely Condition:'}{' '}
                  <span className="text-emerald-800">{resultReport.analysis?.condition?.name}</span>
                </h2>
                <div className="text-xs text-slate-600 mt-1 flex items-center gap-3">
                  <span>Crop: <strong>{resultReport.crop}</strong></span>
                  <span>Stage: <strong>{resultReport.growth_stage}</strong></span>
                  <span>Condition Type: <strong className="uppercase">{resultReport.analysis?.condition?.type}</strong></span>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-xs font-semibold text-slate-500 uppercase">Analysis Confidence</div>
                  <div className="text-2xl font-extrabold text-slate-900">
                    {resultReport.analysis?.confidence?.toFixed(0)}%
                  </div>
                </div>
                <div className="border-l border-slate-200 pl-4 text-right">
                  <div className="text-xs font-semibold text-slate-500 uppercase">Contextual Risk Score</div>
                  <div className="mt-1">
                    <RiskBadge level={resultReport.risk_assessment?.risk_level} size="lg" />
                  </div>
                </div>
              </div>
            </div>

            {resultReport.analysis?.alternatives && resultReport.analysis.alternatives.length > 0 && (
              <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200 text-xs">
                <span className="font-bold text-slate-700 block mb-1">Alternative Conditions Evaluated:</span>
                <div className="flex flex-wrap gap-3">
                  {resultReport.analysis.alternatives.map((alt, idx) => (
                    <span key={idx} className="bg-white border border-slate-300 px-2.5 py-1 rounded font-medium text-slate-700">
                      {alt.name} ({alt.type}) — {alt.confidence}% confidence
                    </span>
                  ))}
                </div>
              </div>
            )}

            <WeatherContextCard district={resultReport.location?.district || district} />

            <div className="space-y-3">

              <h3 className="font-bold text-slate-900 text-base">Main Contributing Factors:</h3>
              <ul className="space-y-2">
                {resultReport.risk_assessment?.contributing_factors?.map((factor, idx) => (
                  <li key={idx} className="text-xs text-slate-700 flex items-start gap-2 bg-emerald-50/60 p-2.5 rounded border border-emerald-100">
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>

            {resultReport.advisory && (
              <div className="bg-emerald-900 text-white p-6 rounded-xl space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-lg text-emerald-100">Recommended IPM Actions</h3>
                  <span className="text-xs uppercase bg-emerald-800 text-emerald-200 border border-emerald-700 px-2.5 py-1 rounded font-mono">
                    Priority: {resultReport.advisory.priority}
                  </span>
                </div>

                <div className="space-y-2">
                  <span className="text-xs font-bold uppercase text-emerald-300 tracking-wider">Field Sanitation & Cultural Management</span>
                  <ul className="space-y-1.5">
                    {resultReport.advisory.actions.map((act, i) => (
                      <li key={i} className="text-xs text-emerald-50 flex items-start gap-2">
                        <span className="text-emerald-400 font-bold">•</span>
                        <span>{act}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-2 border-t border-emerald-800/80 pt-3">
                  <span className="text-xs font-bold uppercase text-emerald-300 tracking-wider">Field Surveillance & Monitoring</span>
                  <ul className="space-y-1.5">
                    {resultReport.advisory.monitoring.map((mon, i) => (
                      <li key={i} className="text-xs text-emerald-100 flex items-start gap-2">
                        <span className="text-emerald-400 font-bold">•</span>
                        <span>{mon}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <p className="text-[11px] text-emerald-200 italic border-t border-emerald-800/80 pt-2">
                  {resultReport.advisory.safety_note}
                </p>
              </div>
            )}

            {((resultReport.analysis?.confidence || 0) < 70 || referredSuccess || resultReport.advisory?.expert_referral) && (
              <div className="bg-amber-50 border border-amber-300 p-5 rounded-xl space-y-3">
                <div className="flex items-start gap-3">
                  <ShieldAlert className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-bold text-amber-950 text-sm">Expert Verification Recommended</h4>
                    <p className="text-xs text-amber-900 mt-0.5">
                      {(resultReport.analysis?.confidence || 0) < 70
                        ? 'Analysis confidence is below 70%. We recommend submitting this report to an Extension Officer for direct verification.'
                        : 'High risk level detected. Extension officer verification is recommended to confirm management actions.'}
                    </p>
                  </div>
                </div>

                {!referredSuccess && resultReport.status !== 'PENDING_VERIFICATION' ? (
                  <button
                    onClick={handleReferToOfficer}
                    disabled={referring}
                    className="w-full sm:w-auto bg-amber-700 hover:bg-amber-800 text-white font-bold px-5 py-2 rounded-lg text-xs shadow transition flex items-center justify-center gap-2"
                  >
                    <FileSearch className="w-4 h-4" />
                    <span>{referring ? 'Submitting to Officer...' : 'Submit for Extension Officer Verification'}</span>
                  </button>
                ) : (
                  <div className="p-2.5 bg-emerald-100 border border-emerald-300 text-emerald-900 text-xs rounded font-bold flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-emerald-700" />
                    <span>Case successfully referred to Extension Officer dashboard for review.</span>
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center justify-between border-t border-slate-100 pt-4">
              <button
                onClick={() => onComplete(resultReport)}
                className="bg-slate-900 hover:bg-slate-800 text-white font-bold px-6 py-2.5 rounded-lg text-sm transition"
              >
                Done / View in My Reports
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

import React, { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { FarmerDashboard } from './components/farmer/FarmerDashboard';
import { CheckCropWizard } from './components/farmer/CheckCropWizard';
import { MyReports } from './components/farmer/MyReports';
import { ReportDetailModal } from './components/farmer/ReportDetailModal';

import { OfficerDashboard } from './components/officer/OfficerDashboard';
import { VerificationTable } from './components/officer/VerificationTable';
import { HotspotsMapView } from './components/officer/HotspotsMapView';
import { VerificationModal } from './components/officer/VerificationModal';
import { DemoLoginModal } from './components/auth/DemoLoginModal';

import type { CropReport } from './types';

import { api } from './api/client';

const MainContent: React.FC = () => {
  const { activeRole } = useAuth();
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [showLoginModal, setShowLoginModal] = useState<boolean>(true);


  // Modal States
  const [selectedReport, setSelectedReport] = useState<CropReport | null>(null);
  const [officerModalReport, setOfficerModalReport] = useState<CropReport | null>(null);


  const handleOpenReportById = async (reportId: number) => {
    try {
      const rep = await api.getReportDetail(reportId);
      if (activeRole === 'EXTENSION_OFFICER') {
        setOfficerModalReport(rep);
      } else {
        setSelectedReport(rep);
      }
    } catch (e) {
      console.error('Error fetching report detail', e);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenLoginModal={() => setShowLoginModal(true)}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* FARMER VIEWS */}
        {activeRole === 'FARMER' && (
          <>
            {activeTab === 'dashboard' && (
              <FarmerDashboard
                onNavigate={setActiveTab}
                onSelectReport={(r) => setSelectedReport(r)}
              />
            )}

            {activeTab === 'check-crop' && (
              <CheckCropWizard
                onComplete={(r) => {
                  setSelectedReport(r);
                  setActiveTab('my-reports');
                }}
                onCancel={() => setActiveTab('dashboard')}
              />
            )}

            {activeTab === 'my-reports' && (
              <MyReports
                onSelectReport={(r) => setSelectedReport(r)}
                onNavigateToCheckCrop={() => setActiveTab('check-crop')}
              />
            )}
          </>
        )}

        {/* EXTENSION OFFICER VIEWS */}
        {activeRole === 'EXTENSION_OFFICER' && (
          <>
            {activeTab === 'dashboard' && (
              <OfficerDashboard
                onNavigate={setActiveTab}
                onOpenReport={(r) => setOfficerModalReport(r)}
              />
            )}

            {activeTab === 'verification-table' && (
              <VerificationTable
                onOpenReport={(r) => setOfficerModalReport(r)}
              />
            )}

            {activeTab === 'hotspot-map' && (
              <HotspotsMapView
                onSelectReport={handleOpenReportById}
              />
            )}
          </>
        )}
      </main>

      {/* Modals */}
      {showLoginModal && (
        <DemoLoginModal
          onSelectRole={() => {
            setActiveTab('dashboard');
            setShowLoginModal(false);
          }}
          onClose={() => setShowLoginModal(false)}
        />
      )}


      <ReportDetailModal
        report={selectedReport}
        onClose={() => setSelectedReport(null)}
      />

      <VerificationModal
        report={officerModalReport}
        onClose={() => setOfficerModalReport(null)}
        onUpdated={() => {
          // Trigger refresh if needed
        }}
      />

      <Footer />
    </div>
  );

};

export default function App() {
  return (
    <AuthProvider>
      <MainContent />
    </AuthProvider>
  );
}

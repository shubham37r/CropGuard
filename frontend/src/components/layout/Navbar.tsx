import { User, UserCheck, Sprout, LogIn } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onOpenLoginModal: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, onOpenLoginModal }) => {

  const { currentUser, activeRole, switchUserRole } = useAuth();

  return (
    <header className="bg-slate-900 text-white border-b border-slate-800 sticky top-0 z-50 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="bg-emerald-600 p-2 rounded-lg text-white shadow-sm flex items-center justify-center">
              <Sprout className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-xl tracking-tight text-white">CropGuard</span>
                <span className="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded font-mono uppercase">
                  SIH MVP
                </span>
              </div>
              <span className="text-xs text-slate-400 block -mt-1 hidden sm:block">
                Localized Crop Health Early Warning System
              </span>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-1">
            {activeRole === 'FARMER' ? (
              <>
                <button
                  onClick={() => setActiveTab('dashboard')}
                  className={`px-3.5 py-2 rounded-md text-sm font-medium transition ${
                    activeTab === 'dashboard' ? 'bg-emerald-700 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  Farmer Dashboard
                </button>
                <button
                  onClick={() => setActiveTab('check-crop')}
                  className={`px-3.5 py-2 rounded-md text-sm font-medium transition ${
                    activeTab === 'check-crop' ? 'bg-emerald-700 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  Check Crop
                </button>
                <button
                  onClick={() => setActiveTab('my-reports')}
                  className={`px-3.5 py-2 rounded-md text-sm font-medium transition ${
                    activeTab === 'my-reports' ? 'bg-emerald-700 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  My Reports
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => setActiveTab('dashboard')}
                  className={`px-3.5 py-2 rounded-md text-sm font-medium transition ${
                    activeTab === 'dashboard' ? 'bg-emerald-700 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  Officer Dashboard
                </button>
                <button
                  onClick={() => setActiveTab('verification-table')}
                  className={`px-3.5 py-2 rounded-md text-sm font-medium transition ${
                    activeTab === 'verification-table' ? 'bg-emerald-700 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  Report Verifications
                </button>
                <button
                  onClick={() => setActiveTab('hotspot-map')}
                  className={`px-3.5 py-2 rounded-md text-sm font-medium transition ${
                    activeTab === 'hotspot-map' ? 'bg-emerald-700 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  Geospatial Hotspot Map
                </button>
              </>
            )}
          </nav>

          <div className="flex items-center gap-3">
            <button
              onClick={onOpenLoginModal}
              className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition shadow-sm"
              title="Open Demo Role Selection Modal"
            >
              <LogIn className="w-3.5 h-3.5 text-emerald-400" />
              <span>Demo Login</span>
            </button>

            <div className="bg-slate-800 border border-slate-700 p-1 rounded-lg flex items-center gap-1">

              <button
                onClick={() => {
                  switchUserRole('FARMER');
                  setActiveTab('dashboard');
                }}
                className={`px-2.5 py-1 text-xs rounded-md font-semibold flex items-center gap-1.5 transition ${
                  activeRole === 'FARMER'
                    ? 'bg-emerald-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Switch to Farmer Role"
              >
                <User className="w-3.5 h-3.5" />
                Farmer
              </button>
              <button
                onClick={() => {
                  switchUserRole('EXTENSION_OFFICER');
                  setActiveTab('dashboard');
                }}
                className={`px-2.5 py-1 text-xs rounded-md font-semibold flex items-center gap-1.5 transition ${
                  activeRole === 'EXTENSION_OFFICER'
                    ? 'bg-blue-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Switch to Officer Role"
              >
                <UserCheck className="w-3.5 h-3.5" />
                Officer
              </button>
            </div>

            <div className="hidden lg:block text-right border-l border-slate-800 pl-3">
              <div className="text-xs font-semibold text-slate-200">{currentUser?.name}</div>
              <div className="text-[10px] text-slate-400">{currentUser?.email}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="md:hidden bg-slate-950 px-4 py-2 flex items-center justify-around border-t border-slate-800 text-xs font-medium">
        {activeRole === 'FARMER' ? (
          <>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`py-1 ${activeTab === 'dashboard' ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}
            >
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('check-crop')}
              className={`py-1 ${activeTab === 'check-crop' ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}
            >
              Check Crop
            </button>
            <button
              onClick={() => setActiveTab('my-reports')}
              className={`py-1 ${activeTab === 'my-reports' ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}
            >
              My Reports
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`py-1 ${activeTab === 'dashboard' ? 'text-blue-400 font-bold' : 'text-slate-400'}`}
            >
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('verification-table')}
              className={`py-1 ${activeTab === 'verification-table' ? 'text-blue-400 font-bold' : 'text-slate-400'}`}
            >
              Verifications
            </button>
            <button
              onClick={() => setActiveTab('hotspot-map')}
              className={`py-1 ${activeTab === 'hotspot-map' ? 'text-blue-400 font-bold' : 'text-slate-400'}`}
            >
              Hotspot Map
            </button>
          </>
        )}
      </div>
    </header>
  );
};

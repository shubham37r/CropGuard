import React, { createContext, useContext, useState, useEffect } from 'react';
import type { User, UserRole } from '../types';
import { api } from '../api/client';

interface AuthContextType {
  currentUser: User | null;
  activeRole: UserRole;
  switchUserRole: (role: UserRole) => Promise<void>;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const DEMO_EMAILS: Record<UserRole, string> = {
  FARMER: 'farmer@example.com',
  EXTENSION_OFFICER: 'officer@example.com',
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeRole, setActiveRole] = useState<UserRole>('FARMER');
  const [currentUser, setCurrentUser] = useState<User | null>({
    id: 1,
    email: 'farmer@example.com',
    name: 'Rajesh Patel',
    role: 'FARMER',
    region: 'Katol, Nagpur'
  });
  const [loading, setLoading] = useState<boolean>(false);

  const fetchUser = async (role: UserRole) => {
    setLoading(true);
    try {
      const email = DEMO_EMAILS[role];
      const user = await api.login(email);
      setCurrentUser(user);
      setActiveRole(role);
    } catch (err) {
      console.warn('Backend login fallback used', err);
      if (role === 'EXTENSION_OFFICER') {
        setCurrentUser({
          id: 4,
          email: 'officer@example.com',
          name: 'Dr. Anish Sharma',
          role: 'EXTENSION_OFFICER',
          region: 'Nagpur Zone'
        });
      } else {
        setCurrentUser({
          id: 1,
          email: 'farmer@example.com',
          name: 'Rajesh Patel',
          role: 'FARMER',
          region: 'Katol, Nagpur'
        });
      }
      setActiveRole(role);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUser('FARMER');
  }, []);

  const switchUserRole = async (newRole: UserRole) => {
    await fetchUser(newRole);
  };

  return (
    <AuthContext.Provider value={{ currentUser, activeRole, switchUserRole, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

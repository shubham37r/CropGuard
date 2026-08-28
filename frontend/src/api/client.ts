import axios from 'axios';
import type { CropReport, HotspotResponse, User } from '../types';

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const getBackendOrigin = (): string => {
  return API_BASE_URL.replace(/\/api\/?$/, '');
};

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});


export const api = {
  // Auth & Roles
  login: async (email: string): Promise<User> => {
    const res = await apiClient.post<User>('/auth/login', { email });
    return res.data;
  },
  getDemoUsers: async (): Promise<User[]> => {
    const res = await apiClient.get<User[]>('/auth/users');
    return res.data;
  },

  // Reports
  uploadImage: async (file: File): Promise<{ image_url: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post<{ image_url: string }>('/reports/upload-image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  createReport: async (payload: {
    farmer_id: number;
    crop: string;
    variety?: string;
    growth_stage: string;
    symptoms_description?: string;
    image_url: string;
    location: {
      latitude: number;
      longitude: number;
      district?: string;
      address?: string;
    };
  }): Promise<CropReport> => {
    const res = await apiClient.post<CropReport>('/reports/', payload);
    return res.data;
  },

  getReports: async (params?: { farmer_id?: number; status?: string; crop?: string }): Promise<CropReport[]> => {
    const res = await apiClient.get<CropReport[]>('/reports/', { params });
    return res.data;
  },

  getReportDetail: async (id: number): Promise<CropReport> => {
    const res = await apiClient.get<CropReport>(`/reports/${id}`);
    return res.data;
  },

  referToExpert: async (reportId: number): Promise<CropReport> => {
    const res = await apiClient.post<CropReport>(`/reports/${reportId}/refer-expert`);
    return res.data;
  },

  // Verification (Officer)
  verifyReport: async (
    reportId: number,
    officerId: number,
    status: 'CONFIRMED' | 'REJECTED' | 'NEEDS_MORE_INFO',
    officerNotes?: string
  ): Promise<CropReport> => {
    const res = await apiClient.post<CropReport>(`/verification/${reportId}`, {
      status,
      officer_notes: officerNotes,
    }, {
      params: { officer_id: officerId }
    });
    return res.data;
  },

  // Geospatial Hotspots
  getHotspots: async (): Promise<HotspotResponse> => {
    const res = await apiClient.get<HotspotResponse>('/hotspots/');
    return res.data;
  }
};

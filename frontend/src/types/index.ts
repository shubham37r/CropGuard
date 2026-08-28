export type UserRole = 'FARMER' | 'EXTENSION_OFFICER';

export interface User {
  id: number;
  email: string;
  name: string;
  role: UserRole;
  phone?: string;
  region?: string;
}

export type ReportStatus =
  | 'SUBMITTED'
  | 'ANALYZED'
  | 'PENDING_VERIFICATION'
  | 'CONFIRMED'
  | 'REJECTED'
  | 'NEEDS_MORE_INFO';

export type GrowthStage =
  | 'Seedling'
  | 'Vegetative'
  | 'Flowering'
  | 'Fruiting'
  | 'Maturity';

export interface LocationReport {
  latitude: number;
  longitude: number;
  district: string;
  address?: string;
  region?: string;
}

export interface ConditionItem {
  name: string;
  type: 'DISEASE' | 'PEST' | 'PHYSIOLOGICAL';
}

export interface AlternativeCondition {
  name: string;
  type: string;
  confidence: number;
}

export interface AnalysisResult {
  id?: number;
  condition: ConditionItem;
  confidence: number;
  alternatives: AlternativeCondition[];
  is_mock: string;
}

export interface ComponentScores {
  disease_confidence: number;
  weather_risk: number;
  stage_risk: number;
  outbreak_signal: number;
}

export interface RiskAssessment {
  id?: number;
  score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  component_scores: ComponentScores;
  contributing_factors: string[];
  methodology_note: string;
}

export interface Verification {
  id: number;
  status: 'PENDING_VERIFICATION' | 'CONFIRMED' | 'REJECTED' | 'NEEDS_MORE_INFO';
  officer_id?: number;
  officer_name?: string;
  officer_notes?: string;
  verified_at?: string;
}

export interface IPMAdvisory {
  priority: 'LOW' | 'MEDIUM' | 'HIGH';
  actions: string[];
  monitoring: string[];
  expert_referral: boolean;
  safety_note: string;
}

export interface CropReport {
  id: number;
  farmer_id: number;
  farmer_name?: string;
  crop: string;
  variety?: string;
  growth_stage: GrowthStage;
  symptoms_description?: string;
  image_url: string;
  status: ReportStatus;
  created_at: string;
  location: LocationReport;
  analysis?: AnalysisResult;
  risk_assessment?: RiskAssessment;
  verification?: Verification;
  advisory?: IPMAdvisory;
}

export interface HotspotPoint {
  report_id: number;
  crop: string;
  condition_name: string;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  latitude: number;
  longitude: number;
  address?: string;
  district: string;
  status: string;
}

export interface HotspotCluster {
  cluster_id: string;
  center_latitude: number;
  center_longitude: number;
  radius_km: number;
  high_risk_count: number;
  total_reports_count: number;
  dominant_crop: string;
  dominant_condition: string;
  district: string;
  title: string;
  description: string;
  report_ids: number[];
}

export interface HotspotResponse {
  points: HotspotPoint[];
  clusters: HotspotCluster[];
  methodology_note: string;
}

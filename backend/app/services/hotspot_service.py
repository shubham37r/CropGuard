import math
from typing import List, Dict, Any

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates approximate distance in kilometers between two GPS coordinates."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class HotspotService:
    """
    Simple radius-based spatial clustering service for identifying emerging report clusters around Nagpur.
    Designed so a proper geospatial risk model can replace it later.
    """

    RADIUS_KM = 15.0

    def compute_hotspots(self, reports: List[Any]) -> Dict[str, Any]:
        points = []
        high_risk_reports = []

        for r in reports:
            # Safely extract lat/lng and risk
            lat = r.location.latitude if r.location else 21.1458
            lon = r.location.longitude if r.location else 79.0882
            district = r.location.district if r.location else "Nagpur"
            address = r.location.address if r.location else f"{district} Sector"
            
            risk_lvl = r.risk_assessment.risk_level if r.risk_assessment else "MEDIUM"
            cond_name = r.analysis.condition_name if r.analysis else "Unspecified Condition"

            pt = {
                "report_id": r.id,
                "crop": r.crop,
                "condition_name": cond_name,
                "risk_level": risk_lvl,
                "latitude": lat,
                "longitude": lon,
                "address": address,
                "district": district,
                "status": r.status.value if hasattr(r.status, "value") else str(r.status)
            }
            points.append(pt)

            if risk_lvl == "HIGH":
                high_risk_reports.append(pt)

        # Simple Radius Clustering Algorithm
        clusters = []
        visited = set()

        for i, center_pt in enumerate(high_risk_reports):
            if center_pt["report_id"] in visited:
                continue

            cluster_members = [center_pt]
            visited.add(center_pt["report_id"])

            for j, neighbor_pt in enumerate(high_risk_reports):
                if neighbor_pt["report_id"] in visited:
                    continue
                
                dist = haversine_distance_km(
                    center_pt["latitude"], center_pt["longitude"],
                    neighbor_pt["latitude"], neighbor_pt["longitude"]
                )
                if dist <= self.RADIUS_KM:
                    cluster_members.append(neighbor_pt)
                    visited.add(neighbor_pt["report_id"])

            # If 2 or more high-risk reports in 15km radius, create emerging hotspot cluster
            if len(cluster_members) >= 2:
                avg_lat = sum(m["latitude"] for m in cluster_members) / len(cluster_members)
                avg_lon = sum(m["longitude"] for m in cluster_members) / len(cluster_members)
                
                crops = [m["crop"] for m in cluster_members]
                dominant_crop = max(set(crops), key=crops.count)
                
                conds = [m["condition_name"] for m in cluster_members]
                dominant_cond = max(set(conds), key=conds.count)

                clusters.append({
                    "cluster_id": f"hotspot-{i+1}",
                    "center_latitude": avg_lat,
                    "center_longitude": avg_lon,
                    "radius_km": self.RADIUS_KM,
                    "high_risk_count": len(cluster_members),
                    "total_reports_count": len(cluster_members),
                    "dominant_crop": dominant_crop,
                    "dominant_condition": dominant_cond,
                    "district": center_pt["district"],
                    "title": f"Emerging Hotspot: {dominant_cond} in {center_pt['district']}",
                    "description": f"High-risk report cluster of {len(cluster_members)} active cases detected within {self.RADIUS_KM}km radius.",
                    "report_ids": [m["report_id"] for m in cluster_members]
                })

        return {
            "points": points,
            "clusters": clusters,
            "methodology_note": "Simple radius-based prototype spatial clustering of high-risk report clusters."
        }

hotspot_service = HotspotService()

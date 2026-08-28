import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { HotspotPoint, HotspotCluster } from '../../types';

interface LeafletMapProps {
  mode: 'picker' | 'hotspots';
  center?: [number, number];
  zoom?: number;
  selectedLat?: number;
  selectedLng?: number;
  onLocationSelect?: (lat: number, lng: number) => void;
  points?: HotspotPoint[];
  clusters?: HotspotCluster[];
  onSelectPoint?: (reportId: number) => void;
}

export const LeafletMap: React.FC<LeafletMapProps> = ({
  mode,
  center = [21.1458, 79.0882],
  zoom = 10,
  selectedLat,
  selectedLng,
  onLocationSelect,
  points = [],
  clusters = [],
  onSelectPoint,
}) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<L.Map | null>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
      iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
    });
  }, []);

  useEffect(() => {
    if (!mapRef.current) return;
    if (leafletMapRef.current) return;

    const map = L.map(mapRef.current).setView(center, zoom);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(map);

    const layerGroup = L.layerGroup().addTo(map);
    leafletMapRef.current = map;
    layerGroupRef.current = layerGroup;

    return () => {
      map.remove();
      leafletMapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = leafletMapRef.current;
    const layerGroup = layerGroupRef.current;
    if (!map || !layerGroup) return;

    layerGroup.clearLayers();

    if (mode === 'picker') {
      const targetLat = selectedLat ?? center[0];
      const targetLng = selectedLng ?? center[1];

      const marker = L.marker([targetLat, targetLng], { draggable: true }).addTo(layerGroup);
      marker.bindPopup('<b>Selected Field Location</b><br/>Drag or click map to move').openPopup();

      marker.on('dragend', (e: any) => {
        const coord = e.target.getLatLng();
        if (onLocationSelect) onLocationSelect(coord.lat, coord.lng);
      });

      map.off('click');
      map.on('click', (e: L.LeafletMouseEvent) => {
        if (onLocationSelect) onLocationSelect(e.latlng.lat, e.latlng.lng);
      });
    } else if (mode === 'hotspots') {
      clusters.forEach((c) => {
        const circle = L.circle([c.center_latitude, c.center_longitude], {
          radius: c.radius_km * 1000,
          color: '#e11d48',
          fillColor: '#f43f5e',
          fillOpacity: 0.25,
          weight: 2,
          dashArray: '6, 6',
        }).addTo(layerGroup);

        circle.bindPopup(`
          <div style="font-size:12px;">
            <strong style="color:#be123c;">${c.title}</strong><br/>
            <p style="margin:4px 0;">${c.description}</p>
            <span>Dominant Pest/Disease: <b>${c.dominant_condition}</b> (${c.dominant_crop})</span>
          </div>
        `);
      });

      points.forEach((pt) => {
        let color = '#22c55e';
        if (pt.risk_level === 'MEDIUM') color = '#f59e0b';
        if (pt.risk_level === 'HIGH') color = '#e11d48';

        const customMarker = L.circleMarker([pt.latitude, pt.longitude], {
          radius: 8,
          fillColor: color,
          color: '#ffffff',
          weight: 2,
          opacity: 1,
          fillOpacity: 0.9,
        }).addTo(layerGroup);

        customMarker.bindPopup(`
          <div style="font-size:12px; min-width:160px;">
            <strong>${pt.crop} - ${pt.condition_name}</strong><br/>
            <span style="font-weight:bold; color:${color};">Risk: ${pt.risk_level}</span><br/>
            <span>Location: ${pt.address || pt.district}</span><br/>
            <span>Status: ${pt.status}</span><br/>
            <button id="btn-view-${pt.report_id}" style="margin-top:6px; background:#15803d; color:white; border:none; padding:4px 8px; border-radius:4px; font-size:11px; cursor:pointer;">
              View Report #${pt.report_id}
            </button>
          </div>
        `);

        customMarker.on('popupopen', () => {
          const btn = document.getElementById(`btn-view-${pt.report_id}`);
          if (btn && onSelectPoint) {
            btn.onclick = () => onSelectPoint(pt.report_id);
          }
        });
      });
    }
  }, [mode, selectedLat, selectedLng, points, clusters]);

  return (
    <div className="relative w-full h-full min-h-[300px] rounded-lg border border-slate-300 overflow-hidden shadow-inner">
      <div ref={mapRef} className="w-full h-full min-h-[300px] z-0" />
      {mode === 'picker' && (
        <div className="absolute bottom-2 left-2 bg-white/90 backdrop-blur px-2.5 py-1 rounded border border-slate-300 text-[11px] font-medium text-slate-700 z-10">
          📍 Lat: {selectedLat?.toFixed(4) ?? center[0].toFixed(4)}, Lng: {selectedLng?.toFixed(4) ?? center[1].toFixed(4)}
        </div>
      )}
    </div>
  );
};

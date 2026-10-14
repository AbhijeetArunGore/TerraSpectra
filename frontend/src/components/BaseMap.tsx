import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Map as MapLibreMap,
  NavigationControl,
  ScaleControl,
  AttributionControl,
  type StyleSpecification,
  type ErrorEvent,
} from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { ScatterplotLayer } from '@deck.gl/layers';

/**
 * Viewport configuration using standard geospatial convention:
 * Coordinates are formatted as [longitude, latitude] (EPSG:4326).
 * Longitude: East (+) / West (-) [-180 to +180]
 * Latitude: North (+) / South (-) [-90 to +90]
 */
export interface MapViewport {
  /** Longitude in decimal degrees (e.g. -121.6555 for Salinas, CA) */
  longitude: number;
  /** Latitude in decimal degrees (e.g. 36.6777 for Salinas, CA) */
  latitude: number;
  /** Zoom level (0 = whole world, 20 = building level) */
  zoom: number;
  /** Tilt angle in degrees (0 = top-down 2D, max 60 = 3D oblique) */
  pitch: number;
  /** Camera heading / rotation in degrees (0 = North-up) */
  bearing: number;
}

/**
 * Default demonstration extent: Salinas Valley, Monterey County, California, USA.
 * 
 * IMPORTANT GEOSPATIAL INTEGRITY NOTE:
 * This geographic coordinate is a regional demonstration extent where AVIRIS airborne campaigns
 * have historically flown. It is NOT the georeferenced boundary of the raw 512x217 Salinas .mat
 * raster cube from Days 1-6, which contains no genuine embedded CRS or affine transform.
 */
export const DEFAULT_VIEWPORT: MapViewport = {
  longitude: -121.6555,
  latitude: 36.6777,
  zoom: 10.5,
  pitch: 0,
  bearing: 0,
};

/** Pre-configured, reliable basemap styles with appropriate attribution */
export const BASEMAP_STYLES = {
  cartoDark: {
    id: 'cartoDark',
    name: 'CARTO Dark Matter (Token-Free)',
    url: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    attribution: '© OpenStreetMap contributors, © CARTO',
    requiresToken: false,
  },
  cartoPositron: {
    id: 'cartoPositron',
    name: 'CARTO Positron Light (Token-Free)',
    url: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    attribution: '© OpenStreetMap contributors, © CARTO',
    requiresToken: false,
  },
  osmRaster: {
    id: 'osmRaster',
    name: 'OpenStreetMap Standard (Raster)',
    url: {
      version: 8,
      sources: {
        'osm-tiles': {
          type: 'raster',
          tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
          tileSize: 256,
          attribution: '© OpenStreetMap contributors',
        },
      },
      layers: [
        {
          id: 'osm-tiles-layer',
          type: 'raster',
          source: 'osm-tiles',
          minzoom: 0,
          maxzoom: 19,
        },
      ],
    },
    attribution: '© OpenStreetMap contributors',
    requiresToken: false,
  },
};

export interface BaseMapProps {
  initialViewport?: Partial<MapViewport>;
  selectedStyleKey?: keyof typeof BASEMAP_STYLES;
  onViewportChange?: (viewport: MapViewport) => void;
}

export const BaseMap: React.FC<BaseMapProps> = ({
  initialViewport = DEFAULT_VIEWPORT,
  selectedStyleKey = 'cartoDark',
  onViewportChange,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  const [currentViewport, setCurrentViewport] = useState<MapViewport>({
    ...DEFAULT_VIEWPORT,
    ...initialViewport,
  });
  const [mapLoaded, setMapLoaded] = useState<boolean>(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Check for optional Mapbox token from environment
  const mapboxToken = import.meta.env.VITE_MAPBOX_ACCESS_TOKEN;
  const isMapboxTokenConfigured = Boolean(
    mapboxToken &&
    mapboxToken !== 'pk.your_mapbox_public_token_here' &&
    mapboxToken.startsWith('pk.')
  );

  const activeStyleConfig = BASEMAP_STYLES[selectedStyleKey] || BASEMAP_STYLES.cartoDark;

  // Initialize MapLibre GL map instance and Deck.gl MapboxOverlay
  useEffect(() => {
    if (!mapContainerRef.current) return;

    try {
      const initialLng = initialViewport.longitude ?? DEFAULT_VIEWPORT.longitude;
      const initialLat = initialViewport.latitude ?? DEFAULT_VIEWPORT.latitude;
      const initialZoom = initialViewport.zoom ?? DEFAULT_VIEWPORT.zoom;
      const initialPitch = initialViewport.pitch ?? DEFAULT_VIEWPORT.pitch;
      const initialBearing = initialViewport.bearing ?? DEFAULT_VIEWPORT.bearing;

      // Determine style specification
      const mapStyle = activeStyleConfig.url as string | StyleSpecification;

      const map = new MapLibreMap({
        container: mapContainerRef.current,
        style: mapStyle,
        center: [initialLng, initialLat],
        zoom: initialZoom,
        pitch: initialPitch,
        bearing: initialBearing,
        attributionControl: false, // Customized attribution control added explicitly
      });

      // Standard map navigation controls (zoom, pitch, compass)
      map.addControl(new NavigationControl({ visualizePitch: true }), 'top-right');
      map.addControl(new ScaleControl({ maxWidth: 100, unit: 'metric' }), 'bottom-left');
      map.addControl(
        new AttributionControl({
          compact: false,
          customAttribution: activeStyleConfig.attribution,
        }),
        'bottom-right'
      );

      // Initialize Deck.gl MapboxOverlay
      // Deck.gl synchronizes its WebGL camera with MapLibre's Web Mercator viewport
      const demonstrationMarkerLayer = new ScatterplotLayer({
        id: 'salinas-demo-regional-anchor',
        data: [
          {
            coordinates: [DEFAULT_VIEWPORT.longitude, DEFAULT_VIEWPORT.latitude],
            name: 'Salinas Valley Regional Center (Demonstration)',
          },
        ],
        getPosition: (d: any) => d.coordinates,
        getRadius: 1200, // 1.2km radius in meters
        getFillColor: [6, 182, 212, 180], // Cyan
        getLineColor: [255, 255, 255, 240],
        getLineWidth: 2,
        stroked: true,
        pickable: true,
      });

      const overlay = new MapboxOverlay({
        interleaved: false,
        layers: [demonstrationMarkerLayer],
      });

      map.addControl(overlay as any);

      map.on('load', () => {
        setMapLoaded(true);
        setLoadError(null);
      });

      map.on('error', (e: ErrorEvent) => {
        console.warn('MapLibre GL warning/error:', e);
        // If tile loading warning occurs, log cleanly without crashing the UI
      });

      const handleMove = () => {
        const center = map.getCenter();
        const updated: MapViewport = {
          longitude: Number(center.lng.toFixed(5)),
          latitude: Number(center.lat.toFixed(5)),
          zoom: Number(map.getZoom().toFixed(2)),
          pitch: Number(map.getPitch().toFixed(1)),
          bearing: Number(map.getBearing().toFixed(1)),
        };
        setCurrentViewport(updated);
        if (onViewportChange) {
          onViewportChange(updated);
        }
      };

      map.on('move', handleMove);

      mapRef.current = map;
      overlayRef.current = overlay;

      // Unmount cleanup: explicitly remove map and finalize Deck.gl overlay
      return () => {
        if (overlayRef.current && mapRef.current) {
          try {
            overlayRef.current.finalize();
          } catch (e) {
            // Ignore teardown errors
          }
        }
        if (mapRef.current) {
          mapRef.current.remove();
          mapRef.current = null;
        }
      };
    } catch (err: any) {
      console.error('Failed to initialize basemap:', err);
      setLoadError(err.message || 'Error initializing WebGL map canvas');
    }
  }, [selectedStyleKey]);

  // Handler to smoothly reset camera to initial demonstration extent
  const handleResetView = useCallback(() => {
    if (!mapRef.current) return;
    mapRef.current.flyTo({
      center: [DEFAULT_VIEWPORT.longitude, DEFAULT_VIEWPORT.latitude],
      zoom: DEFAULT_VIEWPORT.zoom,
      pitch: DEFAULT_VIEWPORT.pitch,
      bearing: DEFAULT_VIEWPORT.bearing,
      essential: true,
    });
  }, []);

  return (
    <div className="geospatial-map-wrapper" id="base-geographical-map">
      {/* Real-time Viewport & Coordinates HUD */}
      <div className="map-hud-overlay">
        <div className="hud-badge">
          <span className="hud-dot" />
          <span>{mapLoaded ? 'WGS84 (EPSG:4326) • Ready' : 'Initializing Basemap...'}</span>
        </div>
        <div className="hud-coordinates">
          <span><strong>Lng:</strong> {currentViewport.longitude}°</span>
          <span>•</span>
          <span><strong>Lat:</strong> {currentViewport.latitude}°</span>
          <span>•</span>
          <span><strong>Zoom:</strong> {currentViewport.zoom}</span>
          <span>•</span>
          <span><strong>Pitch:</strong> {currentViewport.pitch}°</span>
        </div>
        <button
          type="button"
          className="hud-reset-btn"
          onClick={handleResetView}
          title="Reset camera to Salinas demonstration center"
        >
          Reset View
        </button>
      </div>

      {/* Demonstration Extent Disclaimer Banner */}
      <div className="demo-disclaimer-banner">
        <span>📍 <strong>Demonstration Coordinates:</strong> Salinas Valley Extent ([-121.6555°, 36.6777°]). Raw Salinas hyperspectral cube pixels are non-georeferenced raster indices.</span>
      </div>

      {/* Missing Token Notification when user expects custom Mapbox */}
      {!isMapboxTokenConfigured && (
        <div className="token-status-pill" title="Public Mapbox token not detected; using token-free CARTO / OSM basemap.">
          <span>Active Basemap: {activeStyleConfig.name}</span>
        </div>
      )}

      {/* WebGL Error fallback container if hardware WebGL fails */}
      {loadError && (
        <div className="map-error-card">
          <h4>Basemap Initialization Issue</h4>
          <p>{loadError}</p>
          <small>Check WebGL browser support or network connectivity to tile servers.</small>
        </div>
      )}

      {/* MapLibre Canvas Container */}
      <div
        ref={mapContainerRef}
        className="maplibre-container"
        style={{ width: '100%', height: '100%', position: 'absolute', inset: 0 }}
      />
    </div>
  );
};

export default BaseMap;

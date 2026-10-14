# Day 7: Geographical Base Map Architecture & Foundation

This document details the architectural decisions, coordinate reference systems (CRS), Deck.gl integration, and geospatial considerations implemented for Day 7 of TerraSpectra.

---

## 1. Overview & Objective

Day 7 establishes the **geographical visualization foundation** for TerraSpectra, bridging the gap between raw sensor raster processing (Days 1–5), 2D Cartesian orthographic views (Day 6), and true spherical Web Mercator planetary visualization.

### Key Capabilities
- **High-Performance Vector/Raster Basemap**: Rendered with MapLibre GL JS (open-source fork of Mapbox GL).
- **Deck.gl `MapboxOverlay` Synchronization**: Deck.gl layers synchronize camera matrices, pitch, bearing, and zoom with the underlying Web Mercator basemap.
- **Dynamic Viewport HUD**: Real-time readouts of longitude, latitude, zoom, and pitch.
- **Attribution & Licensure**: Transparent display of OpenStreetMap and CARTO attributions.
- **Bi-directional Mode Switching**: Instant toggling between Day 7 Geographical Base Map and Day 6 Orthographic Raster Pixel Grid.

---

## 2. Coordinate Systems: Geographical vs. Raster Array

A fundamental pitfall in hyperspectral engineering is treating raw array rows and columns as latitude and longitude.

| Metric | Raw Hyperspectral Scene (Days 1–6) | Geographical Base Map (Day 7) |
| :--- | :--- | :--- |
| **Coordinate System** | 2D Cartesian Pixel Grid | WGS84 Geographic (`EPSG:4326`) |
| **Projection** | None (Orthographic Camera) | Web Mercator (`EPSG:3857`) |
| **Horizontal Axis ($X$)** | Array Column index: `0 .. 216` | Longitude: `[-180.0° .. +180.0°]` |
| **Vertical Axis ($Y$)** | Array Row index: `0 .. 511` | Latitude: `[-85.05° .. +85.05°]` |
| **Coordinate Order** | `[col, row]` | `[longitude, latitude]` |
| **Demonstration Anchor** | Raw flight line unreferenced pixels | Salinas Valley Center `[-121.6555°, 36.6777°]` |
| **Georeferencing Status**| Unreferenced (no CRS/affine in `.mat`)| Standard Web Mercator Tiling |

> **Important**: The Salinas `.mat` files (`Salinas_corrected.mat`, `Salinas_gt.mat`) contain **no embedded geospatial CRS, affine transform, or GPS polygon**. Salinas raw pixels are therefore NOT yet mapped onto real Earth GPS coordinates. The `[-121.6555°, 36.6777°]` location is a regional demonstration anchor representing the general Salinas Valley, California agricultural region.

---

## 3. Technology & Integration Choices

### Why MapLibre GL + Deck.gl `MapboxOverlay`?
1. **Zero Proprietary Lock-In**: MapLibre GL is completely open-source and free of proprietary access token requirements.
2. **Token-Free Default Basemaps**: Supports CARTO Dark Matter (`cartoDark`), CARTO Positron (`cartoPositron`), and OpenStreetMap Standard raster tiles out of the box with zero broken tiles.
3. **Optional Mapbox Support**: Users who configure `VITE_MAPBOX_ACCESS_TOKEN` in `.env` can load custom `mapbox://` styles without altering code.
4. **Standard Deck.gl Control**: `@deck.gl/mapbox`'s `MapboxOverlay` implements the standard `IControl` interface compatible with both Mapbox GL and MapLibre GL.

---

## 4. Environment Variables (`frontend/.env.example`)

```bash
# Optional: Mapbox Access Token (pk.*)
# If provided, custom Mapbox styles can be loaded.
# If omitted, CARTO Dark Matter / OpenStreetMap tiles are used automatically.
VITE_MAPBOX_ACCESS_TOKEN=pk.your_mapbox_public_token_here

# Optional: Custom Basemap Style JSON URL
VITE_BASEMAP_STYLE_URL=
```

> **Security Note**: All Vite environment variables prefixed with `VITE_` are bundled into the client-side JavaScript bundle and are visible to end users. Never place private secrets or unrestricted cloud API keys in frontend `.env` files.

---

## 5. Viewport Configuration Convention

Coordinates in Deck.gl and GeoJSON follow the standard `[longitude, latitude]` Cartesian array order:

```typescript
export interface MapViewport {
  longitude: number; // -121.6555 (West is negative)
  latitude: number;  // 36.6777   (North is positive)
  zoom: number;      // 10.5
  pitch: number;     // 0 degrees (top-down)
  bearing: number;   // 0 degrees (North-up)
}
```

---

## 6. How to Run & Interact

```bash
cd frontend
npm install
npm run dev
```

1. **Pan**: Click and drag with mouse left-click.
2. **Zoom**: Mouse scroll wheel or the `+` / `-` navigation buttons.
3. **Pitch / Rotate**: Right-click and drag, or use the compass control.
4. **Reset**: Click the **Reset View** button in the HUD to fly back to Salinas Valley.
5. **Change Style**: Select CARTO Dark Matter, CARTO Positron, or OpenStreetMap from the sidebar.
6. **Compare Systems**: Switch between **Geographical Base Map (Day 7)** and **Orthographic Raster Scene (Day 6)** in the View Mode dropdown.

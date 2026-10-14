import React, { useState } from 'react';
import { BaseMap, BASEMAP_STYLES, DEFAULT_VIEWPORT } from './components/BaseMap';
import { HyperspectralViewer, VisualizationMode, SALINAS_PALETTE } from './components/HyperspectralViewer';

type AppViewMode = 'geographical' | 'orthographic';

export const App: React.FC = () => {
  const [appViewMode, setAppViewMode] = useState<AppViewMode>('geographical');
  const [selectedBasemapStyle, setSelectedBasemapStyle] = useState<keyof typeof BASEMAP_STYLES>('cartoDark');
  const [rasterVisMode, setRasterVisMode] = useState<VisualizationMode>('classification');

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">TS</div>
          <div>
            <h1 className="brand-title">TerraSpectra</h1>
            <span className="brand-tag">Day 7 • Geographical Base Map Foundation</span>
          </div>
        </div>

        <div className="header-badges">
          <div className="status-badge">
            <span className="status-dot"></span>
            <span>
              {appViewMode === 'geographical'
                ? 'Geographical Basemap (Deck.gl + MapLibre)'
                : 'Orthographic Pixel Array (Cartesian 2D)'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Sidebar Controls */}
        <aside className="sidebar">
          {/* View Mode Toggle */}
          <div className="card">
            <h2 className="card-title">View Mode</h2>
            <div className="control-group">
              <label htmlFor="app-view-select" className="control-label">Visualization System</label>
              <select
                id="app-view-select"
                className="select-control"
                value={appViewMode}
                onChange={(e) => setAppViewMode(e.target.value as AppViewMode)}
              >
                <option value="geographical">Geographical Base Map (Day 7)</option>
                <option value="orthographic">Orthographic Raster Scene (Day 6)</option>
              </select>
            </div>
            <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem' }}>
              {appViewMode === 'geographical'
                ? 'True geographic coordinates (WGS84 [lng, lat]) projected in Web Mercator.'
                : 'Raw sensor pixel grid (rows 0..511, cols 0..216) in Cartesian space.'}
            </p>
          </div>

          {/* Geographical Mode Settings */}
          {appViewMode === 'geographical' ? (
            <>
              {/* Basemap Style Selector */}
              <div className="card">
                <h2 className="card-title">Basemap Provider</h2>
                <div className="control-group">
                  <label htmlFor="basemap-style-select" className="control-label">Tile Style</label>
                  <select
                    id="basemap-style-select"
                    className="select-control"
                    value={selectedBasemapStyle}
                    onChange={(e) => setSelectedBasemapStyle(e.target.value as keyof typeof BASEMAP_STYLES)}
                  >
                    {Object.entries(BASEMAP_STYLES).map(([key, item]) => (
                      <option key={key} value={key}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  <span><strong>Attribution:</strong> {BASEMAP_STYLES[selectedBasemapStyle].attribution}</span>
                </div>
              </div>

              {/* Geographical Extent Information */}
              <div className="card">
                <h2 className="card-title">Geographic Extent</h2>
                <div className="meta-table">
                  <div className="meta-row">
                    <span className="meta-label">Demonstration Focus</span>
                    <span className="meta-value">Salinas Valley, CA</span>
                  </div>
                  <div className="meta-row">
                    <span className="meta-label">Center Longitude</span>
                    <span className="meta-value">{DEFAULT_VIEWPORT.longitude}° W</span>
                  </div>
                  <div className="meta-row">
                    <span className="meta-label">Center Latitude</span>
                    <span className="meta-value">{DEFAULT_VIEWPORT.latitude}° N</span>
                  </div>
                  <div className="meta-row">
                    <span className="meta-label">Coordinate System</span>
                    <span className="meta-value">WGS84 (EPSG:4326)</span>
                  </div>
                  <div className="meta-row">
                    <span className="meta-label">Display Projection</span>
                    <span className="meta-value">Web Mercator (3857)</span>
                  </div>
                  <div className="meta-row">
                    <span className="meta-label">Overlay Engine</span>
                    <span className="meta-value">Deck.gl MapboxOverlay</span>
                  </div>
                </div>
              </div>

              {/* Geospatial Correctness Card */}
              <div className="card" style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                <h2 className="card-title">Geospatial Integrity Note</h2>
                <p style={{ marginBottom: '0.5rem' }}>
                  The raw Salinas hyperspectral cube contains <strong>512 × 217 raster array pixels</strong> without embedded GPS tags.
                </p>
                <p>
                  Today's milestone renders a genuine geographical basemap centered on the regional Salinas Valley demonstration extent.
                  Raw pixels will only be mapped after rigorous georeferencing in subsequent milestones.
                </p>
              </div>
            </>
          ) : (
            /* Orthographic Day 6 Controls */
            <>
              <div className="card">
                <h2 className="card-title">Raster Visualization Layer</h2>
                <div className="control-group">
                  <label htmlFor="vis-layer-select" className="control-label">Render Mode</label>
                  <select
                    id="vis-layer-select"
                    className="select-control"
                    value={rasterVisMode}
                    onChange={(e) => setRasterVisMode(e.target.value as VisualizationMode)}
                  >
                    <option value="classification">Ground-Truth Classification (16 Classes)</option>
                    <option value="false_color">False-Color Composite (NIR / Red / Green)</option>
                    <option value="pca1">PCA Component 1 Intensity</option>
                  </select>
                </div>
              </div>

              {rasterVisMode === 'classification' && (
                <div className="card">
                  <h2 className="card-title">Crop Classes (Salinas)</h2>
                  <div className="class-legend">
                    {Object.entries(SALINAS_PALETTE).map(([id, item]) => (
                      <div key={id} className="legend-item">
                        <span
                          className="legend-swatch"
                          style={{
                            backgroundColor: `rgb(${item.rgb.join(',')})`,
                          }}
                        ></span>
                        <span style={{ color: '#cbd5e1' }}>{item.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </aside>

        {/* Visualizer Viewport */}
        <main className="viewport-container">
          {appViewMode === 'geographical' ? (
            <BaseMap selectedStyleKey={selectedBasemapStyle} />
          ) : (
            <HyperspectralViewer mode={rasterVisMode} />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;

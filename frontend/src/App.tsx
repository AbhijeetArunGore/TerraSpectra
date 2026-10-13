import React, { useState } from 'react';
import { HyperspectralViewer, VisualizationMode, SALINAS_PALETTE } from './components/HyperspectralViewer';

export const App: React.FC = () => {
  const [visMode, setVisMode] = useState<VisualizationMode>('classification');

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">TS</div>
          <div>
            <h1 className="brand-title">TerraSpectra</h1>
            <span className="brand-tag">Day 6 • Deck.gl Scaffold</span>
          </div>
        </div>

        <div className="header-badges">
          <div className="status-badge">
            <span className="status-dot"></span>
            <span>Deck.gl Active (Orthographic 2D)</span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Sidebar Controls */}
        <aside className="sidebar">
          {/* Layer Selector */}
          <div className="card">
            <h2 className="card-title">Visualization Layer</h2>
            <div className="control-group">
              <label htmlFor="vis-layer-select" className="control-label">Render Mode</label>
              <select
                id="vis-layer-select"
                className="select-control"
                value={visMode}
                onChange={(e) => setVisMode(e.target.value as VisualizationMode)}
              >
                <option value="classification">Ground-Truth Classification (16 Classes)</option>
                <option value="false_color">False-Color Composite (NIR / Red / Green)</option>
                <option value="pca1">PCA Component 1 Intensity</option>
              </select>
            </div>
          </div>

          {/* Dataset Metadata */}
          <div className="card">
            <h2 className="card-title">Salinas Scene Metadata</h2>
            <div className="meta-table">
              <div className="meta-row">
                <span className="meta-label">Spatial Array</span>
                <span className="meta-value">512 × 217 Pixels</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Spectral Bands</span>
                <span className="meta-value">204 Bands</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Total Pixels</span>
                <span className="meta-value">111,104 Pixels</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Projection</span>
                <span className="meta-value">Pixel Grid (Non-GPS)</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">PCA Components</span>
                <span className="meta-value">15 (99.98% Variance)</span>
              </div>
              <div className="meta-row">
                <span className="meta-label">Memory Footprint</span>
                <span className="meta-value">43.34 MB</span>
              </div>
            </div>
          </div>

          {/* Classification Legend */}
          {visMode === 'classification' && (
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

          {/* Architectural Notes */}
          <div className="card" style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            <h2 className="card-title">Geospatial Integrity Note</h2>
            <p>
              Salinas data lacks native georeferencing (no EPSG or affine matrix in <code>.mat</code>).
              Deck.gl uses an <strong>OrthographicView</strong> mapped directly to array row/column coordinates
              without fabricated GPS alignment.
            </p>
          </div>
        </aside>

        {/* Visualizer Canvas */}
        <main className="viewport-container">
          <HyperspectralViewer mode={visMode} />
        </main>
      </div>
    </div>
  );
};

export default App;

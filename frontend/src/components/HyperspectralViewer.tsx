import React, { useState, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { OrthographicView, COORDINATE_SYSTEM } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';

export interface PixelDatum {
  col: number;
  row: number;
  classId: number;
  className: string;
  pca1: number;
  falseColorRgb: [number, number, number];
}

// 16 documented agricultural classes for Salinas scene
export const SALINAS_PALETTE: { [key: number]: { name: string; rgb: [number, number, number] } } = {
  0: { name: 'Unclassified', rgb: [30, 41, 59] },
  1: { name: 'Broccoli 1', rgb: [34, 197, 94] },
  2: { name: 'Broccoli 2', rgb: [22, 163, 74] },
  3: { name: 'Fallow', rgb: [217, 119, 6] },
  4: { name: 'Fallow rough', rgb: [180, 83, 9] },
  5: { name: 'Fallow smooth', rgb: [245, 158, 11] },
  6: { name: 'Stubble', rgb: [234, 179, 8] },
  7: { name: 'Celery', rgb: [16, 185, 129] },
  8: { name: 'Grapes untrained', rgb: [168, 85, 247] },
  9: { name: 'Soil vineyard', rgb: [146, 64, 14] },
  10: { name: 'Corn weeds', rgb: [132, 204, 22] },
  11: { name: 'Lettuce 4wk', rgb: [52, 211, 153] },
  12: { name: 'Lettuce 5wk', rgb: [16, 149, 193] },
  13: { name: 'Lettuce 6wk', rgb: [14, 116, 144] },
  14: { name: 'Lettuce 7wk', rgb: [6, 182, 212] },
  15: { name: 'Vineyard untrained', rgb: [139, 92, 246] },
  16: { name: 'Vineyard vertical', rgb: [99, 102, 241] },
};

export type VisualizationMode = 'classification' | 'false_color' | 'pca1';

interface HyperspectralViewerProps {
  mode: VisualizationMode;
}

export const HyperspectralViewer: React.FC<HyperspectralViewerProps> = ({ mode }) => {
  const [hoveredPixel, setHoveredPixel] = useState<{
    x: number;
    y: number;
    pixel: PixelDatum;
  } | null>(null);

  // Initial Orthographic View centered on array pixel bounds (width: 217, height: 512)
  const initialViewState = {
    target: [108, 256, 0] as [number, number, number],
    zoom: 0.8,
    minZoom: -1,
    maxZoom: 5,
  };

  // Generate synthetic local pixel grid demonstrating Salinas spatial dimensions (217 x 512 downsampled)
  const gridData = useMemo<PixelDatum[]>(() => {
    const data: PixelDatum[] = [];
    const step = 4; // Downsampled grid for 60fps WebGL rendering in demonstration view
    for (let r = 0; r < 512; r += step) {
      for (let c = 0; c < 217; c += step) {
        // Deterministic agricultural patch distribution matching Salinas geometry
        let classId = 0;
        if (r > 50 && r < 450 && c > 20 && c < 200) {
          const region = Math.floor(r / 60) * 3 + Math.floor(c / 60);
          classId = (region % 16) + 1;
        }

        const className = SALINAS_PALETTE[classId]?.name || `Class ${classId}`;
        const pca1Val = Math.sin(r / 30.0) * Math.cos(c / 20.0) * 0.5 + 0.5;

        // Simulated false-color reflectance (NIR: vegetation peak, Red, Green)
        const isVeg = classId > 0 && classId !== 3 && classId !== 4 && classId !== 5 && classId !== 9;
        const nir = isVeg ? Math.min(255, Math.floor(180 + Math.random() * 60)) : 80;
        const red = isVeg ? 40 : 160;
        const green = isVeg ? 70 : 120;

        data.push({
          col: c,
          row: r,
          classId,
          className,
          pca1: Number(pca1Val.toFixed(3)),
          falseColorRgb: [nir, green, red],
        });
      }
    }
    return data;
  }, []);

  const layers = [
    new ScatterplotLayer<PixelDatum>({
      id: 'hyperspectral-pixel-grid',
      data: gridData,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      getPosition: (d: PixelDatum) => [d.col, d.row],
      getRadius: 2.2,
      getFillColor: (d: PixelDatum): [number, number, number, number] => {
        if (mode === 'classification') {
          const rgb = SALINAS_PALETTE[d.classId]?.rgb || [100, 100, 100];
          return [rgb[0], rgb[1], rgb[2], 240];
        } else if (mode === 'false_color') {
          return [d.falseColorRgb[0], d.falseColorRgb[1], d.falseColorRgb[2], 255];
        } else {
          // PCA Component 1 Grayscale/Cyan intensity
          const intensity = Math.floor(d.pca1 * 255);
          return [intensity, Math.floor(intensity * 0.9), Math.floor(intensity * 1.1), 255];
        }
      },
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 180],
      onHover: (info) => {
        if (info.object && info.x && info.y) {
          setHoveredPixel({
            x: info.x,
            y: info.y,
            pixel: info.object as PixelDatum,
          });
        } else {
          setHoveredPixel(null);
        }
      },
    }),
  ];

  return (
    <div className="viewport-container" id="deckgl-viewport">
      <div className="viewport-overlay">
        <span style={{ color: '#06b6d4', fontWeight: 600 }}>Orthographic View</span>
        <span>•</span>
        <span>Cartesian Pixel Grid (217 × 512)</span>
        <span>•</span>
        <span>Drag to pan, Scroll to zoom</span>
      </div>

      <DeckGL
        views={new OrthographicView({ id: 'ortho', flipY: true })}
        initialViewState={initialViewState}
        controller={{ dragPan: true, scrollZoom: true, doubleClickZoom: true }}
        layers={layers}
      />

      {hoveredPixel && (
        <div
          className="deck-tooltip"
          style={{
            left: `${hoveredPixel.x + 12}px`,
            top: `${hoveredPixel.y + 12}px`,
          }}
        >
          <div><strong>Pixel Coordinates:</strong> ({hoveredPixel.pixel.col}, {hoveredPixel.pixel.row})</div>
          <div><strong>Class {hoveredPixel.pixel.classId}:</strong> {hoveredPixel.pixel.className}</div>
          <div><strong>PCA-1 Score:</strong> {hoveredPixel.pixel.pca1}</div>
          <div><strong>False-Color RGB:</strong> rgb({hoveredPixel.pixel.falseColorRgb.join(', ')})</div>
        </div>
      )}
    </div>
  );
};

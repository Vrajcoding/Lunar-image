import { useState } from "react";
import MetricsPanel from "./MetricsPanel";
import OverlayToggle from "./OverlayToggle";
import MatchPointsCanvas from "./MatchPointsCanvas";

export default function ResultView({ result, sourcePreview, referencePreview }) {
  const [activeTab, setActiveTab] = useState("overlay"); // "overlay" | "difference" | "matchpoints"

  if (!result) return null;

  const metadata = result.input_metadata || {};
  const srcMeta = metadata.source || {};
  const refMeta = metadata.reference || {};

  return (
    <div className="result-section">
      {/* Three Image Side-by-Side Comparison */}
      <div className="glass-card">
        <div className="section-title">
          <span>Lunar Image Registration Overview</span>
          {result.method && (
            <span className="method-pill">Method: {result.method}</span>
          )}
        </div>

        <div className="image-row">
          <div className="img-card">
            <div className="img-card-header">
              <span>Source (Chandrayaan-2)</span>
            </div>
            <div className="img-card-body">
              <img src={sourcePreview} alt="Source Orbit Tile" />
            </div>
          </div>

          <div className="img-card">
            <div className="img-card-header">
              <span>Reference (Target Map)</span>
            </div>
            <div className="img-card-body">
              <img src={referencePreview} alt="Reference Map Tile" />
            </div>
          </div>

          <div className="img-card">
            <div className="img-card-header">
              <span>Registered (Aligned Result)</span>
            </div>
            <div className="img-card-body">
              <img src={result.registered_image_url} alt="Registered Result" />
            </div>
          </div>
        </div>
      </div>

      {/* Visual Quality Inspection (Tabs: Blend Overlay / Difference Heatmap / Match Points) */}
      <div className="glass-card">
        <div className="inspection-header">
          <div className="section-title" style={{ margin: 0 }}>
            Visual Quality Inspection
          </div>

          <div className="tab-pill-group">
            <button
              onClick={() => setActiveTab("overlay")}
              className={`tab-pill-btn ${activeTab === "overlay" ? "active" : ""}`}
            >
              Blend Overlay
            </button>
            <button
              onClick={() => setActiveTab("difference")}
              className={`tab-pill-btn ${activeTab === "difference" ? "active" : ""}`}
            >
              Difference Heatmap
            </button>
            <button
              onClick={() => setActiveTab("matchpoints")}
              className={`tab-pill-btn ${activeTab === "matchpoints" ? "active" : ""}`}
            >
              Correspondence Canvas
            </button>
          </div>
        </div>

        {activeTab === "overlay" && (
          <OverlayToggle
            registeredUrl={result.registered_image_url}
            referenceUrl={referencePreview}
          />
        )}

        {activeTab === "difference" && (
          <div className="difference-view-container">
            <div className="diff-image-wrapper">
              <img
                src={result.difference_image_url || result.registered_image_url}
                alt="Alignment Difference Heatmap"
                className="diff-image"
              />
            </div>
            <div className="diff-caption">
              <span>🔍 Pixel-wise alignment residual: <code>|I_ref - I_registered|</code> (Darker/Cooler = perfect alignment, Brighter = residual discrepancy).</span>
            </div>
          </div>
        )}

        {activeTab === "matchpoints" && (
          <MatchPointsCanvas
            imageUrl={result.registered_image_url}
            matchPointsUrl={result.match_points_url}
          />
        )}
      </div>

      {/* Chandrayaan-2 Mission Metadata Panel */}
      {(srcMeta.instrument || srcMeta.sensor || srcMeta.product_id || srcMeta.source_format) && (
        <div className="glass-card">
          <div className="section-title">Chandrayaan-2 Mission Metadata</div>
          <div className="metadata-grid">
            <div className="meta-card">
              <span className="meta-key">Payload Instrument</span>
              <span className="meta-val">{srcMeta.instrument || srcMeta.selected_sensor || "TMC-2"}</span>
            </div>
            <div className="meta-card">
              <span className="meta-key">Sensor / Camera</span>
              <span className="meta-val">{srcMeta.sensor || "Panchromatic"}</span>
            </div>
            <div className="meta-card">
              <span className="meta-key">Product ID</span>
              <span className="meta-val">{srcMeta.product_id?.filename || srcMeta.product_id?.product_id || "Standard Raster Tile"}</span>
            </div>
            <div className="meta-card">
              <span className="meta-key">Format / Compression</span>
              <span className="meta-val">{srcMeta.source_format?.toUpperCase() || "GEOTIFF"} ({srcMeta.original_dtype || "uint8"})</span>
            </div>
            {srcMeta.acquisition_time && (
              <div className="meta-card">
                <span className="meta-key">Acquisition Timestamp</span>
                <span className="meta-val">{srcMeta.acquisition_time}</span>
              </div>
            )}
            {srcMeta.sun_elevation !== null && srcMeta.sun_elevation !== undefined && (
              <div className="meta-card">
                <span className="meta-key">Sun Elevation</span>
                <span className="meta-val">{srcMeta.sun_elevation}°</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Metrics Section */}
      <div className="glass-card">
        <div className="section-title">Registration Performance Metrics</div>
        <MetricsPanel metrics={result.metrics} />
      </div>

      {/* Export Results */}
      <div className="glass-card" style={{ textAlign: "center" }}>
        <div className="section-title" style={{ justifyContent: "center" }}>
          Export Results & Scientific Data
        </div>
        <div className="downloads-bar">
          <a href={result.registered_image_url} download className="btn-download">
            📥 Download Registered Image (.png)
          </a>
          {result.difference_image_url && (
            <a href={result.difference_image_url} download className="btn-download">
              🔬 Download Difference Map (.png)
            </a>
          )}
          <a href={result.match_points_url} download className="btn-download">
            📊 Download Match Points (.csv)
          </a>
        </div>
      </div>
    </div>
  );
}

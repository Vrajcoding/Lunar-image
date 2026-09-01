import { useState } from "react";
import MetricsPanel from "./MetricsPanel";
import OverlayToggle from "./OverlayToggle";
import MatchPointsCanvas from "./MatchPointsCanvas";

export default function ResultView({ result, sourcePreview, referencePreview }) {
  const [activeTab, setActiveTab] = useState("overlay"); // "overlay" | "matchpoints"

  if (!result) return null;

  return (
    <div className="result-section">
      {/* Three Image Side-by-Side View */}
      <div className="glass-card">
        <div className="section-title">Image Registration Comparison</div>
        <div className="image-row">
          <div className="img-card">
            <div className="img-card-header">
              <span>Source (Chandrayaan-2)</span>
            </div>
            <div className="img-card-body">
              <img src={sourcePreview} alt="Source" />
            </div>
          </div>

          <div className="img-card">
            <div className="img-card-header">
              <span>Reference (Lunar Map)</span>
            </div>
            <div className="img-card-body">
              <img src={referencePreview} alt="Reference" />
            </div>
          </div>

          <div className="img-card">
            <div className="img-card-header">
              <span>Registered (Aligned)</span>
            </div>
            <div className="img-card-body">
              <img src={result.registered_image_url} alt="Registered Result" />
            </div>
          </div>
        </div>
      </div>

      {/* Visual Alignment Tools (Tabs: Overlay Slider & Match Points Canvas) */}
      <div className="glass-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "1rem" }}>
          <div className="section-title" style={{ margin: 0 }}>Visual Quality Inspection</div>
          
          <div style={{ display: "flex", gap: "0.5rem", background: "rgba(15, 23, 42, 0.6)", padding: "0.25rem", borderRadius: "999px", border: "1px solid var(--border-color)" }}>
            <button
              onClick={() => setActiveTab("overlay")}
              style={{
                background: activeTab === "overlay" ? "var(--primary)" : "transparent",
                color: activeTab === "overlay" ? "#090d16" : "var(--text-muted)",
                border: "none",
                padding: "0.4rem 1rem",
                borderRadius: "999px",
                fontWeight: "600",
                fontSize: "0.85rem",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
            >
              Blend Overlay Slider
            </button>
            <button
              onClick={() => setActiveTab("matchpoints")}
              style={{
                background: activeTab === "matchpoints" ? "var(--primary)" : "transparent",
                color: activeTab === "matchpoints" ? "#090d16" : "var(--text-muted)",
                border: "none",
                padding: "0.4rem 1rem",
                borderRadius: "999px",
                fontWeight: "600",
                fontSize: "0.85rem",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
            >
              Match Points Canvas
            </button>
          </div>
        </div>

        {activeTab === "overlay" ? (
          <OverlayToggle
            registeredUrl={result.registered_image_url}
            referenceUrl={referencePreview}
          />
        ) : (
          <MatchPointsCanvas
            imageUrl={result.registered_image_url}
            matchPointsUrl={result.match_points_url}
          />
        )}
      </div>

      {/* Metrics Section */}
      <div className="glass-card">
        <div className="section-title">Registration Performance Metrics</div>
        <MetricsPanel metrics={result.metrics} />
      </div>

      {/* Download Actions */}
      <div className="glass-card" style={{ textAlign: "center" }}>
        <div className="section-title" style={{ justifyContent: "center" }}>Export Results & Match Data</div>
        <div className="downloads-bar">
          <a href={result.registered_image_url} download className="btn-download">
            📥 Download Registered Image (.png)
          </a>
          <a href={result.match_points_url} download className="btn-download">
            📊 Download Match Points (.csv)
          </a>
        </div>
      </div>
    </div>
  );
}

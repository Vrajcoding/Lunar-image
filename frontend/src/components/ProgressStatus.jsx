import { useState, useEffect } from "react";

const SCIENTIFIC_STAGES = [
  "Loading image raster & parsing PDS4 metadata...",
  "Radiometric contrast normalization & CLAHE...",
  "Running LoFTR learned dense correspondence...",
  "Filtering spatial grid distribution & entropy...",
  "Estimating coarse geometric transformation (RANSAC)...",
  "Sub-pixel coordinate refinement (cornerSubPix)...",
  "Warping image & computing independent validation RMSE...",
];

export default function ProgressStatus({ status, error, reason }) {
  const [currentStageIndex, setCurrentStageIndex] = useState(0);

  useEffect(() => {
    if (status === "processing") {
      setCurrentStageIndex(0);
      const interval = setInterval(() => {
        setCurrentStageIndex((prev) => (prev + 1) % SCIENTIFIC_STAGES.length);
      }, 1500);
      return () => clearInterval(interval);
    }
  }, [status]);

  if (!status) return null;

  if (status === "processing") {
    return (
      <div className="glass-card status-card">
        <div className="spinner"></div>
        <div className="status-title">⏳ Registering Lunar Imagery...</div>
        <div className="status-sub stage-text">
          {SCIENTIFIC_STAGES[currentStageIndex]}
        </div>
        <div className="stage-indicators">
          {SCIENTIFIC_STAGES.map((_, i) => (
            <div
              key={i}
              className={`stage-dot ${i === currentStageIndex ? "active" : i < currentStageIndex ? "done" : ""}`}
            />
          ))}
        </div>
      </div>
    );
  }

  if (status === "failed") {
    return (
      <div className="glass-card status-card error-card">
        <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>❌</div>
        <div className="status-title error-title">Registration Rejected</div>
        {reason && <div className="error-reason-badge">Reason: {reason}</div>}
        <div className="status-sub">
          {error || "Could not establish a scientifically verifiable geometric registration across this image pair."}
        </div>
      </div>
    );
  }

  return null;
}

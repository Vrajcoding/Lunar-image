export default function ProgressStatus({ status, error }) {
  if (!status) return null;

  if (status === "processing") {
    return (
      <div className="glass-card status-card">
        <div className="spinner"></div>
        <div className="status-title">⏳ Registering Lunar Images...</div>
        <div className="status-sub">
          Extracting SIFT keypoints, enforcing grid uniformity, running RANSAC homography & sub-pixel alignment...
        </div>
      </div>
    );
  }

  if (status === "failed") {
    return (
      <div className="glass-card status-card error-card">
        <div style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>❌</div>
        <div className="status-title error-title">Image Registration Failed</div>
        <div className="status-sub">
          {error || "An error occurred during registration. Please verify image formats and try again."}
        </div>
      </div>
    );
  }

  return null;
}

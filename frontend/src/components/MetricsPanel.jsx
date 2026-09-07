export default function MetricsPanel({ metrics }) {
  if (!metrics) return null;

  const testRmse = metrics.test_rmse_px !== undefined ? metrics.test_rmse_px : metrics.rmse_px;
  const fitRmse = metrics.fit_rmse_px !== undefined ? metrics.fit_rmse_px : metrics.rmse_px;
  const confLevel = metrics.confidence_level || "MEDIUM";
  const method = metrics.method || "LoFTR";

  const cards = [
    {
      label: "Matcher Method",
      value: method,
      badge: method === "LoFTR" ? "Deep Learning" : "Classical",
      hint: "Correspondence engine used",
    },
    {
      label: "Independent Test RMSE",
      value: `${testRmse.toFixed(3)} px`,
      badge: testRmse < 1.0 ? "Sub-pixel (<1.0 px)" : "Standard",
      hint: "Held-out validation accuracy",
    },
    {
      label: "Fitting RMSE",
      value: `${fitRmse.toFixed(3)} px`,
      hint: "Inlier homography fitting error",
    },
    {
      label: "Inlier Count / Ratio",
      value: `${metrics.inlier_count} (${(metrics.inlier_ratio * 100).toFixed(1)}%)`,
      hint: "RANSAC verified correspondences",
    },
    {
      label: "Spatial Uniformity",
      value: `${metrics.uniformity_score.toFixed(3)}`,
      hint: metrics.spatial_entropy ? `Entropy: ${metrics.spatial_entropy.toFixed(2)} nats` : "Grid distribution (0-1)",
    },
    {
      label: "Confidence Grade",
      value: confLevel,
      level: confLevel,
      hint: metrics.confidence_score ? `Score: ${(metrics.confidence_score * 100).toFixed(1)}%` : "Overall quality metric",
    },
    {
      label: "Pipeline Runtime",
      value: `${metrics.runtime_sec.toFixed(2)} s`,
      hint: "Total end-to-end latency",
    },
  ];

  return (
    <div className="metrics-panel">
      {cards.map((c, i) => {
        let levelClass = "";
        if (c.level === "HIGH") levelClass = "badge-high";
        else if (c.level === "MEDIUM") levelClass = "badge-medium";
        else if (c.level === "LOW") levelClass = "badge-low";

        return (
          <div key={i} className={`metric-card ${levelClass}`}>
            <div className="metric-header-row">
              <span className="metric-label">{c.label}</span>
              {c.badge && <span className="metric-tag">{c.badge}</span>}
            </div>
            <span className={`metric-value ${c.level ? `value-${c.level.toLowerCase()}` : ""}`}>
              {c.value}
            </span>
            <span className="metric-hint">{c.hint}</span>
          </div>
        );
      })}
    </div>
  );
}

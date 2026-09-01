export default function MetricsPanel({ metrics }) {
  if (!metrics) return null;

  const cards = [
    { label: "RMSE Error", value: `${metrics.rmse_px} px`, hint: "Sub-pixel precision accuracy" },
    { label: "Inlier Count", value: metrics.inlier_count, hint: "RANSAC verified correspondences" },
    { label: "Inlier Ratio", value: `${(metrics.inlier_ratio * 100).toFixed(1)}%`, hint: "Quality match ratio" },
    { label: "Uniformity Score", value: metrics.uniformity_score, hint: "Spatial distribution (0 - 1)" },
    { label: "Runtime", value: `${metrics.runtime_sec} s`, hint: "Total pipeline latency" },
  ];

  return (
    <div className="metrics-panel">
      {cards.map((c, i) => (
        <div key={i} className="metric-card">
          <span className="metric-label">{c.label}</span>
          <span className="metric-value">{c.value}</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{c.hint}</span>
        </div>
      ))}
    </div>
  );
}

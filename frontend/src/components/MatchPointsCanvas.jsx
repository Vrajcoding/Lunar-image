import { useEffect, useRef, useState } from "react";
import axios from "axios";

export default function MatchPointsCanvas({ imageUrl, matchPointsUrl }) {
  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!matchPointsUrl) return;

    setLoading(true);
    axios.get(matchPointsUrl)
      .then((res) => {
        const text = res.data;
        const lines = text.trim().split("\n");
        const parsed = [];
        // Skip header
        for (let i = 1; i < lines.length; i++) {
          const parts = lines[i].split(",");
          if (parts.length >= 5) {
            parsed.push({
              src_x: parseFloat(parts[0]),
              src_y: parseFloat(parts[1]),
              ref_x: parseFloat(parts[2]),
              ref_y: parseFloat(parts[3]),
              is_inlier: parts[4].trim().toLowerCase() === "true" || parts[4].trim() === "1"
            });
          }
        }
        setPoints(parsed);
        setLoading(false);
      })
      .catch((err) => {
        console.warn("Could not load match points CSV:", err);
        setLoading(false);
      });
  }, [matchPointsUrl]);

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;

    img.onload = () => {
      canvas.width = img.naturalWidth || 800;
      canvas.height = img.naturalHeight || 600;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Render points
      points.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.ref_x, p.ref_y, 4, 0, 2 * Math.PI);
        if (p.is_inlier) {
          ctx.fillStyle = "#34d399";
          ctx.strokeStyle = "#065f46";
        } else {
          ctx.fillStyle = "#f87171";
          ctx.strokeStyle = "#991b1b";
        }
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
      });
    };
  }, [imageUrl, points]);

  return (
    <div style={{ width: "100%" }}>
      {loading ? (
        <div style={{ textAlign: "center", padding: "1rem", color: "var(--text-muted)" }}>
          Loading match points data...
        </div>
      ) : (
        <>
          <div className="canvas-wrapper">
            <canvas ref={canvasRef} />
          </div>
          <div className="canvas-legend">
            <div className="legend-item">
              <span className="legend-dot inlier"></span>
              <span>Inliers (RANSAC verified)</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot outlier"></span>
              <span>Outliers (Rejected)</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

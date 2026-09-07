import { useState, useRef } from "react";
import UploadForm from "./components/UploadForm";
import ProgressStatus from "./components/ProgressStatus";
import ResultView from "./components/ResultView";
import { registerImages, getStatus, getResult } from "./api";
import "./styles.css";

export default function App() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [reason, setReason] = useState(null);
  const [result, setResult] = useState(null);
  const previews = useRef({ source: null, reference: null });

  const handleSubmit = async (
    sourceFile,
    referenceFile,
    sourceLabelFile,
    referenceLabelFile,
    mode,
    sourceSensor,
    referenceSensor
  ) => {
    try {
      previews.current.source = URL.createObjectURL(sourceFile);
      previews.current.reference = URL.createObjectURL(referenceFile);

      setStatus("processing");
      setError(null);
      setReason(null);
      setResult(null);

      const res = await registerImages(
        sourceFile,
        referenceFile,
        sourceLabelFile,
        referenceLabelFile,
        null,
        mode,
        sourceSensor,
        referenceSensor
      );

      if (res && res.job_id) {
        pollStatus(res.job_id);
      } else {
        setStatus("failed");
        setError("Invalid response received from registration server.");
      }
    } catch (err) {
      console.error("Registration submit error:", err);
      setStatus("failed");
      setError(err.response?.data?.detail || err.message || "Failed to connect to backend server.");
    }
  };

  const pollStatus = (jobId) => {
    const interval = setInterval(async () => {
      try {
        const data = await getStatus(jobId);
        if (data.status === "completed") {
          clearInterval(interval);
          const finalResult = await getResult(jobId);
          setResult(finalResult);
          setStatus("completed");
        } else if (data.status === "failed") {
          clearInterval(interval);
          setStatus("failed");
          setReason(data.reason || "REGISTRATION_REJECTED");
          setError(data.error || "Correspondence or geometric verification failed.");
        }
      } catch (err) {
        console.error("Polling error:", err);
        clearInterval(interval);
        setStatus("failed");
        setError("Error polling job status from backend.");
      }
    }, 1500);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand-badge">
          <span>🌔 SIH 2026 PS 26166 • Chandrayaan-2 LunarMatch AI</span>
        </div>
        <h1>Multi-Modal Lunar Image Registration</h1>
        <p>
          Sub-pixel precision automated co-registration for Chandrayaan-2 orbital imagery (TMC-2, OHRC, IIRS)
          under extreme illumination, scale, and cross-sensor variations.
        </p>
      </header>

      <main>
        <UploadForm onSubmit={handleSubmit} />
        <ProgressStatus status={status} error={error} reason={reason} />
        {status === "completed" && result && (
          <ResultView
            result={result}
            sourcePreview={previews.current.source}
            referencePreview={previews.current.reference}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>LunarMatch AI Platform • SIH 2026 PS 26166 • Powered by Deep LoFTR, SIFT Fallback & Sub-Pixel RANSAC</p>
      </footer>
    </div>
  );
}

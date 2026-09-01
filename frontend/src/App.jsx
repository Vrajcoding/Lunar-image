import { useState, useRef } from "react";
import UploadForm from "./components/UploadForm";
import ProgressStatus from "./components/ProgressStatus";
import ResultView from "./components/ResultView";
import { registerImages, getStatus, getResult } from "./api";
import "./styles.css";

export default function App() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const previews = useRef({ source: null, reference: null });

  const handleSubmit = async (sourceFile, referenceFile) => {
    try {
      previews.current.source = URL.createObjectURL(sourceFile);
      previews.current.reference = URL.createObjectURL(referenceFile);
      
      setStatus("processing");
      setError(null);
      setResult(null);

      const res = await registerImages(sourceFile, referenceFile);
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
          setError(data.error || "Image registration pipeline encountered an error.");
        }
      } catch (err) {
        console.error("Polling error:", err);
        clearInterval(interval);
        setStatus("failed");
        setError("Error polling job status from backend.");
      }
    }, 2000);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand-badge">
          <span>🌔 Chandrayaan-2 TMC-2 / LROC AI Registration</span>
        </div>
        <h1>Lunar Image Registration System</h1>
        <p>
          Sub-pixel precision automated co-registration of Chandrayaan-2 lunar orbital imagery 
          against high-resolution lunar reference basemaps.
        </p>
      </header>

      <main>
        <UploadForm onSubmit={handleSubmit} />
        <ProgressStatus status={status} error={error} />
        {status === "completed" && result && (
          <ResultView
            result={result}
            sourcePreview={previews.current.source}
            referencePreview={previews.current.reference}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>Lunar Image Registration Platform • Built for SIH Hackathon • Powered by FastAPI & React</p>
      </footer>
    </div>
  );
}

import { useState } from "react";

export default function UploadForm({ onSubmit }) {
  const [sourceFile, setSourceFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [sourceLabelFile, setSourceLabelFile] = useState(null);
  const [referenceLabelFile, setReferenceLabelFile] = useState(null);
  const [sourceSensor, setSourceSensor] = useState("TMC-2");
  const [referenceSensor, setReferenceSensor] = useState("LROC / External");
  const [mode, setMode] = useState("auto");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (sourceFile && referenceFile) {
      onSubmit(
        sourceFile,
        referenceFile,
        sourceLabelFile,
        referenceLabelFile,
        mode,
        sourceSensor,
        referenceSensor
      );
    }
  };

  return (
    <div className="upload-section glass-card">
      <form onSubmit={handleSubmit}>
        {/* Sensor & Pipeline Configuration Bar */}
        <div className="sensor-config-grid">
          <div className="config-group">
            <label htmlFor="source-sensor-select">Source Payload</label>
            <select
              id="source-sensor-select"
              value={sourceSensor}
              onChange={(e) => setSourceSensor(e.target.value)}
              className="custom-select"
            >
              <option value="TMC-2">Chandrayaan-2 TMC-2 (5 m/px)</option>
              <option value="OHRC">Chandrayaan-2 OHRC (0.25 m/px)</option>
              <option value="IIRS">Chandrayaan-2 IIRS (Hyperspectral)</option>
              <option value="Unknown">Auto / Other Lunar Product</option>
            </select>
          </div>

          <div className="config-group">
            <label htmlFor="reference-sensor-select">Reference Target</label>
            <select
              id="reference-sensor-select"
              value={referenceSensor}
              onChange={(e) => setReferenceSensor(e.target.value)}
              className="custom-select"
            >
              <option value="LROC / External">LROC NAC / Global Lunar Basemap</option>
              <option value="TMC-2">Chandrayaan-2 TMC-2 Reference</option>
              <option value="OHRC">Chandrayaan-2 OHRC Reference</option>
              <option value="IIRS">Chandrayaan-2 IIRS Reference</option>
            </select>
          </div>

          <div className="config-group">
            <label htmlFor="mode-select">Correspondence Mode</label>
            <select
              id="mode-select"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="custom-select"
            >
              <option value="auto">Auto (Hybrid LoFTR + SIFT Fallback)</option>
              <option value="accuracy">Maximum Accuracy (Deep LoFTR)</option>
              <option value="fast">Fast (Classical SIFT/ORB)</option>
            </select>
          </div>
        </div>

        {/* Dropzones */}
        <div className="upload-grid" style={{ marginTop: "1rem" }}>
          {/* Source Image Dropzone */}
          <div className={`upload-dropzone ${sourceFile ? "has-file" : ""}`}>
            <input
              id="source-file-input"
              type="file"
              accept="image/*,.tif,.tiff,.xml,.img,.lbl"
              onChange={(e) => setSourceFile(e.target.files[0] || null)}
            />
            <span className="dropzone-icon">📷</span>
            <div className="dropzone-label">Source Orbit Image</div>
            <div className="dropzone-subtext">Click or drag & drop TMC-2, OHRC, or IIRS tile</div>
            {sourceFile && (
              <div className="file-preview-name">
                Selected: {sourceFile.name} ({(sourceFile.size / 1024).toFixed(1)} KB)
              </div>
            )}
          </div>

          {/* Reference Image Dropzone */}
          <div className={`upload-dropzone ${referenceFile ? "has-file" : ""}`}>
            <input
              id="reference-file-input"
              type="file"
              accept="image/*,.tif,.tiff,.xml,.img,.lbl"
              onChange={(e) => setReferenceFile(e.target.files[0] || null)}
            />
            <span className="dropzone-icon">🗺️</span>
            <div className="dropzone-label">Reference Image / Basemap</div>
            <div className="dropzone-subtext">Click or drag & drop reference lunar tile</div>
            {referenceFile && (
              <div className="file-preview-name">
                Selected: {referenceFile.name} ({(referenceFile.size / 1024).toFixed(1)} KB)
              </div>
            )}
          </div>
        </div>

        {/* Optional detached labels (PDS4 XML / PDS3 LBL) */}
        <div className="upload-grid label-grid">
          <label className="label-file-field">
            <span>Source detached label (Optional — PDS4 .xml / PDS3 .lbl)</span>
            <input
              id="source-label-file-input"
              type="file"
              accept=".xml,.lbl"
              onChange={(e) => setSourceLabelFile(e.target.files[0] || null)}
            />
            {sourceLabelFile && (
              <div className="file-preview-name">Selected: {sourceLabelFile.name}</div>
            )}
          </label>

          <label className="label-file-field">
            <span>Reference detached label (Optional — PDS4 .xml / PDS3 .lbl)</span>
            <input
              id="reference-label-file-input"
              type="file"
              accept=".xml,.lbl"
              onChange={(e) => setReferenceLabelFile(e.target.files[0] || null)}
            />
            {referenceLabelFile && (
              <div className="file-preview-name">Selected: {referenceLabelFile.name}</div>
            )}
          </label>
        </div>

        <div className="submit-btn-wrapper">
          <button
            id="register-btn"
            type="submit"
            className="btn-primary"
            disabled={!sourceFile || !referenceFile}
          >
            <span>🚀 Run Sub-Pixel Registration</span>
          </button>
        </div>
      </form>
    </div>
  );
}

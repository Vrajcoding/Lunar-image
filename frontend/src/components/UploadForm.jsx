import { useState } from "react";

export default function UploadForm({ onSubmit }) {
  const [sourceFile, setSourceFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [sourceLabelFile, setSourceLabelFile] = useState(null);
  const [referenceLabelFile, setReferenceLabelFile] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (sourceFile && referenceFile) {
      onSubmit(sourceFile, referenceFile, sourceLabelFile, referenceLabelFile);
    }
  };

  return (
    <div className="upload-section glass-card">
      <form onSubmit={handleSubmit}>
        <div className="upload-grid">
          {/* Source Image Dropzone */}
          <div className={`upload-dropzone ${sourceFile ? "has-file" : ""}`}>
            <input
              id="source-file-input"
              type="file"
              accept="image/*,.tif,.tiff,.xml,.img,.lbl"
              onChange={(e) => setSourceFile(e.target.files[0] || null)}
            />
            <span className="dropzone-icon">📷</span>
            <div className="dropzone-label">Source Image (Chandrayaan-2)</div>
            <div className="dropzone-subtext">Click or drag & drop TMC-2 / OHRC lunar tile</div>
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
            <div className="dropzone-label">Reference Image (Lunar Map)</div>
            <div className="dropzone-subtext">Click or drag & drop LROC basemap / reference tile</div>
            {referenceFile && (
              <div className="file-preview-name">
                Selected: {referenceFile.name} ({(referenceFile.size / 1024).toFixed(1)} KB)
              </div>
            )}
          </div>
        </div>

        {/* Optional detached labels — only needed for a raw PDS4 (.img + .xml)
            or PDS3 (.img + .lbl) pair. A single-file PNG/TIFF upload above
            leaves both of these empty and nothing else changes. */}
        <div className="upload-grid label-grid">
          <label className="label-file-field">
            <span>Source label (optional — PDS4 .xml / PDS3 .lbl)</span>
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
            <span>Reference label (optional — PDS4 .xml / PDS3 .lbl)</span>
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
            <span>🚀 Run Sub-Pixel Alignment</span>
          </button>
        </div>
      </form>
    </div>
  );
}

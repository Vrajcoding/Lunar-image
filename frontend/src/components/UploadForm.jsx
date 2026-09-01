import { useState } from "react";

export default function UploadForm({ onSubmit }) {
  const [sourceFile, setSourceFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (sourceFile && referenceFile) {
      onSubmit(sourceFile, referenceFile);
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
              accept="image/*"
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
              accept="image/*"
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

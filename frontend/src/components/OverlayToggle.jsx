import { useState } from "react";

export default function OverlayToggle({ registeredUrl, referenceUrl }) {
  const [opacity, setOpacity] = useState(0.5);

  return (
    <div className="overlay-container">
      <div className="overlay-viewport">
        <div className="overlay-stack">
          {referenceUrl && (
            <img src={referenceUrl} alt="Reference Base" className="overlay-base" />
          )}
          {registeredUrl && (
            <img
              src={registeredUrl}
              alt="Registered Overlay"
              className="overlay-top"
              style={{ opacity }}
            />
          )}
        </div>
      </div>

      <div className="overlay-controls">
        <div className="slider-group">
          <label htmlFor="opacity-slider">Blend Opacity: {Math.round(opacity * 100)}%</label>
          <input
            id="opacity-slider"
            type="range"
            min="0"
            max="1"
            step="0.02"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="range-slider"
          />
        </div>
        <p className="overlay-hint">
          💡 Drag the slider to cross-fade between the Reference basemap and the Warped Registered image. Perfectly aligned lunar craters will line up seamlessly.
        </p>
      </div>
    </div>
  );
}

# Frontend Module — Lunar Image Registration
## Owner: Person 3 (Frontend Engineer) — Dockerized React UI

> **Your job in one sentence:** Give judges/users a simple screen to upload two moon images, watch the alignment happen, and see the result — image overlay, match points drawn, and accuracy numbers — by calling the Backend's REST API. You never call the ML service directly.

---

# 1. What You're Building

A **React web app** (`frontend`) that:

1. Lets the user upload a Source (Chandrayaan-2) image and a Reference (lunar map) image.
2. Calls the Backend's `/api/register` endpoint and shows a loading state.
3. Polls `/api/status/{job_id}` until the job completes.
4. Fetches `/api/result/{job_id}` and displays:
   - Source image, Reference image, and Registered (aligned) image side by side.
   - A toggle to overlay the registered image on top of the reference (checkerboard/blend) so misalignment is visible.
   - The match points drawn as connecting lines/dots.
   - A metrics panel: RMSE, inlier count/ratio, uniformity score, runtime.
5. Provides download buttons/links for the registered image and match-point file.
6. Runs in its own Docker container, served via Nginx.

You only ever talk to the **Backend** (`http://localhost:8000/api/...` in dev, `http://backend:8000/api/...` is NOT used by you — the browser calls the Backend directly on its exposed host port, since the browser runs on the user's machine, not inside the Docker network).

---

# 2. What To Download / Install

| Tool | Version | Download |
|---|---|---|
| Node.js | 20.x LTS | https://nodejs.org/ |
| npm | comes with Node | — |
| Git | latest | https://git-scm.com/downloads |
| Docker Desktop | latest | https://www.docker.com/products/docker-desktop/ |
| VS Code | latest | https://code.visualstudio.com/ |

Check installs:
```bash
node --version     # v20.x
npm --version
docker --version
```

---

# 3. Folder Structure (create exactly this)

```text
lunar-registration/
└── frontend/
    ├── public/
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── api.js               # all Backend API calls live here
    │   ├── components/
    │   │   ├── UploadForm.jsx
    │   │   ├── ProgressStatus.jsx
    │   │   ├── ResultView.jsx
    │   │   ├── OverlayToggle.jsx
    │   │   ├── MatchPointsCanvas.jsx
    │   │   └── MetricsPanel.jsx
    │   └── styles.css
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── Dockerfile
    ├── nginx.conf
    └── README.md
```

Create it with Vite (fast, minimal React setup):
```bash
cd lunar-registration
npm create vite@latest frontend -- --template react
cd frontend
npm install
```

---

# 4. Dependencies — `package.json` (key additions beyond Vite defaults)

```bash
npm install axios
```

Your `package.json` scripts section should look like:
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  }
}
```

---

# 5. The API Contract You Call (owned by Backend — confirm with them before changing)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/register` | Upload source + reference image (multipart/form-data) |
| `GET` | `/api/status/{job_id}` | Poll job status |
| `GET` | `/api/result/{job_id}` | Get metrics + file URLs once completed |
| `GET` | `/api/download/{job_id}/{file_type}` | Download `registered` or `matches` file |

**Base URL in development:** `http://localhost:8000` (Backend's exposed Docker port)
**Base URL in production (behind the same domain/Nginx):** you can proxy `/api` through Nginx to the backend container — see Section 8.

---

# 6. Code — What Each File Does & Starter Code

## 6.1 `src/api.js` — single place for all backend calls
```javascript
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function registerImages(sourceFile, referenceFile) {
  const formData = new FormData();
  formData.append("source", sourceFile);
  formData.append("reference", referenceFile);

  const res = await axios.post(`${BASE_URL}/api/register`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // { job_id, status }
}

export async function getStatus(jobId) {
  const res = await axios.get(`${BASE_URL}/api/status/${jobId}`);
  return res.data; // { job_id, status }
}

export async function getResult(jobId) {
  const res = await axios.get(`${BASE_URL}/api/result/${jobId}`);
  return res.data; // { registered_image_url, match_points_url, metrics }
}

export function downloadUrl(jobId, fileType) {
  return `${BASE_URL}/api/download/${jobId}/${fileType}`;
}
```

## 6.2 `src/components/UploadForm.jsx`
```jsx
import { useState } from "react";

export default function UploadForm({ onSubmit }) {
  const [sourceFile, setSourceFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (sourceFile && referenceFile) onSubmit(sourceFile, referenceFile);
  };

  return (
    <form onSubmit={handleSubmit} className="upload-form">
      <label>
        Source Image (Chandrayaan-2)
        <input type="file" accept="image/*" onChange={(e) => setSourceFile(e.target.files[0])} />
      </label>
      <label>
        Reference Image (Lunar Map)
        <input type="file" accept="image/*" onChange={(e) => setReferenceFile(e.target.files[0])} />
      </label>
      <button type="submit" disabled={!sourceFile || !referenceFile}>
        Register Images
      </button>
    </form>
  );
}
```

## 6.3 `src/components/ProgressStatus.jsx`
```jsx
export default function ProgressStatus({ status }) {
  if (status === "processing") return <p className="status">⏳ Registering images…</p>;
  if (status === "failed") return <p className="status error">❌ Registration failed.</p>;
  return null;
}
```

## 6.4 `src/components/MetricsPanel.jsx`
```jsx
export default function MetricsPanel({ metrics }) {
  if (!metrics) return null;
  const rows = [
    ["RMSE (px)", metrics.rmse_px],
    ["Inlier Count", metrics.inlier_count],
    ["Inlier Ratio", metrics.inlier_ratio],
    ["Uniformity Score", metrics.uniformity_score],
    ["Runtime (s)", metrics.runtime_sec],
  ];
  return (
    <div className="metrics-panel">
      {rows.map(([label, value]) => (
        <div key={label} className="metric-card">
          <span className="metric-label">{label}</span>
          <span className="metric-value">{value}</span>
        </div>
      ))}
    </div>
  );
}
```

## 6.5 `src/components/OverlayToggle.jsx` — checkerboard/blend comparison
```jsx
import { useState } from "react";

export default function OverlayToggle({ registeredUrl, referenceUrl }) {
  const [opacity, setOpacity] = useState(0.5);
  return (
    <div className="overlay-container">
      <div className="overlay-stack">
        <img src={referenceUrl} alt="Reference" className="overlay-base" />
        <img
          src={registeredUrl}
          alt="Registered"
          className="overlay-top"
          style={{ opacity }}
        />
      </div>
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={opacity}
        onChange={(e) => setOpacity(parseFloat(e.target.value))}
      />
      <p>Slide to blend registered image over reference — good alignment means features line up as you slide.</p>
    </div>
  );
}
```

## 6.6 `src/components/MatchPointsCanvas.jsx` — optional visualization if match CSV is parsed client-side
```jsx
import { useEffect, useRef } from "react";

// points: [{src_x, src_y, ref_x, ref_y, is_inlier}]
export default function MatchPointsCanvas({ imageUrl, points, width, height }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
      points.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.ref_x, p.ref_y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = p.is_inlier ? "lime" : "red";
        ctx.fill();
      });
    };
  }, [imageUrl, points, width, height]);

  return <canvas ref={canvasRef} width={width} height={height} />;
}
```

## 6.7 `src/components/ResultView.jsx`
```jsx
import MetricsPanel from "./MetricsPanel";
import OverlayToggle from "./OverlayToggle";

export default function ResultView({ result, sourcePreview, referencePreview }) {
  return (
    <div className="result-view">
      <div className="image-row">
        <div><h4>Source</h4><img src={sourcePreview} alt="Source" /></div>
        <div><h4>Reference</h4><img src={referencePreview} alt="Reference" /></div>
        <div><h4>Registered</h4><img src={result.registered_image_url} alt="Registered" /></div>
      </div>

      <h4>Overlay Comparison</h4>
      <OverlayToggle
        registeredUrl={result.registered_image_url}
        referenceUrl={referencePreview}
      />

      <h4>Evaluation Metrics</h4>
      <MetricsPanel metrics={result.metrics} />

      <div className="downloads">
        <a href={result.registered_image_url} download>Download Registered Image</a>
        <a href={result.match_points_url} download>Download Match Points (CSV)</a>
      </div>
    </div>
  );
}
```

## 6.8 `src/App.jsx` — ties it all together
```jsx
import { useState, useRef } from "react";
import UploadForm from "./components/UploadForm";
import ProgressStatus from "./components/ProgressStatus";
import ResultView from "./components/ResultView";
import { registerImages, getStatus, getResult } from "./api";
import "./styles.css";

export default function App() {
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const previews = useRef({ source: null, reference: null });

  const handleSubmit = async (sourceFile, referenceFile) => {
    previews.current.source = URL.createObjectURL(sourceFile);
    previews.current.reference = URL.createObjectURL(referenceFile);
    setStatus("processing");
    setResult(null);

    const { job_id } = await registerImages(sourceFile, referenceFile);
    pollStatus(job_id);
  };

  const pollStatus = (jobId) => {
    const interval = setInterval(async () => {
      const { status: currentStatus } = await getStatus(jobId);
      if (currentStatus === "completed") {
        clearInterval(interval);
        const finalResult = await getResult(jobId);
        setResult(finalResult);
        setStatus("completed");
      } else if (currentStatus === "failed") {
        clearInterval(interval);
        setStatus("failed");
      }
    }, 2000); // poll every 2 seconds
  };

  return (
    <div className="app">
      <h1>Lunar Background Image Registration</h1>
      <UploadForm onSubmit={handleSubmit} />
      <ProgressStatus status={status} />
      {status === "completed" && result && (
        <ResultView
          result={result}
          sourcePreview={previews.current.source}
          referencePreview={previews.current.reference}
        />
      )}
    </div>
  );
}
```

Run locally to test (needs backend running on 8000):
```bash
npm run dev
```
Open http://localhost:5173

---

# 7. Dockerfile (`frontend/Dockerfile`) — multi-stage build served by Nginx

```dockerfile
# Stage 1: build the React app
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: serve with Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## `frontend/nginx.conf`
```nginx
server {
    listen 80;
    server_name localhost;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

> Note: the frontend calls the Backend directly from the **browser** at `http://localhost:8000`, so no Nginx reverse-proxy to the backend is required for the hackathon setup — just make sure `VITE_API_BASE_URL` (or the hardcoded default in `api.js`) points at the Backend's exposed host port.

---

# 8. Your Section of `docker-compose.yml` (Backend owns the full file — this is your service block, for reference)

```yaml
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    networks:
      - lunar-net
```

After `docker compose up --build` (run by anyone from the project root), your app is live at **http://localhost:3000**.

---

# 9. Your Day-by-Day Checklist

**Day 1**
- [ ] Scaffold the Vite React app, confirm `npm run dev` shows the default page
- [ ] Build `UploadForm.jsx` and `api.js` — confirm you can upload two files and get a `job_id` back (test against Backend's locally running service or with a mocked response first if Backend isn't ready yet)
- [ ] Build `ProgressStatus.jsx` and the polling loop in `App.jsx`

**Day 2**
- [ ] Build `ResultView.jsx`, `MetricsPanel.jsx`, `OverlayToggle.jsx`
- [ ] Style the app (`styles.css`) — clean, readable layout, not necessarily fancy
- [ ] Write `Dockerfile` + `nginx.conf`, confirm `docker build` succeeds and the built app loads static files correctly
- [ ] Full integration test against the real Backend (once it's ready) — confirm end-to-end upload → result flow works

**Day 3**
- [ ] Add `MatchPointsCanvas.jsx` if match-point visualization is prioritized (stretch goal)
- [ ] Handle edge cases: large files, wrong file types, backend errors, slow responses (loading states)
- [ ] Polish overlay slider UX — this is often the most visually convincing part of the demo
- [ ] Test the full `docker compose up --build` flow end-to-end (all 3 containers) from a clean clone

**Final Day**
- [ ] Final visual polish, add a title/header, maybe a small explanation of what illumination/viewpoint/scale variation means for context
- [ ] Confirm the app works after a fresh `docker compose up --build` on a teammate's machine
- [ ] Prepare a 2-minute walkthrough of the UI for the demo

---

# 10. Acceptance Criteria

- [ ] User can upload two images and see a loading/processing state (no frozen UI)
- [ ] On completion, source/reference/registered images all display correctly
- [ ] Overlay slider visibly demonstrates alignment quality
- [ ] Metrics panel shows RMSE, inlier count/ratio, uniformity score, runtime
- [ ] Download buttons for registered image and match points both work
- [ ] Failed jobs show a clear error state, not a silent hang
- [ ] `docker compose up --build` serves the working app at `http://localhost:3000`
- [ ] App does not crash on large images or slow backend responses

---

# 11. Common Pitfalls

- CORS errors in the browser console — this is a **Backend** fix (`CORSMiddleware`), but you'll be the one who sees it first; tell Backend immediately if you see `blocked by CORS policy`.
- Forgetting the multi-stage Docker build — building React inside the final Nginx image bloats it massively; always build in a separate `node` stage and copy only `dist/`.
- Hardcoding `http://localhost:8000` everywhere instead of centralizing it in `api.js` — makes it painful to change later.
- Polling too fast (every 200ms) — hammers the backend; 2 seconds is plenty for a hackathon demo.
- Not handling the `failed` status — the UI should never spin forever if the ML pipeline errors out.
- Testing only with tiny sample images — always test the full flow with a real, large Chandrayaan-2-sized image before the demo.

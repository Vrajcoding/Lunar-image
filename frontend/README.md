# Lunar Registration Frontend Service

Dockerized React UI for the Lunar Image Registration Platform.

## Quick Start

### Local Development
```bash
npm install
npm run dev
```
The Vite dev server will run at http://localhost:3000 (or http://localhost:5173).

### Docker Build & Run
```bash
docker build -t lunar-frontend .
docker run -p 3000:80 lunar-frontend
```
App will be accessible at http://localhost:3000.

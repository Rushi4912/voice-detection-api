# Voice Detection API - Deployment Guide

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- GitHub account (for model hosting)

---

## Step 1: Upload Models to GitHub Releases

1. **Create a new Release on GitHub:**
   - Go to your repo → Releases → Create new release
   - Tag: `v1.0.0-models`
   - Title: "ML Models v1.0"

2. **Upload these files (from `python/models/`):**
   - `resnet_asvspoof.pt` (43 MB)
   - `acoustic_classifier.pkl` (574 KB)
   - `acoustic_scaler.pkl` (2 KB)
   - `ensemble_weights.json` (118 bytes)

3. **Update `download_models.py`:**
   ```python
   GITHUB_REPO = "your-username/voice-detection-api"  # Change this
   RELEASE_TAG = "v1.0.0-models"
   ```

---

## Step 2: Local Docker Testing

```bash
# Build and run both services
docker-compose up --build

# Test health endpoints
curl http://localhost:5000/health  # Python
curl http://localhost:3000/health  # Node.js
```

---

## Step 3: Deploy to Railway

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Initialize project:**
   ```bash
   railway init
   ```

3. **Deploy:**
   ```bash
   railway up
   ```

4. **Set environment variables on Railway dashboard:**
   - `PYTHON_SERVICE_URL` = `<internal python service URL>`

---

## Alternative: Deploy to Render

1. Create a new **Blueprint** on Render
2. Connect your GitHub repo
3. Render will auto-detect `docker-compose.yml`
4. Set environment variables

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze audio (Python) |
| `/api/detect` | POST | Detect AI voice (Node.js gateway) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000/5000 | Server port |
| `PYTHON_SERVICE_URL` | http://localhost:5000 | Python service URL |

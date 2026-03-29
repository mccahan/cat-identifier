# Cat Identifier 🐱

A web app to track when each cat (Hawthorne, Roxie, Sadie) was last seen using Frigate NVR events and a custom ML model.

## Features

- **Last Seen Dashboard** - Shows when each cat was last spotted with thumbnail
- **Recent Sightings** - Gallery of recent cat detections with timestamps
- **Feedback System** - Correct misidentifications to improve accuracy
- **Auto-polling** - Fetches new events from Frigate every 2 minutes

## Architecture

- Polls Frigate's API for cat events
- Runs images through a custom ONNX classifier (MobileNetV2, trained on ~450 labeled images)
- Stores sightings in SQLite with thumbnails
- Serves a simple web UI

## Deployment

### Docker

```bash
docker run -d \
  --name cat-identifier \
  -p 8080:8080 \
  -v cat-data:/data \
  -v cat-training:/training \
  -v cat-models:/models \
  -e FRIGATE_URL=http://your-frigate:5000 \
  ghcr.io/mccahan/cat-identifier:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRIGATE_URL` | `http://10.0.1.2:5000` | Frigate API URL |
| `POLL_INTERVAL` | `120` | Seconds between polls |
| `DATA_DIR` | `/data` | Where to store SQLite DB |
| `TRAINING_DIR` | `/training` | Where to store training images |
| `MODELS_DIR` | `/models` | Where to store trained models |
| `RETRAIN_THRESHOLD` | `10` | Corrections before suggesting retrain |

## Model

The classifier was trained using PyTorch on ~450 manually labeled cat images:
- **Hawthorne** (black): 385 images
- **Sadie** (white): 42 images  
- **Roxie** (brown): 22 images

Achieved 100% validation accuracy. Model is exported to ONNX for inference.

## API Endpoints

- `GET /` - Web UI
- `GET /api/cats` - Last seen info per cat
- `GET /api/recent?limit=N` - Recent sightings
- `GET /api/sighting/:id/thumbnail` - Get thumbnail image
- `POST /api/feedback` - Submit correction
- `GET /api/stats` - Sighting statistics
- `POST /api/poll` - Trigger manual poll
- `GET /api/retrain/status` - Get retraining status
- `POST /api/retrain` - Trigger model retraining
- `POST /api/model/reload` - Hot-reload model without restart

## Retraining

When you correct misidentifications, the corrected images are saved to the training directory. Once you have enough corrections (default: 10), you can trigger retraining:

```bash
curl -X POST https://cats.mccahan.dev/api/retrain
```

The model will retrain using:
1. Original training data (if present)
2. Current sightings with their labels
3. Human-corrected images (highest priority)

After training completes, the new model is automatically loaded.

---
*Built 2026-03-28*

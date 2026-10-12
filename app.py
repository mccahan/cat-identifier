#!/usr/bin/env python3
"""
Cat Identifier - Track when each cat was last seen using Frigate events
"""
import os
import sys
import re
import ast
import json
import shutil
import subprocess
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO

import requests
import onnxruntime as ort
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory, send_file

# Configuration
FRIGATE_URL = os.getenv("FRIGATE_URL", "http://10.0.1.2:5000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "120"))  # seconds
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/cat_classifier.onnx"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "/app/models/cat_classifier_labels.txt"))
TRAINING_DIR = Path(os.getenv("TRAINING_DIR", "/training"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "10"))  # corrections before suggesting retrain
SAMPLE_PER_CAT = int(os.getenv("SAMPLE_PER_CAT", "250"))  # confident sightings per cat a retrain adds
SAMPLE_MIN_CONFIDENCE = float(os.getenv("SAMPLE_MIN_CONFIDENCE", "0.9"))
TRAIN_TIMEOUT = int(os.getenv("TRAIN_TIMEOUT", "3600"))  # seconds
TRAINING_STATE = DATA_DIR / "training.json"
MODEL_FILES = ("cat_classifier.onnx", "cat_classifier_labels.txt", "cat_classifier.pth")
NOT_A_CAT = "not_a_cat"  # Label for sightings that turned out not to be a cat: hidden, and never trained on

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Global state
db_lock = threading.Lock()
model_lock = threading.Lock()
model_session = None
labels = []
is_training = False
training_lock = threading.Lock()
loaded_model_path = None

def init_db():
    """Initialize SQLite database"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / "sightings.db"
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # Sightings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            cat_name TEXT,
            confidence REAL,
            camera TEXT,
            timestamp REAL,
            thumbnail BLOB,
            created_at REAL DEFAULT (strftime('%s', 'now'))
        )
    ''')
    
    # Feedback table for corrections
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sighting_id INTEGER,
            original_prediction TEXT,
            correct_label TEXT,
            created_at REAL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (sighting_id) REFERENCES sightings(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_model():
    """Load ONNX model and labels"""
    global model_session, labels, loaded_model_path
    
    # Check for model in MODELS_DIR first (for hot-reload after training)
    model_path = MODELS_DIR / "cat_classifier.onnx" if (MODELS_DIR / "cat_classifier.onnx").exists() else MODEL_PATH
    
    with model_lock:
        if model_path.exists():
            model_session = ort.InferenceSession(str(model_path))
            loaded_model_path = model_path
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
            return
        
        labels_path = MODELS_DIR / "cat_classifier_labels.txt" if (MODELS_DIR / "cat_classifier_labels.txt").exists() else LABELS_PATH
        if labels_path.exists():
            labels = labels_path.read_text().strip().split('\n')
            print(f"Loaded labels: {labels}")
        else:
            labels = ["hawthorne", "roxie", "sadie"]
            print(f"Using default labels: {labels}")

def preprocess_image(image_bytes):
    """Preprocess image for model inference"""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128), Image.LANCZOS)
    
    # Convert to numpy and normalize (ImageNet stats)
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = np.expand_dims(img_array, 0).astype(np.float32)  # Add batch dimension
    
    return img_array

def classify_cat(image_bytes):
    """Run inference on cat image"""
    if model_session is None:
        return None, 0.0
    
    try:
        input_data = preprocess_image(image_bytes)
        input_name = model_session.get_inputs()[0].name
        output = model_session.run(None, {input_name: input_data})[0]
        
        # Softmax
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / exp_output.sum()
        
        predicted_idx = np.argmax(probs)
        confidence = float(probs[0][predicted_idx])
        cat_name = labels[predicted_idx]
        
        return cat_name, confidence
    except Exception as e:
        print(f"Classification error: {e}")
        return None, 0.0

def get_db():
    """Get database connection"""
    return sqlite3.connect(str(DATA_DIR / "sightings.db"))

def poll_frigate(lookback_hours=24):
    """Poll Frigate for new cat events"""
    print("Polling Frigate for cat events...")
    
    try:
        # Get recent cat events
        after = time.time() - (lookback_hours * 3600)
        resp = requests.get(
            f"{FRIGATE_URL}/api/events",
            params={"label": "cat", "after": after, "limit": 100},
            timeout=10
        )
        resp.raise_for_status()
        events = resp.json()
        
        print(f"Found {len(events)} recent cat events")
        
        with db_lock:
            conn = get_db()
            c = conn.cursor()
            
            for event in events:
                event_id = event.get("id")
                
                # Check if we've already processed this event
                c.execute("SELECT id FROM sightings WHERE event_id = ?", (event_id,))
                if c.fetchone():
                    continue
                
                # Get thumbnail
                try:
                    thumb_resp = requests.get(
                        f"{FRIGATE_URL}/api/events/{event_id}/thumbnail.jpg",
                        timeout=10
                    )
                    thumb_resp.raise_for_status()
                    thumbnail = thumb_resp.content
                except Exception as e:
                    print(f"Failed to get thumbnail for {event_id}: {e}")
                    continue
                
                # Classify
                cat_name, confidence = classify_cat(thumbnail)
                
                if cat_name is None:
                    continue
                
                # Store sighting
                c.execute('''
                    INSERT INTO sightings (event_id, cat_name, confidence, camera, timestamp, thumbnail)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    event_id,
                    cat_name,
                    confidence,
                    event.get("camera"),
                    event.get("start_time"),
                    thumbnail
                ))
                
                print(f"New sighting: {cat_name} ({confidence:.1%}) on {event.get('camera')}")
            
            conn.commit()
            conn.close()
            
    except Exception as e:
        print(f"Error polling Frigate: {e}")

def polling_loop():
    """Background polling loop"""
    while True:
        poll_frigate()
        time.sleep(POLL_INTERVAL)

# API Routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/widget.html')
def widget():
    return send_from_directory('static', 'widget.html')

@app.route('/api/cats')
def get_cats():
    """Get last seen info for each cat"""
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        result = {}
        for cat in labels:
            c.execute('''
                SELECT id, timestamp, camera, confidence
                FROM sightings
                WHERE cat_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (cat,))
            row = c.fetchone()
            
            if row:
                result[cat] = {
                    "sighting_id": row[0],
                    "last_seen": row[1],
                    "camera": row[2],
                    "confidence": row[3]
                }
            else:
                result[cat] = None
        
        conn.close()
    
    return jsonify(result)

@app.route('/api/recent')
def get_recent():
    """Get recent sightings with images, optionally for one cat and before a timestamp (for paging)"""
    limit = request.args.get('limit', 20, type=int)
    cat = request.args.get('cat')
    before = request.args.get('before', type=float)

    where, params = [], []
    if cat:
        where.append("cat_name = ?")
        params.append(cat)
    else:
        # Not-a-cat sightings only show when asked for (cat=not_a_cat)
        where.append("cat_name != ?")
        params.append(NOT_A_CAT)
    if before is not None:
        where.append("timestamp < ?")
        params.append(before)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with db_lock:
        conn = get_db()
        c = conn.cursor()

        c.execute(f'''
            SELECT id, event_id, cat_name, confidence, camera, timestamp
            FROM sightings
            {where_sql}
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (*params, limit))
        
        rows = c.fetchall()
        conn.close()
    
    sightings = []
    for row in rows:
        sightings.append({
            "id": row[0],
            "event_id": row[1],
            "cat_name": row[2],
            "confidence": row[3],
            "camera": row[4],
            "timestamp": row[5]
        })
    
    return jsonify(sightings)

@app.route('/api/sighting/<int:sighting_id>/thumbnail')
def get_thumbnail(sighting_id):
    """Get thumbnail for a sighting"""
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT thumbnail FROM sightings WHERE id = ?", (sighting_id,))
        row = c.fetchone()
        conn.close()
    
    if row and row[0]:
        return send_file(BytesIO(row[0]), mimetype='image/jpeg')
    return "Not found", 404

def forget_training_images(sighting_id):
    """Remove a sighting's earlier correction and exported images, under any label"""
    for pattern in (f"corrections/*/{sighting_id}_*.jpg", f"current/*/{sighting_id}.jpg"):
        for filepath in TRAINING_DIR.glob(pattern):
            filepath.unlink(missing_ok=True)

def save_training_image(thumbnail_bytes, label, sighting_id):
    """Save a corrected image to the training directory"""
    training_path = TRAINING_DIR / "corrections" / label
    training_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{sighting_id}_{int(time.time())}.jpg"
    filepath = training_path / filename
    
    with open(filepath, 'wb') as f:
        f.write(thumbnail_bytes)
    
    print(f"Saved training image: {filepath}")
    return str(filepath)

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit correction feedback"""
    data = request.json
    sighting_id = data.get('sighting_id')
    correct_label = data.get('correct_label')
    
    if not sighting_id or not correct_label:
        return jsonify({"error": "Missing sighting_id or correct_label"}), 400
    
    if correct_label not in labels and correct_label != NOT_A_CAT:
        return jsonify({"error": f"Invalid label. Must be one of: {labels + [NOT_A_CAT]}"}), 400
    
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        # Get original prediction and thumbnail
        c.execute("SELECT cat_name, thumbnail FROM sightings WHERE id = ?", (sighting_id,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return jsonify({"error": "Sighting not found"}), 404
        
        original = row[0]
        thumbnail = row[1]
        
        # Store feedback
        c.execute('''
            INSERT INTO feedback (sighting_id, original_prediction, correct_label)
            VALUES (?, ?, ?)
        ''', (sighting_id, original, correct_label))
        
        # Update the sighting with corrected label
        c.execute('''
            UPDATE sightings SET cat_name = ?, confidence = -1
            WHERE id = ?
        ''', (correct_label, sighting_id))
        
        # Get total corrections count
        c.execute("SELECT COUNT(*) FROM feedback")
        total_corrections = c.fetchone()[0]
        
        conn.commit()
        conn.close()
    
    # Replace any training copies under its old label; not-a-cat ones aren't trained on at all
    forget_training_images(sighting_id)
    if thumbnail and correct_label != NOT_A_CAT:
        save_training_image(thumbnail, correct_label, sighting_id)
    
    response = {
        "success": True, 
        "original": original, 
        "corrected": correct_label,
        "total_corrections": total_corrections
    }
    
    # Suggest retraining if threshold reached
    if total_corrections >= RETRAIN_THRESHOLD and total_corrections % RETRAIN_THRESHOLD == 0:
        response["retrain_suggested"] = True
        response["message"] = f"You've made {total_corrections} corrections. Consider retraining the model!"
    
    return jsonify(response)

@app.route('/api/stats')
def get_stats():
    """Get sighting statistics"""
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        # Total sightings per cat
        c.execute('''
            SELECT cat_name, COUNT(*) as count
            FROM sightings
            WHERE cat_name != ?
            GROUP BY cat_name
        ''', (NOT_A_CAT,))
        totals = {row[0]: row[1] for row in c.fetchall()}
        
        # Corrections count
        c.execute("SELECT COUNT(*) FROM feedback")
        corrections = c.fetchone()[0]
        
        # Today's sightings
        today_start = time.time() - (time.time() % 86400)
        c.execute('''
            SELECT cat_name, COUNT(*) as count
            FROM sightings
            WHERE timestamp >= ? AND cat_name != ?
            GROUP BY cat_name
        ''', (today_start, NOT_A_CAT))
        today = {row[0]: row[1] for row in c.fetchall()}
        
        conn.close()
    
    return jsonify({
        "totals": totals,
        "today": today,
        "corrections": corrections
    })

@app.route('/api/poll', methods=['POST'])
def trigger_poll():
    """Manually trigger a poll"""
    threading.Thread(target=poll_frigate, daemon=True).start()
    return jsonify({"success": True, "message": "Poll triggered"})

def read_training_state():
    """The last (or running) retrain: status, step, epoch, images, accuracy, error, using"""
    try:
        return json.loads(TRAINING_STATE.read_text())
    except (FileNotFoundError, ValueError):
        return {}

def write_training_state(state):
    tmp = TRAINING_STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state))
    tmp.replace(TRAINING_STATE)

def count_images(folder):
    return len(list(folder.glob("*.jpg"))) if folder.exists() else 0

@app.route('/api/retrain/status')
def retrain_status():
    """Get retraining status, what a retrain would learn from, and the last one's result"""
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        # Count corrections per label
        c.execute('''
            SELECT correct_label, COUNT(*) 
            FROM feedback 
            GROUP BY correct_label
        ''')
        corrections_by_cat = {row[0]: row[1] for row in c.fetchall()}
        
        c.execute("SELECT COUNT(*) FROM feedback")
        total_corrections = c.fetchone()[0]
        
        # Confident sightings, which a retrain samples from
        c.execute('''
            SELECT cat_name, COUNT(*)
            FROM sightings
            WHERE confidence >= ?
            GROUP BY cat_name
        ''', (SAMPLE_MIN_CONFIDENCE,))
        confident = {row[0]: row[1] for row in c.fetchall()}
        
        conn.close()
    
    original_counts = {label: count_images(TRAINING_DIR / "original" / label) for label in labels}
    # Images a retrain would use per cat: originals, corrections and the sample
    planned = {
        label: original_counts[label]
            + count_images(TRAINING_DIR / "corrections" / label)
            + min(SAMPLE_PER_CAT, confident.get(label, 0))
        for label in labels
    }
    
    state = read_training_state()
    return jsonify({
        "is_training": is_training,
        "total_corrections": total_corrections,
        "corrections_by_cat": corrections_by_cat,
        "original_counts": original_counts,
        "planned": planned,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "can_retrain": not is_training,
        "training": state,
        # When the model in use was trained (null: the one built into the image)
        "model_trained_at": loaded_model_path.stat().st_mtime if loaded_model_path and loaded_model_path.parent == MODELS_DIR else None,
        "can_undo": not is_training and state.get("using") in ("new", "previous"),
    })

@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining"""
    global is_training
    
    with training_lock:
        if is_training:
            return jsonify({"error": "Training already in progress"}), 409
        is_training = True
    
    state = {"status": "running", "started_at": time.time(), "step": "Exporting training images"}
    # A failed run leaves the models as they were, so an earlier run's undo still applies
    if read_training_state().get("using"):
        state["using"] = read_training_state()["using"]
    write_training_state(state)
    threading.Thread(target=run_training, daemon=True).start()
    
    return jsonify({
        "success": True, 
        "message": "Training started",
        "status_url": "/api/retrain/status"
    })

@app.route('/api/retrain/undo', methods=['POST'])
def undo_retrain():
    """Switch between the retrained model and the one it replaced"""
    with training_lock:
        if is_training:
            return jsonify({"error": "Training in progress"}), 409
        state = read_training_state()
        if state.get("using") not in ("new", "previous"):
            return jsonify({"error": "There's no previous model to go back to"}), 400
        swap_models(MODELS_DIR, MODELS_DIR / "previous")
        load_model()
        state["using"] = "previous" if state["using"] == "new" else "new"
        write_training_state(state)
    return jsonify({"success": True, "using": state["using"]})

def swap_models(a, b):
    """Exchange the model files of two folders (one missing from a folder ends up missing from the other)"""
    b.mkdir(parents=True, exist_ok=True)
    tmp = MODELS_DIR / ".swap"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for name in MODEL_FILES:
        if (a / name).exists():
            (a / name).replace(tmp / name)
        if (b / name).exists():
            (b / name).replace(a / name)
        if (tmp / name).exists():
            (tmp / name).replace(b / name)
    tmp.rmdir()

def run_training():
    """Run the training process: export, train into a candidate folder, then swap it in"""
    global is_training
    
    state = read_training_state()
    candidate = MODELS_DIR / "candidate"
    previous = MODELS_DIR / "previous"
    log = []
    timed_out = threading.Event()
    
    try:
        print("Starting model retraining...")
        
        # Ensure directories exist
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(candidate, ignore_errors=True)
        
        export_training_data()
        state["step"] = "Training"
        write_training_state(state)
        
        # Train into the candidate folder; the live model is untouched until it succeeds
        env = {**os.environ, "OUTPUT_DIR": str(candidate), "TORCH_HOME": str(MODELS_DIR / ".torch")}
        proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).parent / "train.py")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        def kill():
            timed_out.set()
            proc.kill()
        timer = threading.Timer(TRAIN_TIMEOUT, kill)
        timer.start()
        
        for line in proc.stdout:
            line = line.rstrip()
            print(line)
            log = (log + [line])[-15:]
            epoch = re.match(r"Epoch (\d+)/(\d+)", line)
            best = re.match(r"Best validation accuracy: ([\d.]+)%", line)
            if epoch:
                state["epoch"], state["epochs"] = int(epoch[1]), int(epoch[2])
            elif line.startswith("Per-label counts:"):
                state["images"] = ast.literal_eval(line.split(":", 1)[1].strip())
            elif best:
                state["accuracy"] = float(best[1])
            else:
                continue
            write_training_state(state)
        
        proc.wait()
        timer.cancel()
        if timed_out.is_set():
            raise RuntimeError(f"Stopped after {TRAIN_TIMEOUT // 60} minutes")
        if proc.returncode != 0:
            # The last line is usually the error itself (or train.py saying why it stopped)
            raise RuntimeError(log[-1] if log else f"train.py failed (exit code {proc.returncode})")
        if not (candidate / "cat_classifier.onnx").exists():
            raise RuntimeError("Training didn't produce a model")
        
        # Keep the model it replaces, for undo
        shutil.rmtree(previous, ignore_errors=True)
        swap_models(MODELS_DIR, previous)
        swap_models(MODELS_DIR, candidate)
        load_model()
        print("Training completed, model reloaded")
        state.update(status="done", using="new")
    except Exception as e:
        print(f"Training error: {e}")
        state.update(status="failed", error=str(e), log=log)
    finally:
        state.pop("step", None)
        state["finished_at"] = time.time()
        write_training_state(state)
        shutil.rmtree(candidate, ignore_errors=True)
        is_training = False

def export_training_data():
    """Export a sample of confident sightings per cat, and drop corrections of relabeled sightings"""
    print("Exporting training data...")
    
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        c.execute("SELECT id, cat_name FROM sightings")
        current_labels = {row[0]: row[1] for row in c.fetchall()}
        
        # Up to SAMPLE_PER_CAT confident sightings per cat, at random so they span day and night.
        # Corrected sightings (confidence -1) are already in corrections/
        samples = {}
        for label in labels:
            c.execute('''
                SELECT id, thumbnail
                FROM sightings
                WHERE cat_name = ? AND confidence >= ? AND thumbnail IS NOT NULL
                ORDER BY RANDOM()
                LIMIT ?
            ''', (label, SAMPLE_MIN_CONFIDENCE, SAMPLE_PER_CAT))
            samples[label] = c.fetchall()
        
        conn.close()
    
    # Drop corrections whose sighting has since been relabeled (or marked not a cat), so it
    # isn't trained under its old label too. Files are named <id>_<time>.jpg
    for filepath in TRAINING_DIR.glob("corrections/*/*.jpg"):
        stem = filepath.stem.split('_')[0]
        if stem.isdigit() and int(stem) in current_labels and current_labels[int(stem)] != filepath.parent.name:
            filepath.unlink(missing_ok=True)
    
    # A fresh sample each time
    current_dir = TRAINING_DIR / "current"
    shutil.rmtree(current_dir, ignore_errors=True)
    for label, rows in samples.items():
        training_path = current_dir / label
        training_path.mkdir(parents=True, exist_ok=True)
        for sighting_id, thumbnail in rows:
            (training_path / f"{sighting_id}.jpg").write_bytes(thumbnail)
    
    print(f"Training data exported: {({label: len(rows) for label, rows in samples.items()})}")

@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """Hot-reload the model without restarting"""
    load_model()
    return jsonify({"success": True, "message": "Model reloaded"})

if __name__ == '__main__':
    print("Initializing Cat Identifier...")
    init_db()
    load_model()
    
    state = read_training_state()
    if state.get("status") == "running":
        state.update(status="failed", error="Interrupted by a restart", finished_at=time.time())
        state.pop("step", None)
        write_training_state(state)
    
    # Start background polling
    polling_thread = threading.Thread(target=polling_loop, daemon=True)
    polling_thread.start()
    
    # Run Flask
    app.run(host='0.0.0.0', port=8080, debug=False)

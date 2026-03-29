#!/usr/bin/env python3
"""
Cat Identifier - Track when each cat was last seen using Frigate events
"""
import os
import json
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

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Global state
db_lock = threading.Lock()
model_lock = threading.Lock()
model_session = None
labels = []
is_training = False

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
    global model_session, labels
    
    # Check for model in MODELS_DIR first (for hot-reload after training)
    model_path = MODELS_DIR / "cat_classifier.onnx" if (MODELS_DIR / "cat_classifier.onnx").exists() else MODEL_PATH
    
    with model_lock:
        if model_path.exists():
            model_session = ort.InferenceSession(str(model_path))
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
    """Get recent sightings with images"""
    limit = request.args.get('limit', 20, type=int)
    
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        c.execute('''
            SELECT id, event_id, cat_name, confidence, camera, timestamp
            FROM sightings
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
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
    
    if correct_label not in labels:
        return jsonify({"error": f"Invalid label. Must be one of: {labels}"}), 400
    
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
    
    # Save the image to training directory
    if thumbnail:
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
            GROUP BY cat_name
        ''')
        totals = {row[0]: row[1] for row in c.fetchall()}
        
        # Corrections count
        c.execute("SELECT COUNT(*) FROM feedback")
        corrections = c.fetchone()[0]
        
        # Today's sightings
        today_start = time.time() - (time.time() % 86400)
        c.execute('''
            SELECT cat_name, COUNT(*) as count
            FROM sightings
            WHERE timestamp >= ?
            GROUP BY cat_name
        ''', (today_start,))
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

@app.route('/api/retrain/status')
def retrain_status():
    """Get retraining status and correction counts"""
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        # Count corrections per cat
        c.execute('''
            SELECT correct_label, COUNT(*) 
            FROM feedback 
            GROUP BY correct_label
        ''')
        corrections_by_cat = {row[0]: row[1] for row in c.fetchall()}
        
        # Count original training images
        original_counts = {}
        if TRAINING_DIR.exists():
            for label in labels:
                original_path = TRAINING_DIR / "original" / label
                if original_path.exists():
                    original_counts[label] = len(list(original_path.glob("*.jpg")))
                corrections_path = TRAINING_DIR / "corrections" / label
                if corrections_path.exists():
                    corrections_by_cat[label] = corrections_by_cat.get(label, 0)
        
        c.execute("SELECT COUNT(*) FROM feedback")
        total_corrections = c.fetchone()[0]
        
        conn.close()
    
    return jsonify({
        "is_training": is_training,
        "total_corrections": total_corrections,
        "corrections_by_cat": corrections_by_cat,
        "original_counts": original_counts,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "can_retrain": total_corrections > 0 and not is_training
    })

@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining"""
    global is_training
    
    if is_training:
        return jsonify({"error": "Training already in progress"}), 409
    
    # Check if we have corrections
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM feedback")
        corrections = c.fetchone()[0]
        conn.close()
    
    if corrections == 0:
        return jsonify({"error": "No corrections to train on"}), 400
    
    # Start training in background
    is_training = True
    threading.Thread(target=run_training, daemon=True).start()
    
    return jsonify({
        "success": True, 
        "message": f"Training started with {corrections} corrections",
        "status_url": "/api/retrain/status"
    })

def run_training():
    """Run the training process"""
    global is_training
    
    try:
        print("Starting model retraining...")
        
        # Ensure directories exist
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Export all sightings with current labels as training data
        export_training_data()
        
        # Run the training script
        import subprocess
        result = subprocess.run(
            ["python3", "/app/train.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print("Training completed successfully!")
            print(result.stdout)
            
            # Reload the model
            load_model()
            print("Model reloaded")
        else:
            print(f"Training failed: {result.stderr}")
            
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        is_training = False

def export_training_data():
    """Export current sightings as training data"""
    print("Exporting training data...")
    
    with db_lock:
        conn = get_db()
        c = conn.cursor()
        
        # Get all sightings with their current labels
        c.execute('''
            SELECT id, cat_name, thumbnail 
            FROM sightings 
            WHERE thumbnail IS NOT NULL
        ''')
        
        for row in c.fetchall():
            sighting_id, label, thumbnail = row
            if thumbnail and label in labels:
                training_path = TRAINING_DIR / "current" / label
                training_path.mkdir(parents=True, exist_ok=True)
                
                filepath = training_path / f"{sighting_id}.jpg"
                if not filepath.exists():
                    with open(filepath, 'wb') as f:
                        f.write(thumbnail)
        
        conn.close()
    
    print("Training data exported")

@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """Hot-reload the model without restarting"""
    load_model()
    return jsonify({"success": True, "message": "Model reloaded"})

if __name__ == '__main__':
    print("Initializing Cat Identifier...")
    init_db()
    load_model()
    
    # Start background polling
    polling_thread = threading.Thread(target=polling_loop, daemon=True)
    polling_thread.start()
    
    # Run Flask
    app.run(host='0.0.0.0', port=8080, debug=False)

FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU version first
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies from PyPI
RUN pip install --no-cache-dir \
    flask \
    requests \
    onnxruntime \
    pillow \
    numpy

# Copy application
COPY app.py .
COPY train.py .
COPY static/ static/
COPY models/ models/

# Create directories
RUN mkdir -p /data /training /models

ENV FRIGATE_URL=http://10.0.1.2:5000
ENV POLL_INTERVAL=120
ENV DATA_DIR=/data
ENV TRAINING_DIR=/training
ENV MODELS_DIR=/models
ENV MODEL_PATH=/app/models/cat_classifier.onnx
ENV LABELS_PATH=/app/models/cat_classifier_labels.txt
ENV RETRAIN_THRESHOLD=10

EXPOSE 8080

CMD ["python", "app.py"]

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    flask \
    requests \
    onnxruntime \
    pillow \
    numpy

# Copy application
COPY app.py .
COPY static/ static/
COPY models/ models/

# Create data directory
RUN mkdir -p /data

ENV FRIGATE_URL=http://10.0.1.2:5000
ENV POLL_INTERVAL=120
ENV DATA_DIR=/data
ENV MODEL_PATH=/app/models/cat_classifier.onnx
ENV LABELS_PATH=/app/models/cat_classifier_labels.txt

EXPOSE 8080

CMD ["python", "app.py"]

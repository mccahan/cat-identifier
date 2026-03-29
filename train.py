#!/usr/bin/env python3
"""
Cat Classifier Training Script
Trains a MobileNetV2-based classifier on cat images
"""
import os
import sys
from pathlib import Path
import random

# Check for required packages
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install torch torchvision pillow numpy")
    sys.exit(1)

# Configuration
TRAINING_DIR = Path(os.getenv("TRAINING_DIR", "/training"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
LABELS = ["hawthorne", "roxie", "sadie"]
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

class CatDataset(Dataset):
    """Dataset for cat images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.label_to_idx[label]
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            if self.transform:
                image = self.transform(image)
            return image, self.label_to_idx[label]

def collect_training_images():
    """Collect all training images from various sources"""
    image_paths = []
    labels = []
    
    # Sources to check (in priority order)
    sources = [
        TRAINING_DIR / "corrections",  # Human-corrected images (highest priority)
        TRAINING_DIR / "current",       # Current sightings with labels
        TRAINING_DIR / "original",      # Original training data
    ]
    
    for source in sources:
        if not source.exists():
            continue
            
        for label in LABELS:
            label_dir = source / label
            if not label_dir.exists():
                continue
                
            for img_file in label_dir.glob("*.jpg"):
                image_paths.append(str(img_file))
                labels.append(label)
    
    print(f"Collected {len(image_paths)} training images")
    
    # Count per label
    label_counts = {label: labels.count(label) for label in LABELS}
    print(f"Per-label counts: {label_counts}")
    
    return image_paths, labels

def train_model():
    """Train the cat classifier"""
    print("=" * 50)
    print("Cat Classifier Training")
    print("=" * 50)
    
    # Collect training data
    image_paths, labels = collect_training_images()
    
    if len(image_paths) < 10:
        print("Not enough training images (minimum 10)")
        sys.exit(1)
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Split into train/val
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    
    split_idx = int(0.8 * len(image_paths))
    train_paths, train_labels = image_paths[:split_idx], labels[:split_idx]
    val_paths, val_labels = image_paths[split_idx:], labels[split_idx:]
    
    print(f"Training: {len(train_paths)}, Validation: {len(val_paths)}")
    
    # Create datasets and loaders
    train_dataset = CatDataset(train_paths, train_labels, train_transform)
    val_dataset = CatDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    print("Loading MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, len(LABELS))
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    
    # Load best model and export
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save PyTorch model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "cat_classifier.pth")
    print(f"Saved PyTorch model to {MODELS_DIR / 'cat_classifier.pth'}")
    
    # Export to ONNX
    model.eval()
    model = model.to("cpu")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(MODELS_DIR / "cat_classifier.onnx"),
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        dynamo=False  # Use legacy exporter to embed weights
    )
    
    # Verify ONNX file size (should be ~8MB with weights)
    onnx_size = (MODELS_DIR / "cat_classifier.onnx").stat().st_size
    print(f"Exported ONNX model: {onnx_size / 1024 / 1024:.1f} MB")
    
    if onnx_size < 1_000_000:  # Less than 1MB means weights weren't embedded
        print("WARNING: ONNX file seems too small, weights may not be embedded!")
    
    # Save labels
    with open(MODELS_DIR / "cat_classifier_labels.txt", 'w') as f:
        f.write('\n'.join(LABELS))
    
    print(f"Saved labels to {MODELS_DIR / 'cat_classifier_labels.txt'}")
    print("\nTraining complete!")

if __name__ == '__main__':
    train_model()

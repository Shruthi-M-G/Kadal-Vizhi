import os
import torch
import torch.serialization
import functools

# 1. Environment variable & Safety Bypass
os.environ['TORCH_USE_WEIGHTS_ONLY_LOAD'] = '0'
original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load
torch.serialization.load = patched_load

from ultralytics import YOLO

def train_fish():
    print("Model load aagudhu, konjam porunga...")
    model = YOLO('yolov8n.pt') 

    print("Training aarambikkudhu! Accuracy-ah improve panna parameters add panniruken.")
    
    # Indha section-la dhaan indentation fix panniruken
    model.train(
        data='data.yaml',
        epochs=150,
        imgsz=640,
        batch=8,
        mosaic=1.0,      # Mixes images to help detect overlapping fish
        degrees=15.0,     # Random rotation
        flipud=0.5,      # Vertical flip
        fliplr=0.5,      # Horizontal flip
        scale=0.5,       # Perspective scaling
        patience=30,     # Auto-stop if no improvement
        lr0=0.001,       # Learning rate-ah konjam kammi panniruken for stability
        device='cpu'
    )

if __name__ == '__main__':
    train_fish()
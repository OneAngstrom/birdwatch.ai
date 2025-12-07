"""
Configuration settings for Birdwatch.ai
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"
DETECTED_BIRDS_DIR = DATA_DIR / "detected_birds"

# Ensure directories exist
DETECTED_BIRDS_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
YOLO_MODEL = "yolov8n.pt"  # Nano model for faster inference
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for bird detection

# Bird class ID in COCO dataset (YOLOv8 is trained on COCO)
BIRD_CLASS_ID = 14  # 'bird' class in COCO dataset

# Webcam settings
WEBCAM_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Bird classification settings
# Using a simple mapping for demonstration
# In production, you would use a more sophisticated bird species classifier
BIRD_SPECIES = [
    "Unknown Bird",
    "Sparrow",
    "Robin",
    "Blue Jay",
    "Cardinal",
    "Crow",
    "Pigeon",
    "Hawk",
    "Eagle",
    "Owl",
]

# Streamlit settings
PAGE_TITLE = "Birdwatch.ai"
PAGE_ICON = "🐦"

"""
Utility functions for Birdwatch.ai
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from birdwatch.config import DETECTED_BIRDS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_bird_image(image: np.ndarray, species: str) -> Optional[str]:
    """
    Save a detected bird image to disk

    Args:
        image: Bird image to save
        species: Classified bird species name

    Returns:
        Path to saved image or None if failed
    """
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        species_clean = species.replace(" ", "_").lower()
        filename = f"{species_clean}_{timestamp}.jpg"
        filepath = DETECTED_BIRDS_DIR / filename

        # Save image
        cv2.imwrite(str(filepath), image)
        logger.info(f"Saved bird image: {filepath}")

        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving bird image: {e}")
        return None


def convert_cv2_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR image to RGB

    Args:
        image: BGR image from OpenCV

    Returns:
        RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio

    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def load_saved_birds() -> list:
    """
    Load information about previously detected birds

    Returns:
        List of dictionaries containing bird information
    """
    birds = []
    
    try:
        if not DETECTED_BIRDS_DIR.exists():
            return birds
        
        # Get all image files sorted by modification time (newest first)
        image_files = sorted(
            DETECTED_BIRDS_DIR.glob("*.jpg"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        
        for img_path in image_files:
            # Extract species from filename
            filename = img_path.stem
            parts = filename.split("_")
            
            # Reconstruct species name
            if len(parts) >= 3:
                species_parts = parts[:-3]  # Remove timestamp parts
                species = " ".join(species_parts).title()
            else:
                species = "Unknown"
            
            # Get timestamp
            timestamp = datetime.fromtimestamp(img_path.stat().st_mtime)
            
            birds.append({
                "species": species,
                "timestamp": timestamp,
                "image_path": str(img_path),
            })
    
    except Exception as e:
        logger.error(f"Error loading saved birds: {e}")
    
    return birds

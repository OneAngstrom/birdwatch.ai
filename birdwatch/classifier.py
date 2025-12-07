"""
Bird classification module
"""

import numpy as np
import random
from typing import Optional
import logging

from birdwatch.config import BIRD_SPECIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BirdClassifier:
    """Classifies bird species from cropped images"""

    def __init__(self):
        """Initialize the bird classifier"""
        # In a production system, this would load a trained bird species classifier
        # For this demo, we'll use a simple random classifier
        logger.info("Bird classifier initialized")

    def classify(self, bird_image: np.ndarray) -> str:
        """
        Classify a bird species from a cropped image

        Args:
            bird_image: Cropped image of a detected bird

        Returns:
            Bird species name
        """
        try:
            # In production, this would use a real classifier model
            # For demo purposes, we'll analyze basic features and return a species
            
            # Simple heuristic based on color analysis
            species = self._simple_classification(bird_image)
            
            return species

        except Exception as e:
            logger.error(f"Error during bird classification: {e}")
            return "Unknown Bird"

    def _simple_classification(self, bird_image: np.ndarray) -> str:
        """
        Simple classification based on image features
        
        This is a placeholder for a real classifier.
        In production, you would use a trained CNN model for bird species classification.
        """
        # Calculate average color values
        avg_color = np.mean(bird_image, axis=(0, 1))
        
        # Simple heuristic: classify based on dominant color
        # This is just for demonstration purposes
        b, g, r = avg_color
        
        # Blue-ish birds
        if b > r and b > g:
            return random.choice(["Blue Jay", "Unknown Bird"])
        # Red-ish birds
        elif r > b and r > g:
            return random.choice(["Cardinal", "Robin", "Unknown Bird"])
        # Dark birds
        elif avg_color.mean() < 80:
            return random.choice(["Crow", "Unknown Bird"])
        # Light/gray birds
        else:
            return random.choice(["Sparrow", "Pigeon", "Unknown Bird"])

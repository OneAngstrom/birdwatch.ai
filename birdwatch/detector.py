"""
Bird detection module using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging

from birdwatch.config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    BIRD_CLASS_ID,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BirdDetector:
    """Detects birds in images using YOLOv8"""

    def __init__(self):
        """Initialize the bird detector"""
        try:
            self.model = YOLO(YOLO_MODEL)
            logger.info(f"Loaded YOLO model: {YOLO_MODEL}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

    def detect_birds(self, frame: np.ndarray) -> Tuple[bool, List[dict], np.ndarray]:
        """
        Detect birds in a frame

        Args:
            frame: Input image frame (BGR format from OpenCV)

        Returns:
            Tuple of (has_birds, detections, annotated_frame)
            - has_birds: Boolean indicating if any birds were detected
            - detections: List of detection dictionaries with bbox and confidence
            - annotated_frame: Frame with bounding boxes drawn
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False)

            # Process results
            detections = []
            has_birds = False
            annotated_frame = frame.copy()

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class ID and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Check if it's a bird and above confidence threshold
                    if cls == BIRD_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
                        has_birds = True
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detection = {
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                            "class_id": cls,
                        }
                        detections.append(detection)

                        # Draw bounding box on frame
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Bird: {conf:.2f}"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

            return has_birds, detections, annotated_frame

        except Exception as e:
            logger.error(f"Error during bird detection: {e}")
            return False, [], frame

    def extract_bird_images(
        self, frame: np.ndarray, detections: List[dict]
    ) -> List[np.ndarray]:
        """
        Extract cropped images of detected birds

        Args:
            frame: Input image frame
            detections: List of detection dictionaries

        Returns:
            List of cropped bird images
        """
        bird_images = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            # Add some padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            bird_img = frame[y1:y2, x1:x2]
            bird_images.append(bird_img)

        return bird_images

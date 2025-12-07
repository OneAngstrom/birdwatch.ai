"""
Webcam capture module
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

from birdwatch.config import (
    WEBCAM_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebcamCapture:
    """Handles webcam video capture"""

    def __init__(self, camera_id: int = WEBCAM_ID):
        """
        Initialize webcam capture

        Args:
            camera_id: ID of the camera to use (default: 0)
        """
        self.camera_id = camera_id
        self.cap = None
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize the camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.warning(f"Could not open camera {self.camera_id}")
                # For demo purposes without a camera, we'll create a dummy frame
                self.cap = None
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, FPS)
                logger.info(f"Camera {self.camera_id} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the webcam

        Returns:
            Tuple of (success, frame)
            - success: Boolean indicating if frame was read successfully
            - frame: The captured frame or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            # Return a dummy frame for testing without a camera
            return True, self._create_dummy_frame()

        try:
            ret, frame = self.cap.read()
            if ret:
                return True, frame
            else:
                logger.warning("Failed to read frame from camera")
                return False, None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def _create_dummy_frame(self) -> np.ndarray:
        """
        Create a dummy frame for testing without a camera
        
        Returns:
            A placeholder frame
        """
        # Create a frame with gradient background
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
        # Add gradient
        for i in range(FRAME_HEIGHT):
            frame[i, :] = [50 + i // 3, 100 + i // 5, 150 + i // 4]
        
        # Add text
        text = "No Camera Connected - Demo Mode"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (FRAME_WIDTH - text_size[0]) // 2
        text_y = (FRAME_HEIGHT + text_size[1]) // 2
        
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        
        return frame

    def release(self):
        """Release the camera"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")

    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.release()

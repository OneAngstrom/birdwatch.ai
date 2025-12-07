"""
Main Streamlit application for Birdwatch.ai
"""

import streamlit as st
import cv2
import time
from datetime import datetime
from PIL import Image
import numpy as np

from birdwatch.config import PAGE_TITLE, PAGE_ICON
from birdwatch.detector import BirdDetector
from birdwatch.classifier import BirdClassifier
from birdwatch.webcam import WebcamCapture
from birdwatch.utils import (
    save_bird_image,
    convert_cv2_to_rgb,
    load_saved_birds,
)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "detector" not in st.session_state:
        st.session_state.detector = None
    if "classifier" not in st.session_state:
        st.session_state.classifier = None
    if "webcam" not in st.session_state:
        st.session_state.webcam = None
    if "detected_birds" not in st.session_state:
        st.session_state.detected_birds = []
    if "running" not in st.session_state:
        st.session_state.running = False


def initialize_models():
    """Initialize detection and classification models"""
    if st.session_state.detector is None:
        with st.spinner("Loading bird detection model..."):
            st.session_state.detector = BirdDetector()
    
    if st.session_state.classifier is None:
        with st.spinner("Loading bird classification model..."):
            st.session_state.classifier = BirdClassifier()
    
    if st.session_state.webcam is None:
        st.session_state.webcam = WebcamCapture()


def main():
    """Main application function"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
    )

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("### Real-time Bird Detection and Classification")

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        if st.button("Start Detection" if not st.session_state.running else "Stop Detection"):
            st.session_state.running = not st.session_state.running
        
        st.divider()
        
        st.header("Settings")
        show_detections = st.checkbox("Show Bounding Boxes", value=True)
        auto_save = st.checkbox("Auto-save Detected Birds", value=True)
        
        st.divider()
        
        st.header("Statistics")
        st.metric("Total Birds Detected", len(st.session_state.detected_birds))

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Webcam Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col2:
        st.subheader("Recently Detected Birds")
        birds_placeholder = st.empty()

    # Initialize models if running
    if st.session_state.running:
        initialize_models()

        # Read frame from webcam
        success, frame = st.session_state.webcam.read_frame()
        
        if not success or frame is None:
            status_placeholder.error("Failed to read frame from webcam")
        else:
            # Detect birds in the frame
            has_birds, detections, annotated_frame = st.session_state.detector.detect_birds(frame)

            # Display appropriate frame
            display_frame = annotated_frame if show_detections else frame
            video_placeholder.image(
                convert_cv2_to_rgb(display_frame),
                channels="RGB",
                use_container_width=True,
            )

            # Process detected birds
            if has_birds and detections:
                status_placeholder.success(f"🐦 Detected {len(detections)} bird(s)!")
                
                # Extract and classify birds
                bird_images = st.session_state.detector.extract_bird_images(frame, detections)
                
                for i, (bird_img, detection) in enumerate(zip(bird_images, detections)):
                    # Classify the bird
                    species = st.session_state.classifier.classify(bird_img)
                    
                    # Create bird record
                    bird_record = {
                        "species": species,
                        "confidence": detection["confidence"],
                        "timestamp": datetime.now(),
                        "image": bird_img,
                    }
                    
                    # Save image if auto-save is enabled
                    if auto_save:
                        image_path = save_bird_image(bird_img, species)
                        bird_record["image_path"] = image_path
                    
                    # Add to detected birds (avoid duplicates in quick succession)
                    # Check if any recent detection (within last 2 seconds) is the same species
                    should_add = True
                    current_time = bird_record["timestamp"]
                    for recent_bird in st.session_state.detected_birds[:3]:  # Check last 3 detections
                        time_diff = (current_time - recent_bird["timestamp"]).total_seconds()
                        if time_diff < 2.0 and species == recent_bird["species"]:
                            should_add = False
                            break
                    
                    if should_add:
                        st.session_state.detected_birds.insert(0, bird_record)
                        # Keep only last 50 detections
                        st.session_state.detected_birds = st.session_state.detected_birds[:50]
            else:
                status_placeholder.info("No birds detected. Keep watching...")

        # Display detected birds list
        with birds_placeholder.container():
            if st.session_state.detected_birds:
                for idx, bird in enumerate(st.session_state.detected_birds[:10]):  # Show top 10
                    with st.container():
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            # Display bird image
                            bird_img_rgb = convert_cv2_to_rgb(bird["image"])
                            st.image(bird_img_rgb, use_container_width=True)
                        
                        with col_b:
                            st.markdown(f"**{bird['species']}**")
                            st.caption(f"Confidence: {bird['confidence']:.2%}")
                            st.caption(bird["timestamp"].strftime("%H:%M:%S"))
                        
                        if idx < min(len(st.session_state.detected_birds), 10) - 1:
                            st.divider()
            else:
                st.info("No birds detected yet. Start detection to begin watching!")

        # Auto-refresh to continue detection
        time.sleep(0.1)  # Small delay to avoid overwhelming the system
        st.rerun()
    else:
        # Show placeholder when not running
        video_placeholder.info("Click 'Start Detection' to begin watching for birds")
        
        # Load and display previously saved birds
        with birds_placeholder.container():
            saved_birds = load_saved_birds()
            if saved_birds:
                st.markdown("**Previously Detected Birds:**")
                for bird in saved_birds[:10]:
                    with st.container():
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            try:
                                img = Image.open(bird["image_path"])
                                st.image(img, use_container_width=True)
                            except Exception:
                                st.write("Image not available")
                        
                        with col_b:
                            st.markdown(f"**{bird['species']}**")
                            st.caption(bird["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
                        
                        st.divider()
            else:
                st.info("No saved bird detections yet.")


if __name__ == "__main__":
    main()

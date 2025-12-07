# 🐦 Birdwatch.ai

A real-time bird detection and classification application that uses computer vision to identify birds from a webcam feed.

## Features

- **Real-time Bird Detection**: Uses YOLOv8 for detecting birds in webcam feed
- **Bird Classification**: Classifies detected birds by species
- **Live Web Interface**: Streamlit-based UI showing live feed and detected birds
- **Auto-save**: Automatically saves images of detected birds
- **Detection History**: View recently detected birds with timestamps

## Requirements

- Python 3.10 or higher
- Webcam (optional - runs in demo mode without one)
- UV package manager

## Installation

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone https://github.com/OneAngstrom/birdwatch.ai.git
cd birdwatch.ai
```

3. **Install dependencies using UV**:
```bash
uv sync
```

This will create a virtual environment and install all required dependencies including:
- OpenCV for image processing
- YOLOv8 (Ultralytics) for object detection
- Streamlit for the web interface
- PyTorch for deep learning models
- And other necessary packages

## Usage

### Running the Application

Start the Streamlit application:

```bash
uv run streamlit run birdwatch/app.py
```

Or use the installed script:

```bash
uv run birdwatch
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Start Detection**: Click the "Start Detection" button in the sidebar
2. **View Live Feed**: The main area shows your webcam feed with bird detections highlighted
3. **Monitor Detections**: Recently detected birds appear in the right panel with their species and confidence scores
4. **Settings**: 
   - Toggle "Show Bounding Boxes" to show/hide detection boxes
   - Toggle "Auto-save Detected Birds" to automatically save bird images

### Demo Mode

If no webcam is connected, the application runs in demo mode with a placeholder frame. This is useful for testing the interface.

## Project Structure

```
birdwatch.ai/
├── birdwatch/
│   ├── __init__.py
│   ├── app.py              # Main Streamlit application
│   ├── detector.py         # Bird detection using YOLOv8
│   ├── classifier.py       # Bird species classification
│   ├── webcam.py          # Webcam capture module
│   ├── config.py          # Configuration settings
│   └── utils.py           # Utility functions
├── data/
│   └── detected_birds/    # Saved bird images
├── pyproject.toml         # UV project configuration
└── README.md
```

## How It Works

1. **Frame Capture**: The application continuously reads frames from the connected webcam
2. **Bird Detection**: Each frame is processed by YOLOv8 to detect birds (class ID 14 in COCO dataset)
3. **Classification**: When a bird is detected, the cropped image is sent to the classifier to identify the species
4. **Display**: The results are displayed in real-time on the Streamlit interface
5. **Storage**: Detected bird images are automatically saved to the `data/detected_birds/` directory

## Configuration

You can modify settings in `birdwatch/config.py`:

- `CONFIDENCE_THRESHOLD`: Minimum confidence for bird detection (default: 0.5)
- `WEBCAM_ID`: Camera device ID (default: 0)
- `FRAME_WIDTH`, `FRAME_HEIGHT`: Video resolution
- `YOLO_MODEL`: YOLOv8 model variant (default: yolov8n.pt for speed)

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black birdwatch/
uv run ruff check birdwatch/
```

## Notes

- The bird classification module currently uses a simple heuristic classifier for demonstration purposes
- For production use, you should integrate a proper bird species classification model trained on a comprehensive bird dataset
- The YOLOv8 model will be automatically downloaded on first run
- Detected bird images are saved with timestamps in the `data/detected_birds/` directory

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

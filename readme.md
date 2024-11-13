# Motion Barcode Detection System with Object Tracking

A comprehensive system for monitoring objects moving in and out of a space (e.g., a fridge) using motion detection, barcode scanning, and object tracking with YOLO.

## Overview

This system uses a Raspberry Pi camera to:
1. Detect motion in the monitored space
2. Record video when motion is detected
3. Scan for barcodes during recording
4. Track objects moving in/out using YOLO object detection
5. Store and analyze the collected data

## Features

- **Motion Detection**
  - Configurable motion sensitivity
  - Automatic recording start/stop
  - Minimum recording duration enforcement

- **Barcode Scanning**
  - Real-time barcode detection
  - Multiple barcode format support
  - Duplicate detection prevention

- **Object Tracking**
  - YOLO-based object detection
  - Direction tracking (in/out)
  - Object path visualization
  - JSON format result storage

- **Data Management**
  - Local JSON storage
  - Optional server upload
  - Automatic file organization
  - Robust error handling

## Prerequisites

### Hardware Requirements
- Raspberry Pi (3 or newer recommended)
- Raspberry Pi Camera Module
- Adequate storage for recordings

### Software Requirements
```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv

# Python packages
pip3 install -r requirements.txt
```

### Required Python Packages
```
picamera2
opencv-python
numpy
pyzbar
ultralytics
requests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Glebenator/FF-modular.git
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Set up the YOLO model:
```bash
# Create models directory
mkdir -p models

# Download YOLO model to models directory
# Place your YOLO model files in the models/yolo11n_ncnn_model directory
```

4. Configure settings:
```bash
# Edit config/settings.py to match your environment
# Key settings to review:
# - Camera resolution
# - Motion detection sensitivity
# - Server URL (if using)
# - File paths
```

## Directory Structure
```
project_folder/
│
├── main.py                 # Main application entry point
├── config/                 # Configuration files
├── core/                   # Core functionality modules
├── network/               # Network communication
├── utils/                 # Utility functions
├── models/                # YOLO model files
├── recordings/            # Recorded data
└── logs/                  # System logs
```

## Usage

1. Start the system:
```bash
python3 main.py
```

2. Monitor logs:
```bash
tail -f logs/motion_barcode_YYYYMMDD.log
```

3. View results:
- Recorded videos: `recordings/videos/raw/`
- Analyzed videos: `recordings/videos/analyzed/`
- Barcode data: `recordings/json/barcodes/`
- Tracking data: `recordings/json/tracking/`

## Configuration

### Camera Settings (config/settings.py)
```python
class CameraConfig:
    MAIN_RESOLUTION = (640, 640)
    LORES_RESOLUTION = (320, 240)
    FRAME_RATE = 120
```

### Processing Settings
```python
class ProcessingConfig:
    MOTION_THRESHOLD = 30
    MOTION_TIMEOUT = 11
    VIDEO_PROCESSING_STRIDE = 5
```

### Network Settings
```python
class NetworkConfig:
    SERVER_URL = "http://your-server.com"
    MAX_QUEUE_SIZE = 1000
```

## JSON Data Formats

### Barcode Detection
```json
{
    "session_start": "20241112_001826",
    "barcodes": [
        {
            "barcode": "123456789",
            "detection_time": 1699743506.789
        }
    ]
}
```

### Object Tracking
```json
{
    "video_file": "motion_20241112_001826.h264",
    "processing_time": 1699743506.789,
    "events": [
        {
            "timestamp": 1699743506.789,
            "track_id": 98,
            "object_type": "apple",
            "direction": "OUT"
        }
    ]
}
```

## Troubleshooting

### Common Issues

1. Camera Not Found
```bash
# Check camera connection
vcgencmd get_camera
```

2. Motion Detection Issues
- Adjust MOTION_THRESHOLD in settings.py
- Check ROI settings in motion_detector.py

3. YOLO Model Issues
- Verify model path in settings.py
- Check model compatibility
- Ensure sufficient system resources

### Logs

- System logs are stored in `logs/motion_barcode_YYYYMMDD.log`
- Include timestamps and component information
- Log rotation is automatic

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Your chosen license]

## Acknowledgments

- YOLO object detection
- OpenCV community
- Raspberry Pi community

## Support

For support, please:
1. Check the documentation
2. Review logs for specific errors
3. Open an issue with:
   - System configuration
   - Relevant logs
   - Steps to reproduce


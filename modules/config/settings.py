# motion_barcode_system/config/settings.py
"""Configuration settings for the motion barcode system."""
import os
from dataclasses import dataclass

@dataclass
class CameraConfig:
    MAIN_RESOLUTION = (640, 640)
    LORES_RESOLUTION = (320, 240)
    FRAME_RATE = 120
    EXPOSURE_TIME = 8000
    ANALOG_GAIN = 4.0
    SHARPNESS = 1.5

@dataclass
class ProcessingConfig:
    MOTION_THRESHOLD = 30
    MOTION_TIMEOUT = 6
    MIN_RECORDING_TIME = 5
    MOTION_CHECK_INTERVAL = 0.1
    BARCODE_SCAN_INTERVAL = 0.0001
    
    # Video processing settings
    YOLO_MODEL_PATH = os.getenv(
        "YOLO_MODEL_PATH",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/yolo11n_ncnn_model")
    )
    ENABLE_VIDEO_PROCESSING = True
    VIDEO_PROCESSING_CONFIDENCE = 0.3
    VIDEO_PROCESSING_IOU = 0.5
    MAX_DETECTIONS = 3
    VIDEO_PROCESSING_STRIDE = 4  # Process every Nth frame
    YOLO_TASK = "detect"  # Explicitly set YOLO task

@dataclass
class NetworkConfig:
    SERVER_URL = os.getenv("BARCODE_SERVER_URL", "http://glebs.net")
    MAX_QUEUE_SIZE = 1000
    MAX_RETRIES = 3
    RETRY_DELAY = 5.0
    CONNECTION_TIMEOUT = 10.0
    QUEUE_TIMEOUT = 0.5

@dataclass
class PathConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'recordings')
    VIDEO_DIR = os.path.join(OUTPUT_DIR, 'videos')
    VIDEO_RAW_DIR = os.path.join(VIDEO_DIR, 'raw')
    VIDEO_ANALYZED_DIR = os.path.join(VIDEO_DIR, 'analyzed')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    JSON_DIR = os.path.join(OUTPUT_DIR, 'json')
    JSON_BARCODES_DIR = os.path.join(JSON_DIR, 'barcodes')
    JSON_TRACKING_DIR = os.path.join(JSON_DIR, 'tracking')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

@dataclass
class HardwareConfig:
    RED_PIN = 17
    GREEN_PIN = 27
    BLUE_PIN = 22
    BUZZER_PIN = 18
    BUZZER_TONE = "A4"
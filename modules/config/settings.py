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
    MOTION_TIMEOUT = 11
    MIN_RECORDING_TIME = 6
    MOTION_CHECK_INTERVAL = 0.1
    BARCODE_SCAN_INTERVAL = 0.0001

@dataclass
class NetworkConfig:
    SERVER_URL = os.getenv("BARCODE_SERVER_URL", "http://your-server.com/api/barcodes")
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
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
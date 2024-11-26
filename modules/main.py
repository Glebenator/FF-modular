"""Main entry point for the motion barcode system."""
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pyzbar.pyzbar import decode
import logging

from config.settings import PathConfig, ProcessingConfig, NetworkConfig
from core.camera_manager import CameraManager
from core.motion_detector import MotionDetector
from core.barcode_processor import BarcodeProcessor
from core.video_processor import VideoProcessor
from network.json_sender import JSONSender
from utils.logging_config import setup_logging
from hardware.hardware_controller import HardwareController, LEDStatus

class MotionBarcodeSystem:
    def __init__(self):
        """Initialize the motion barcode system."""
        self.logger = setup_logging()
        self._setup_output_directories()
        
        # Initialize components
        self.hardware_controller = HardwareController()
        self.camera = CameraManager()
        self.motion_detector = MotionDetector(
            roi_x=60, roi_y=40, roi_width=200, roi_height=160
        )
        self.barcode_processor = BarcodeProcessor()
        self.json_sender = JSONSender(
            NetworkConfig.SERVER_URL,
            max_queue_size=NetworkConfig.MAX_QUEUE_SIZE,
            max_retries=NetworkConfig.MAX_RETRIES,
            retry_delay=NetworkConfig.RETRY_DELAY,
            connection_timeout=NetworkConfig.CONNECTION_TIMEOUT,
            queue_timeout=NetworkConfig.QUEUE_TIMEOUT
        )
        
        # Initialize video processor if enabled
        if ProcessingConfig.ENABLE_VIDEO_PROCESSING:
            self.video_processor = VideoProcessor(ProcessingConfig.YOLO_MODEL_PATH, self.hardware_controller)
        else:
            self.video_processor = None
            
        # Initialize state
        self.is_recording = False
        self.running = False
        self.last_motion_time = 0
        self.recording_start_time = 0
        self.current_session_timestamp = None
        self.current_video_path = None
        self.lock = threading.Lock()
        
    def _handle_json_send_failure(self, error_msg: str):
        """Handle JSON send failure."""
        self.logger.error(f"Failed to send barcode data: {error_msg}")
        self.hardware_controller.set_status(LEDStatus.ERROR)
        # Set a timer to revert to appropriate status after 5 seconds
        threading.Timer(5.0, self._revert_status).start()

    def _handle_json_send_success(self):
        """Handle JSON send success."""
        self.logger.info("Successfully sent barcode data")
        # Revert to appropriate status
        self._revert_status()

    def _revert_status(self):
        """Revert LED status to appropriate state based on current system state."""
        if self.is_recording:
            self.hardware_controller.set_status(LEDStatus.RECORDING)
        else:
            self.hardware_controller.set_status(LEDStatus.RUNNING)
    
    def _setup_output_directories(self):
        """Create necessary output directories."""
        os.makedirs(PathConfig.VIDEO_DIR, exist_ok=True)
        os.makedirs(PathConfig.MODELS_DIR, exist_ok=True)
        
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)
        
    def start_recording(self):
        """Start a new recording session."""
        try:
            self.current_session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_video_path = os.path.join(
                PathConfig.VIDEO_DIR,
                f'motion_{self.current_session_timestamp}.h264'
            )
            
            os.makedirs(os.path.dirname(self.current_video_path), exist_ok=True)
            
            self.encoder, self.output = self.camera.start_recording(self.current_video_path)
            self.barcode_processor.start_new_session()
            
            self.is_recording = True
            # Always set to RECORDING when starting a new recording
            self.hardware_controller.set_status(LEDStatus.RECORDING)
            self.logger.info(f"Started recording: {self.current_video_path}")
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.hardware_controller.set_status(LEDStatus.ERROR)
            time.sleep(5)
            raise
        
    
    def stop_recording(self):
        """Stop the current recording session."""
        if self.is_recording:
            self.camera.stop_recording()
            
            if self.barcode_processor.current_session_barcodes:
                session_data = {
                    "session_start": self.current_session_timestamp,
                    "items": self.barcode_processor.current_session_barcodes
                }
                self.json_sender.send_recording_data(
                    self.current_session_timestamp,
                    session_data
                )
            
            # Queue video for processing if enabled
            if (self.video_processor and 
                self.current_video_path and 
                os.path.exists(self.current_video_path)):
                self.video_processor.queue_video(self.current_video_path)
            
            self.barcode_processor.end_session()
            self.is_recording = False
            
            # Explicitly check processing state and set LED accordingly
            if self.video_processor and not self.video_processor.processing_queue.empty():
                self.hardware_controller.set_status(LEDStatus.PROCESSING)
            else:
                self.hardware_controller.set_status(LEDStatus.RUNNING)
                
            self.logger.info("Stopped recording")
            
    def scan_barcode(self, frame):
        """Scan for barcodes in the provided frame."""
        try:
            barcodes = decode(frame)
            for barcode in barcodes:
                barcode_data = barcode.data.decode("utf-8")
                if self.barcode_processor.process_barcode(barcode_data):
                    # Play success sound and temporarily show processing status
                    self.hardware_controller.play_barcode_sound()
                    self.logger.info(f"Detected barcode: {barcode_data}")
                    
        except Exception as e:
            self.logger.error(f"Error processing barcode: {e}")
            self.hardware_controller.set_status(LEDStatus.ERROR)
            threading.Timer(5.0, self._revert_status).start()
            
    def motion_detection_loop(self):
        """Main loop for motion detection."""
        while self.running:
            try:
                frame = self.camera.capture_frame("lores")
                motion_detected = self.motion_detector.detect_motion(frame)
                
                current_time = time.time()
                with self.lock:
                    if motion_detected:
                        self.last_motion_time = current_time
                        if not self.is_recording:
                            self.start_recording()
                    elif self.is_recording:
                        if (current_time - self.last_motion_time >
                            ProcessingConfig.MOTION_TIMEOUT):
                            self.stop_recording()
                        elif (current_time - self.recording_start_time <
                              ProcessingConfig.MIN_RECORDING_TIME):
                            self.last_motion_time = current_time
                            
            except Exception as e:
                self.logger.error(f"Error in motion detection loop: {e}")
                
            time.sleep(ProcessingConfig.MOTION_CHECK_INTERVAL)
            
    def barcode_scanning_loop(self):
        """Main loop for barcode scanning with priority over video processing."""
        while self.running:
            try:
                if self.is_recording:
                    # Pause video processing during barcode scan
                    if self.video_processor:
                        self.video_processor.pause_processing()
                        
                    frame = self.camera.capture_frame("main")
                    self.scan_barcode(frame)
                    
                    # Resume video processing
                    if self.video_processor:
                        self.video_processor.resume_processing()
                        
                    time.sleep(ProcessingConfig.BARCODE_SCAN_INTERVAL)
                else:
                    time.sleep(ProcessingConfig.MOTION_CHECK_INTERVAL)
            except Exception as e:
                self.logger.error(f"Error in barcode scanning loop: {e}")
                
    def start(self):
        """Start the motion barcode system."""
        try:
            self.running = True
            self.camera.start()
            
            # Start video processor if enabled
            if self.video_processor:
                self.video_processor.start()
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            
            # Start processing threads
            self.motion_thread = threading.Thread(target=self.motion_detection_loop)
            self.barcode_thread = threading.Thread(target=self.barcode_scanning_loop)
            
            self.motion_thread.start()
            self.barcode_thread.start()

            self.hardware_controller.set_status(LEDStatus.RUNNING)
            self.logger.info(f"System started - saving recordings to {PathConfig.OUTPUT_DIR}")
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.hardware_controller.set_status(LEDStatus.ERROR)
            raise
        
    def stop(self):
        """Stop the motion barcode system."""
        self.logger.info("Stopping system...")
        self.running = False
        
        if hasattr(self, 'motion_thread') and self.motion_thread.is_alive():
            self.motion_thread.join()
        if hasattr(self, 'barcode_thread') and self.barcode_thread.is_alive():
            self.barcode_thread.join()
            
        if self.is_recording:
            self.stop_recording()
            
        if self.video_processor:
            self.video_processor.stop()
            
        self.hardware_controller.set_status(LEDStatus.OFF)
        self.json_sender.stop()
        self.camera.stop()
        self.hardware_controller.cleanup()
        self.logger.info("System stopped")

def main():
    """Main entry point for the application."""
    system = None
    try:
        system = MotionBarcodeSystem()
        system.start()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except Exception as e:
        logging.error(f"Critical error: {e}")
        if system:
            system.hardware_controller.set_status(LEDStatus.ERROR)  # Set error status
        sys.exit(1)

if __name__ == "__main__":
    main()
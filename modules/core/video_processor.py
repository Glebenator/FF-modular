# motion_barcode_system/core/video_processor.py
"""Video processing module for post-recording analysis."""
import logging
import threading
import os
from queue import Queue
import time
from ultralytics import YOLO

from config.settings import PathConfig, ProcessingConfig
from core.videotrack import FridgeDirectionDetector, process_h264_video
from hardware.hardware_controller import LEDStatus

class VideoProcessor:
    def __init__(self, model_path, hardware_controller):
        """Initialize the video processor."""
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.processing_queue = Queue()
        self.running = False
        self.processing_thread = None
        self.hardware_controller = hardware_controller
        
    def start(self):
        """Start the video processing service."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Video processor started")
        
    def stop(self):
        """Stop the video processing service."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=30)
        self.logger.info("Video processor stopped")
        
    def queue_video(self, video_path):
        """Add a video to the processing queue."""
        try:
            if os.path.exists(video_path):
                self.processing_queue.put(video_path)
                self.logger.info(f"Queued video for processing: {video_path}")
            else:
                self.logger.error(f"Video file not found: {video_path}")
        except Exception as e:
            self.logger.error(f"Error queueing video: {e}")
            
    def _process_queue(self):
        """Process videos in the queue."""
        while self.running:
            try:
                if not self.processing_queue.empty():
                    # Set LED to processing status
                    self.hardware_controller.set_status(LEDStatus.PROCESSING)
                    
                    video_path = self.processing_queue.get()
                    self._process_video(video_path)
                    self.processing_queue.task_done()
                    
                    # Return to running status if queue is empty
                    if self.processing_queue.empty():
                        self.hardware_controller.set_status(LEDStatus.RUNNING)
                else:
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in processing queue: {e}")
                self.hardware_controller.set_status(LEDStatus.ERROR)
                time.sleep(5)  # Show error status briefly
                self.hardware_controller.set_status(LEDStatus.RUNNING)
                
    def _process_video(self, video_path):
        """Process a single video file."""
        try:
            self.logger.info(f"Processing video: {video_path}")
            process_h264_video(video_path, self.model_path)
            self.logger.info(f"Completed processing video: {video_path}")
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            self.hardware_controller.set_status(LEDStatus.ERROR)
            raise
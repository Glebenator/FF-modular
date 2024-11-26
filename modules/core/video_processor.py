# motion_barcode_system/core/video_processor.py
"""Video processing module for post-recording analysis."""
import logging
import threading
import os
import json
from queue import Queue
import time
from ultralytics import YOLO

from config.settings import PathConfig, ProcessingConfig, NetworkConfig
from core.videotrack import FridgeDirectionDetector, process_h264_video
from hardware.hardware_controller import LEDStatus
from network.json_sender import JSONSender

class VideoProcessor:
    def __init__(self, model_path, hardware_controller):
        """Initialize the video processor."""
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.processing_queue = Queue()
        self.running = False
        self.processing_thread = None
        self.hardware_controller = hardware_controller
        self.is_processing = False  # Add flag to track processing state
        
        # Initialize JSON sender for video processing results
        self.json_sender = JSONSender(
            NetworkConfig.SERVER_URL,
            max_queue_size=NetworkConfig.MAX_QUEUE_SIZE,
            max_retries=NetworkConfig.MAX_RETRIES,
            retry_delay=NetworkConfig.RETRY_DELAY,
            connection_timeout=NetworkConfig.CONNECTION_TIMEOUT,
            queue_timeout=NetworkConfig.QUEUE_TIMEOUT
        )

        # **NEW** Add priority control
        self.priority_lock = threading.Lock()
        self.processing_active = threading.Event()
        self.processing_pause = threading.Event()

    def pause_processing(self):
        """Pause the processing thread."""
        self.processing_pause.set()
    
    def resume_processing(self):
        """Resume the processing thread."""
        self.processing_pause.clear()

    def _handle_send_failure(self, error_msg: str):
        """Handle JSON send failure."""
        self.logger.error(f"Failed to send video processing results: {error_msg}")
        self.hardware_controller.set_status(LEDStatus.ERROR)
        # Set a timer to revert to RUNNING status after 5 seconds
        threading.Timer(5.0, self._revert_status).start()

    def _handle_send_success(self):
        """Handle JSON send success."""
        self.logger.info("Successfully sent video processing results")
        # Only set to RUNNING if we're not currently processing another video
        if self.processing_queue.empty():
            self.hardware_controller.set_status(LEDStatus.RUNNING)

    def _revert_status(self):
        """Revert LED status to RUNNING if not processing."""
        if self.processing_queue.empty():
            self.hardware_controller.set_status(LEDStatus.RUNNING)
            
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
        if not self.processing_queue.empty():
            # Maintain processing status if there are still items in queue
            self.hardware_controller.set_status(LEDStatus.PROCESSING)
        else:
            self.is_processing = False
            self.hardware_controller.set_status(LEDStatus.RUNNING)
        self.logger.info("Video processor stopped")
        
    def queue_video(self, video_path):
        """Add a video to the processing queue."""
        try:
            if os.path.exists(video_path):
                # Extract session timestamp from filename
                session_start = os.path.basename(video_path).split('_', 1)[1].rsplit('.', 1)[0]
                
                if not self.is_processing:
                    self.is_processing = True
                    self.hardware_controller.set_status(LEDStatus.PROCESSING)
                    
                # Add both video path and session timestamp to queue
                self.processing_queue.put((video_path, session_start))
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
                    video_path = self.processing_queue.get()
                    # Ensure LED is blue while processing
                    self.hardware_controller.set_status(LEDStatus.PROCESSING)
                    self._process_video(video_path)
                    self.processing_queue.task_done()
                    
                    if self.processing_queue.empty():
                        self.is_processing = False
                        # Only set to RUNNING if we're not recording
                        if not hasattr(self, 'is_recording') or not self.is_recording:
                            self.hardware_controller.set_status(LEDStatus.RUNNING)
                else:
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in processing queue: {e}")
                self.hardware_controller.set_status(LEDStatus.ERROR)
                time.sleep(5)
                # Return to appropriate status after error
                if not self.processing_queue.empty():
                    self.hardware_controller.set_status(LEDStatus.PROCESSING)
                elif hasattr(self, 'is_recording') and self.is_recording:
                    self.hardware_controller.set_status(LEDStatus.RECORDING)
                else:
                    self.hardware_controller.set_status(LEDStatus.RUNNING)
                
    def _process_video(self, video_data):
        """Process a single video file with pause support."""
        try:
            video_path, session_start = video_data  # Unpack the tuple
            self.logger.info(f"Processing video: {video_path}")
            
            # Only process when not paused
            while self.processing_pause.is_set():
                time.sleep(0.1)
            
            # Pass session_start to the processing function
            process_h264_video(video_path, self.model_path)
            
            # Get the results file path
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            results_path = os.path.join(
                PathConfig.JSON_DIR,
                f"{base_name}_tracking_results.json"
            )
            
            # Read and send the results
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results_data = json.load(f)
                    
                    # Send results directly using JSONSender
                    self.json_sender.send_recording_data(
                        f"video_processing_{base_name}",
                        results_data  # Contains exactly {session_start, items[{name, direction}]}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error sending video processing results: {e}")
                    self._handle_send_failure(str(e))
            else:
                self.logger.warning(f"No results file found for video: {video_path}")
                self._handle_send_failure("No results file found")
            
            self.logger.info(f"Completed processing video: {video_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            self.hardware_controller.set_status(LEDStatus.ERROR)
            raise
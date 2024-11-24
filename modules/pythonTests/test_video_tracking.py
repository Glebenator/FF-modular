# tests/test_video_tracking.py
import os
import sys
import logging
from datetime import datetime
import cv2
from pathlib import Path

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.videotrack import process_h264_video, FridgeDirectionDetector
from config.settings import PathConfig, ProcessingConfig
from utils.logging_config import setup_logging

class VideoTrackingTester:
    def __init__(self):
        """Initialize the video tracking tester."""
        self.logger = setup_logging()
        self.test_file = "motion_20241123_201458.h264"
        
    def setup_test_environment(self):
        """Set up necessary directories and paths for testing."""
        # Ensure all required directories exist
        os.makedirs(PathConfig.VIDEO_DIR, exist_ok=True)
        os.makedirs(PathConfig.VIDEO_ANALYZED_DIR, exist_ok=True)
        os.makedirs(PathConfig.JSON_DIR, exist_ok=True)
        
        # Get the full path to the test video
        self.test_video_path = os.path.join(PathConfig.VIDEO_DIR, self.test_file)
        
        # Check if test video exists
        if not os.path.exists(self.test_video_path):
            raise FileNotFoundError(f"Test video not found: {self.test_video_path}")
            
        # Set up output paths
        self.output_video_path = os.path.join(
            PathConfig.VIDEO_ANALYZED_DIR,
            self.test_file.replace('.h264', '_analyzed.mp4')
        )
        self.output_json_path = os.path.join(
            PathConfig.JSON_DIR,
            self.test_file.replace('.h264', '_tracking_results.json')
        )
        
        # Clean up previous test outputs
        self._cleanup_previous_outputs()
        
    def _cleanup_previous_outputs(self):
        """Remove previous test outputs if they exist."""
        for path in [self.output_video_path, self.output_json_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    self.logger.info(f"Cleaned up previous test output: {path}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {path}: {e}")
                    
    def validate_video_output(self):
        """Validate the processed video output."""
        if not os.path.exists(self.output_video_path):
            self.logger.error("Processed video not found!")
            return False
            
        try:
            cap = cv2.VideoCapture(self.output_video_path)
            if not cap.isOpened():
                self.logger.error("Cannot open processed video!")
                return False
                
            # Check video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Processed video properties:")
            self.logger.info(f"- Resolution: {width}x{height}")
            self.logger.info(f"- FPS: {fps}")
            self.logger.info(f"- Frame count: {frame_count}")
            
            # Check if video has frames
            if frame_count == 0:
                self.logger.error("Processed video has no frames!")
                return False
                
            # Sample some frames to ensure they contain annotations
            frames_to_check = min(10, frame_count)
            for i in range(frames_to_check):
                frame_pos = int((i / frames_to_check) * frame_count)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.logger.error(f"Cannot read frame at position {frame_pos}")
                    return False
                    
            cap.release()
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating video output: {e}")
            return False
            
    def print_tracking_results(self):
        """Print a summary of the tracking results."""
        import json
        
        try:
            if not os.path.exists(self.output_json_path):
                self.logger.error("Tracking results JSON not found!")
                return
                
            with open(self.output_json_path, 'r') as f:
                results = json.load(f)
                
            events = results.get('events', [])
            self.logger.info("\nTracking Results Summary:")
            self.logger.info(f"Total events detected: {len(events)}")
            
            # Count events by direction
            direction_counts = {'IN': 0, 'OUT': 0}
            object_counts = {}
            
            for event in events:
                direction = event.get('direction', 'UNKNOWN')
                object_type = event.get('object_type', 'UNKNOWN')
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
                object_counts[object_type] = object_counts.get(object_type, 0) + 1
                
            self.logger.info("\nEvents by direction:")
            for direction, count in direction_counts.items():
                self.logger.info(f"- {direction}: {count}")
                
            self.logger.info("\nEvents by object type:")
            for obj_type, count in object_counts.items():
                self.logger.info(f"- {obj_type}: {count}")
                
        except Exception as e:
            self.logger.error(f"Error reading tracking results: {e}")
            
    def run_test(self):
        """Run the video tracking test."""
        try:
            self.logger.info("Starting video tracking test...")
            self.setup_test_environment()
            
            # Process the video
            self.logger.info(f"Processing test video: {self.test_file}")
            success = process_h264_video(
                self.test_video_path,
                ProcessingConfig.YOLO_MODEL_PATH
            )
            
            if not success:
                self.logger.error("Video processing failed!")
                return False
                
            # Validate outputs
            self.logger.info("Validating video output...")
            if not self.validate_video_output():
                self.logger.error("Video validation failed!")
                return False
                
            # Print tracking results
            self.print_tracking_results()
            
            self.logger.info("Test completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Test failed with error: {e}")
            return False

def main():
    """Main entry point for testing."""
    tester = VideoTrackingTester()
    success = tester.run_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
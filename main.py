from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import controls
from picamera2.encoders import Quality
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time
import threading
import os
from datetime import datetime
from json_sender import JSONSender
from barcode_processor import BarcodeProcessor

class MotionBarcodeRecorder:
    def __init__(self, server_url, roi_x=0, roi_y=0, roi_width=None, roi_height=None):
        """Initialize the motion and barcode recording system."""
        self.picam2 = Picamera2()
        
        # Create output directories
        self.output_dir = os.path.join(os.getcwd(), 'recordings')
        os.makedirs(os.path.join(self.output_dir, 'videos'), exist_ok=True)
        
        # Initialize JSON sender and barcode processor
        self.json_sender = JSONSender(server_url)
        self.barcode_processor = BarcodeProcessor()
        
        # Configure camera streams:
        # - lores: motion detection (320x240)
        # - main: barcode scanning (1920x1080)
        # - video: high fps recording (640x640)
        self.config = self.picam2.create_video_configuration(
            main={"size": (640, 640), "format": "RGB888"},    # For high fps recording
            lores={"size": (320, 240), "format": "YUV420"},   # For motion detection
            encode="main",  # Use main stream for recording
            buffer_count=8,  # Increase buffer count for high fps
            controls={
                "FrameDurationLimits": (8333, 8333),  # Try for 120fps (1/120s = 8333μs)
                "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,
                "AfMode": controls.AfModeEnum.Continuous,
                "AfSpeed": controls.AfSpeedEnum.Fast,
                "ExposureTime": 8000,  # Fast exposure to minimize motion blur
                "AnalogueGain": 4.0,   # Increased gain to compensate for fast exposure
                "Sharpness": 1.5       # Slightly increased sharpness
        }
    )
        self.picam2.configure(self.config)
        
        # Get the actual stream configurations
        self.lores_size = self.picam2.stream_configuration("lores")["size"]
        
        # Set up ROI for motion detection
        if roi_width is None:
            roi_width = self.lores_size[0]
        if roi_height is None:
            roi_height = self.lores_size[1]
        self.roi = (roi_x, roi_y, roi_width, roi_height)
        
        # Initialize state variables
        self.is_recording = False
        self.last_motion_time = 0
        self.recording_start_time = 0
        self.encoder = None
        self.previous_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.current_session_timestamp = None

    def detect_motion_yuv(self, yuv_frame, threshold=30):
        """Detect motion in the YUV frame using the lores stream."""
        y_height = self.lores_size[1]
        y_width = self.lores_size[0]
        y_channel = yuv_frame[:y_height, :y_width]
        
        roi = y_channel[self.roi[1]:self.roi[1]+self.roi[3], 
                       self.roi[0]:self.roi[0]+self.roi[2]]
        
        blurred = cv2.GaussianBlur(roi, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = blurred
            return False
        
        frame_delta = cv2.absdiff(self.previous_frame, blurred)
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.previous_frame = blurred
        
        # Return True if any contour is large enough
        return any(cv2.contourArea(c) > 50 for c in contours)

    def start_recording(self):
        """Start recording video and initialize a new barcode session."""
        self.current_session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.output_dir, 'videos', 
                                f'motion_{self.current_session_timestamp}.h264')
        
        # Configure encoder optimized for high fps, low motion blur
        self.encoder = H264Encoder(
            bitrate=4000000,    # 4 Mbps should be sufficient for 640x640
        )
        self.output = FileOutput(video_path)
        self.picam2.start_encoder(self.encoder, self.output)
        
        # Start new barcode session
        self.barcode_processor.start_new_session()
        
        self.is_recording = True
        self.recording_start_time = time.time()
        print(f"Started recording: {video_path}")

    def stop_recording(self):
        """Stop recording and process any detected barcodes."""
        if self.is_recording:
            self.picam2.stop_encoder()
            
            # Send barcode data if any were detected
            if self.barcode_processor.current_session_barcodes:
                session_data = {
                    "session_start": self.current_session_timestamp,
                    "barcodes": self.barcode_processor.current_session_barcodes
                }
                
                # Queue data for sending
                self.json_sender.send_recording_data(
                    self.current_session_timestamp,
                    session_data
                )
            
            self.barcode_processor.end_session()
            self.is_recording = False
            print("Stopped recording")

    def scan_barcode(self, frame):
        """Scan for barcodes in the provided frame."""
        try:
            barcodes = decode(frame)
            for barcode in barcodes:
                barcode_data = barcode.data.decode("utf-8")
                if self.barcode_processor.process_barcode(barcode_data):
                    print(f"Detected barcode: {barcode_data}")
        except Exception as e:
            print(f"Error processing barcode: {e}")

    def motion_detection_loop(self):
        """Main loop for motion detection using the lores stream."""
        while self.running:
            try:
                frame = self.picam2.capture_array("lores")
                motion_detected = self.detect_motion_yuv(frame)
                
                current_time = time.time()
                with self.lock:
                    if motion_detected:
                        self.last_motion_time = current_time
                        if not self.is_recording:
                            self.start_recording()
                    elif self.is_recording:
                        if current_time - self.last_motion_time > 11:
                            self.stop_recording()
                        elif current_time - self.recording_start_time < 6:
                            self.last_motion_time = current_time
            except Exception as e:
                print(f"Error in motion detection loop: {e}")
            
            time.sleep(0.1)  # Minimal sleep to prevent CPU overload

    def barcode_scanning_loop(self):
        """Main loop for barcode scanning using the main stream."""
        while self.running:
            try:
                if self.is_recording:  # Only scan when recording is active
                    frame = self.picam2.capture_array("main")
                    self.scan_barcode(frame)
                    time.sleep(0.0001)  # Slightly longer sleep for barcode scanning
                else:
                    time.sleep(0.1)  # Longer sleep when not scanning
            except Exception as e:
                print(f"Error in barcode scanning loop: {e}")

    def start(self):
        """Start the recording system and processing threads."""
        self.running = True
        self.picam2.start()
        
        # Start the processing threads
        self.motion_thread = threading.Thread(target=self.motion_detection_loop)
        self.barcode_thread = threading.Thread(target=self.barcode_scanning_loop)
        
        self.motion_thread.start()
        self.barcode_thread.start()
        
        print(f"System started - saving recordings to {self.output_dir}")
        print("Press Ctrl+C to exit")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop all processing and clean up resources."""
        print("\nStopping system...")
        self.running = False
        
        if self.motion_thread.is_alive():
            self.motion_thread.join()
        if self.barcode_thread.is_alive():
            self.barcode_thread.join()
            
        if self.is_recording:
            self.stop_recording()
        
        # Stop the JSON sender
        self.json_sender.stop()
        
        self.picam2.stop()
        print("System stopped")

def main():
    # Replace with your server URL
    SERVER_URL = "http://your-server.com/api/barcodes"
    
    try:
        # Initialize with ROI in center of frame
        recorder = MotionBarcodeRecorder(
            server_url=SERVER_URL,
            roi_x=60, 
            roi_y=40, 
            roi_width=200, 
            roi_height=160
        )
        
        recorder.start()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
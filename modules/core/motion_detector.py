"""Motion detection module for the motion barcode system."""
import cv2
import numpy as np
import logging
from config.settings import ProcessingConfig

class MotionDetector:
    def __init__(self, roi_x=0, roi_y=0, roi_width=None, roi_height=None):
        """Initialize the motion detector."""
        self.logger = logging.getLogger(__name__)
        self.roi = (roi_x, roi_y, roi_width, roi_height)
        self.previous_frame = None
        
    def set_frame_size(self, width, height):
        """Set the frame size and adjust ROI if needed."""
        if self.roi[2] is None:
            self.roi = (self.roi[0], self.roi[1], width, height)
            
    def detect_motion(self, frame, threshold=ProcessingConfig.MOTION_THRESHOLD):
        """Detect motion in the provided frame."""
        try:
            roi = frame[self.roi[1]:self.roi[1]+self.roi[3],
                       self.roi[0]:self.roi[0]+self.roi[2]]
            
            blurred = cv2.GaussianBlur(roi, (21, 21), 0)
            
            if self.previous_frame is None:
                self.previous_frame = blurred
                return False
            
            frame_delta = cv2.absdiff(self.previous_frame, blurred)
            thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            self.previous_frame = blurred
            return any(cv2.contourArea(c) > 50 for c in contours)
            
        except Exception as e:
            self.logger.error(f"Error detecting motion: {e}")
            return False
import time
from collections import defaultdict

class BarcodeProcessor:
    def __init__(self):
        self.last_detection = defaultdict(float)  # Last detection time for each barcode
        self.consecutive_detections = defaultdict(int)  # Count consecutive detections
        self.last_processed_time = defaultdict(float)  # Last processing time for each barcode
        self.current_session_barcodes = []  # List to store barcodes for current session
        self.RESCAN_DELAY = 2.0  # Seconds before allowing reprocessing of same barcode

    def start_new_session(self):
        """Start a new recording session"""
        self.current_session_barcodes = []

    def end_session(self):
        """End current recording session"""
        self.current_session_barcodes = []

    def process_barcode(self, barcode_data, current_time=None):
        """
        Process a barcode detection
        Returns True if barcode should be processed (met consecutive detection criteria)
        """
        if current_time is None:
            current_time = time.time()
        
        # Check if this is a consecutive detection of the same barcode
        if current_time - self.last_detection[barcode_data] < 0.5:  # 0.5s window for consecutive detection
            self.consecutive_detections[barcode_data] += 1
        else:
            self.consecutive_detections[barcode_data] = 1
        
        self.last_detection[barcode_data] = current_time
        
        # If we have two consecutive detections and enough time has passed since last processing
        if (self.consecutive_detections[barcode_data] >= 2 and 
            current_time - self.last_processed_time[barcode_data] >= self.RESCAN_DELAY):
            
            # Add to current session
            barcode_entry = {
                "barcode": barcode_data,
                "detection_time": current_time,
            }
            self.current_session_barcodes.append(barcode_entry)
            
            self.last_processed_time[barcode_data] = current_time
            self.consecutive_detections[barcode_data] = 0
            return True
        
        return False
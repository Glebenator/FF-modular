# motion_barcode_system/core/barcode_processor.py
"""Barcode processing module for the motion barcode system."""
import time
from collections import defaultdict
import logging

class BarcodeProcessor:
    def __init__(self):
        """Initialize the barcode processor."""
        self.logger = logging.getLogger(__name__)
        self.last_detection = defaultdict(float)
        self.consecutive_detections = defaultdict(int)
        self.last_processed_time = defaultdict(float)
        self.current_session_barcodes = []
        self.RESCAN_DELAY = 2.0

    def start_new_session(self):
        """Start a new recording session."""
        self.logger.debug("Starting new barcode session")
        self.current_session_barcodes = []

    def end_session(self):
        """End current recording session."""
        self.logger.debug(f"Ending session with {len(self.current_session_barcodes)} barcodes")
        self.current_session_barcodes = []

    def process_barcode(self, barcode_data, current_time=None):
        """
        Process a barcode detection.
        
        Args:
            barcode_data: Decoded barcode data
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            bool: True if barcode should be processed
        """
        if current_time is None:
            current_time = time.time()

        try:
            # Check for consecutive detection
            if current_time - self.last_detection[barcode_data] < 0.5:
                self.consecutive_detections[barcode_data] += 1
            else:
                self.consecutive_detections[barcode_data] = 1

            self.last_detection[barcode_data] = current_time

            # Process if we have multiple detections and enough time has passed
            if (self.consecutive_detections[barcode_data] >= 2 and 
                current_time - self.last_processed_time[barcode_data] >= self.RESCAN_DELAY):
                
                barcode_entry = {
                    "barcode": barcode_data,
                    "direction": "in"
                }
                self.current_session_barcodes.append(barcode_entry)
                
                self.last_processed_time[barcode_data] = current_time
                self.consecutive_detections[barcode_data] = 0
                
                self.logger.info(f"Successfully processed barcode: {barcode_data}")
                return True

        except Exception as e:
            self.logger.error(f"Error processing barcode {barcode_data}: {e}")
            
        return False
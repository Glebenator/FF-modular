# hardware/hardware_controller.py
from gpiozero import RGBLED, DigitalOutputDevice
import threading
import time
import logging
from enum import Enum

from config.settings import HardwareConfig

class LEDStatus(Enum):
    """LED status indicators."""
    RUNNING = "running"          # Green - normal operation
    ERROR = "error"             # Red - system error
    WARNING = "warning"         # Yellow - system warning
    PROCESSING = "processing"    # Blue - video processing active
    RECORDING = "recording"      # Purple - actively recording
    OFF = "off"                # All off

class HardwareController:
    """Controls RGB LED and buzzer hardware components."""
    
    def __init__(self):
        """Initialize hardware controller."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize RGB LED
        try:
            self.led = RGBLED(red=HardwareConfig.RED_PIN, green=HardwareConfig.GREEN_PIN, blue=HardwareConfig.BLUE_PIN)
            self.logger.info("RGB LED initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RGB LED: {e}")
            raise
            
        # Initialize Active Buzzer
        try:
            self.buzzer = DigitalOutputDevice(HardwareConfig.BUZZER_PIN, active_high=True, initial_value=False)
            self.logger.info("Active buzzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize buzzer: {e}")
            raise
        
        self._current_status = LEDStatus.OFF
        self._led_lock = threading.Lock()
        self._buzzer_lock = threading.Lock()
        self._buzzer_timer = None
        
    def set_status(self, status: LEDStatus):
        """Set LED status indicator with solid colors."""
        try:
            with self._led_lock:
                if status == LEDStatus.RUNNING:
                    self.led.color = (0, 1, 0)      # Green
                elif status == LEDStatus.ERROR:
                    self.led.color = (1, 0, 0)      # Red
                elif status == LEDStatus.WARNING:
                    self.led.color = (1, 1, 0)      # Yellow
                elif status == LEDStatus.PROCESSING:
                    self.led.color = (0, 0, 1)      # Blue
                elif status == LEDStatus.RECORDING:
                    self.led.color = (1, 0, 1)      # Purple
                elif status == LEDStatus.OFF:
                    self.led.off()
                    
                self._current_status = status
                self.logger.debug(f"LED status set to: {status.value}")
                
        except Exception as e:
            self.logger.error(f"Error setting LED status: {e}")
            
    def get_status(self) -> LEDStatus:
        """Get current LED status."""
        return self._current_status
        
    def _stop_buzzer_timer(self):
        """Cancel any existing buzzer timer."""
        if self._buzzer_timer is not None:
            self._buzzer_timer.cancel()
            self._buzzer_timer = None
    
    def _delayed_buzzer_stop(self):
        """Stop the buzzer and clear the timer."""
        try:
            self.buzzer.off()
            self._buzzer_timer = None
        except Exception as e:
            self.logger.error(f"Error stopping buzzer: {e}")
        
    def play_barcode_sound(self, duration=0.05):
        """
        Play a short beep sound when barcode is scanned.
        
        Args:
            duration (float): Duration of the beep in seconds. Default is 50ms.
        """
        try:
            with self._buzzer_lock:
                # Cancel any existing timer
                self._stop_buzzer_timer()
                
                # Turn buzzer on
                self.buzzer.on()
                
                # Set up new timer to turn it off
                self._buzzer_timer = threading.Timer(duration, self._delayed_buzzer_stop)
                self._buzzer_timer.start()
                
            self.logger.debug("Played barcode scan sound")
            
        except Exception as e:
            self.logger.error(f"Error playing barcode sound: {e}")
            self.buzzer.off()  # Ensure buzzer is off in case of error
            
    def cleanup(self):
        """Clean up hardware resources."""
        try:
            self.set_status(LEDStatus.OFF)
            
            # Clean up buzzer
            self._stop_buzzer_timer()
            self.buzzer.off()
            self.buzzer.close()
            
            # Clean up LED
            self.led.close()
            
            self.logger.info("Hardware resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up hardware resources: {e}")
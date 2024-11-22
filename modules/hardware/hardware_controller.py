# hardware/hardware_controller.py
from gpiozero import RGBLED, TonalBuzzer
from gpiozero.tones import Tone
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
            
        # Initialize Buzzer
        try:
            self.buzzer = TonalBuzzer(HardwareConfig.BUZZER_PIN)
            self.logger.info("Buzzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize buzzer: {e}")
            raise
        
        self._current_status = LEDStatus.OFF
        self._led_lock = threading.Lock()
        self._buzzer_lock = threading.Lock()
        
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
        
    def play_barcode_sound(self):
        """Play a short beep sound when barcode is scanned."""
        try:
            with self._buzzer_lock:
                # Play A4 note (440Hz) for 100ms
                self.buzzer.play(Tone(HardwareConfig.BUZZER_TONE))
                time.sleep(0.5)
                self.buzzer.stop()
                
            self.logger.debug("Played barcode scan sound")
            
        except Exception as e:
            self.logger.error(f"Error playing barcode sound: {e}")
            
    def cleanup(self):
        """Clean up hardware resources."""
        try:
            self.set_status(LEDStatus.OFF)
            self.buzzer.stop()
            self.led.close()
            self.buzzer.close()
            self.logger.info("Hardware resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up hardware resources: {e}")
# hardware/hardware_controller.py
from gpiozero import RGBLED, TonalBuzzer
from gpiozero.tones import Tone
import threading
import time
import logging
from enum import Enum
import math
from config.settings import HardwareConfig

class LEDStatus(Enum):
    """LED status indicators."""
    RUNNING = "running"          # Solid green
    ERROR = "error"             # Solid red
    WARNING = "warning"         # Solid yellow
    PROCESSING = "processing"   # Pulsing blue
    RECORDING = "recording"     # Blinking green
    OFF = "off"                # All off

class HardwareController:
    """Controls RGB LED and buzzer hardware components."""
    
    def __init__(self):
        """Initialize hardware controller."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize RGB LED
        try:
            self.led = RGBLED(
                red=HardwareConfig.RED_PIN,
                green=HardwareConfig.GREEN_PIN,
                blue=HardwareConfig.BLUE_PIN
            )
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
        self._effect_thread = None
        self._stop_effect = threading.Event()
        
    def _pulse_effect(self, color, period=2.0):
        """Create a pulsing effect with the specified color."""
        while not self._stop_effect.is_set():
            for i in range(100):
                if self._stop_effect.is_set():
                    break
                # Use sine wave for smooth pulsing
                brightness = (math.sin(i / 100.0 * 2 * math.pi) + 1) / 2
                with self._led_lock:
                    self.led.color = tuple(c * brightness for c in color)
                time.sleep(period / 100)

    def _blink_effect(self, color, on_time=0.5, off_time=0.5):
        """Create a blinking effect with the specified color."""
        while not self._stop_effect.is_set():
            with self._led_lock:
                self.led.color = color
                time.sleep(on_time)
                if not self._stop_effect.is_set():
                    self.led.off()
                    time.sleep(off_time)

    def _stop_current_effect(self):
        """Stop any running LED effect."""
        if self._effect_thread and self._effect_thread.is_alive():
            self._stop_effect.set()
            self._effect_thread.join()
        self._stop_effect.clear()

    def set_status(self, status: LEDStatus):
        """Set LED status indicator."""
        try:
            self._stop_current_effect()
            
            with self._led_lock:
                if status == LEDStatus.RUNNING:
                    self.led.color = (0, 1, 0)  # Solid green
                elif status == LEDStatus.ERROR:
                    self.led.color = (1, 0, 0)  # Solid red
                elif status == LEDStatus.WARNING:
                    self.led.color = (1, 1, 0)  # Solid yellow
                elif status == LEDStatus.PROCESSING:
                    # Start pulsing blue effect in a separate thread
                    self._effect_thread = threading.Thread(
                        target=self._pulse_effect,
                        args=((0, 0, 1),)  # Blue color
                    )
                    self._effect_thread.daemon = True
                    self._effect_thread.start()
                elif status == LEDStatus.RECORDING:
                    # Start blinking green effect in a separate thread
                    self._effect_thread = threading.Thread(
                        target=self._blink_effect,
                        args=((0, 1, 0),)  # Green color
                    )
                    self._effect_thread.daemon = True
                    self._effect_thread.start()
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
                self.buzzer.play(Tone(HardwareConfig.BUZZER_TONE))
                time.sleep(0.1)
                self.buzzer.stop()
                
            self.logger.debug("Played barcode scan sound")
            
        except Exception as e:
            self.logger.error(f"Error playing barcode sound: {e}")
            
    def cleanup(self):
        """Clean up hardware resources."""
        try:
            self._stop_current_effect()
            self.set_status(LEDStatus.OFF)
            self.buzzer.stop()
            self.led.close()
            self.buzzer.close()
            self.logger.info("Hardware resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up hardware resources: {e}")
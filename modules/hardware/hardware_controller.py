# hardware/hardware_controller.py
from gpiozero import RGBLED, DigitalOutputDevice, Button
import threading
import time
import logging
import subprocess
from enum import Enum
from config.settings import HardwareConfig

class LEDStatus(Enum):
    """LED status indicators."""
    RUNNING = "running"          # Green - normal operation
    ERROR = "error"             # Red - system error
    WARNING = "warning"         # Yellow - system warning
    PROCESSING = "processing"    # Blue - video processing active
    RECORDING = "recording"      # Purple - actively recording
    SHUTDOWN = "shutdown"       # Fast blinking Red - shutdown in progress
    OFF = "off"                # All off

class HardwareController:
    """Controls RGB LED, buzzer and shutdown button hardware components."""
    
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
            
        # Initialize Active Buzzer
        try:
            self.buzzer = DigitalOutputDevice(
                HardwareConfig.BUZZER_PIN,
                active_high=True,
                initial_value=False
            )
            self.logger.info("Active buzzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize buzzer: {e}")
            raise
            
        # Initialize shutdown button
        try:
            self.shutdown_button = Button(
                HardwareConfig.SHUTDOWN_BUTTON_PIN,
                hold_time=HardwareConfig.SHUTDOWN_HOLD_TIME
            )
            self.shutdown_button.when_held = self._handle_shutdown_request
            self.logger.info("Shutdown button initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize shutdown button: {e}")
            raise
        
        self._current_status = LEDStatus.OFF
        self._led_lock = threading.Lock()
        self._buzzer_lock = threading.Lock()
        self._buzzer_timer = None
        self._shutdown_callback = None
        self._shutdown_in_progress = False
        
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
        
    def register_shutdown_callback(self, callback):
        """Register callback function to be called before shutdown."""
        self._shutdown_callback = callback
        
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

    def _handle_shutdown_request(self):
        """Handle shutdown button press."""
        try:
            if self._shutdown_in_progress:
                return
                
            self._shutdown_in_progress = True
            self.logger.info("Shutdown button held - initiating shutdown sequence")
            
            # Start warning blink in separate thread
            threading.Thread(target=self._blink_shutdown_warning).start()
            
            # Play shutdown warning sound
            self._play_shutdown_warning()
            
            # Execute shutdown sequence
            threading.Thread(target=self._execute_shutdown).start()
            
        except Exception as e:
            self.logger.error(f"Error handling shutdown request: {e}")
            self._shutdown_in_progress = False
            if self._current_status:
                self.set_status(self._current_status)

    def _blink_shutdown_warning(self):
        """Blink LED to indicate shutdown is in progress."""
        try:
            start_time = time.time()
            while time.time() - start_time < HardwareConfig.SHUTDOWN_WARNING_TIME:
                with self._led_lock:
                    self.led.color = (1, 0, 0)  # Red
                time.sleep(0.2)
                with self._led_lock:
                    self.led.off()
                time.sleep(0.2)
            
            # Set solid red for actual shutdown
            with self._led_lock:
                self.led.color = (1, 0, 0)
                
        except Exception as e:
            self.logger.error(f"Error in shutdown warning blink: {e}")

    def _play_shutdown_warning(self):
        """Play shutdown warning sound."""
        try:
            with self._buzzer_lock:
                self.buzzer.on()
                time.sleep(0.5)
                self.buzzer.off()
                
        except Exception as e:
            self.logger.error(f"Error playing shutdown warning: {e}")

    def _execute_shutdown(self):
        """Execute the shutdown sequence."""
        try:
            # Wait for warning period
            time.sleep(HardwareConfig.SHUTDOWN_WARNING_TIME)
            
            # Call registered shutdown callback
            if self._shutdown_callback:
                try:
                    self._shutdown_callback()
                except Exception as e:
                    self.logger.error(f"Error in shutdown callback: {e}")
            
            # Final system shutdown
            self.logger.info("Executing system shutdown")
            subprocess.run(['sudo', 'shutdown', '-h', 'now'])
            
        except Exception as e:
            self.logger.error(f"Error during shutdown sequence: {e}")
            self._shutdown_in_progress = False
            if self._current_status:
                self.set_status(self._current_status)
            
    def cleanup(self):
        """Clean up hardware resources."""
        try:
            self.set_status(LEDStatus.OFF)
            
            # Clean up buzzer
            if hasattr(self, 'buzzer'):
                if self._buzzer_timer:
                    self._buzzer_timer.cancel()
                self.buzzer.off()
                self.buzzer.close()
            
            # Clean up shutdown button
            if hasattr(self, 'shutdown_button'):
                self.shutdown_button.close()
            
            # Clean up LED
            if hasattr(self, 'led'):
                self.led.close()
            
            self.logger.info("Hardware resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up hardware resources: {e}")
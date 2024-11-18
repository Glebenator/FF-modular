# hardware/test_led.py
import RPi.GPIO as GPIO
import threading
import time
from enum import Enum

class LEDStatus(Enum):
    """LED status indicators."""
    RUNNING = "running"      # Green
    ERROR = "error"         # Red
    WARNING = "warning"     # Yellow
    OFF = "off"            # All off

class HardwareTest:
    """Simple test class for RGB LED."""
    
    def __init__(self, red_pin=17, green_pin=27, blue_pin=22):
        """Initialize with default GPIO pins."""
        # Set up GPIO pins
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup LED pins as outputs
        GPIO.setup(self.red_pin, GPIO.OUT)
        GPIO.setup(self.green_pin, GPIO.OUT)
        GPIO.setup(self.blue_pin, GPIO.OUT)
        
        # Create PWM objects for LED pins (100 Hz frequency)
        self.red_pwm = GPIO.PWM(self.red_pin, 100)
        self.green_pwm = GPIO.PWM(self.green_pin, 100)
        self.blue_pwm = GPIO.PWM(self.blue_pin, 100)
        
        # Start PWM with 0% duty cycle
        self.red_pwm.start(0)
        self.green_pwm.start(0)
        self.blue_pwm.start(0)
        
        self._current_status = LEDStatus.OFF
        self._led_lock = threading.Lock()
            
    def _set_color(self, red, green, blue):
        """Set RGB LED color using PWM values (0-100)."""
        with self._led_lock:
            self.red_pwm.ChangeDutyCycle(red)
            self.green_pwm.ChangeDutyCycle(green)
            self.blue_pwm.ChangeDutyCycle(blue)
            
    def set_status(self, status: LEDStatus):
        """Set LED status indicator."""
        try:
            if status == LEDStatus.RUNNING:
                self._set_color(0, 100, 0)  # Green
            elif status == LEDStatus.ERROR:
                self._set_color(100, 0, 0)  # Red
            elif status == LEDStatus.WARNING:
                self._set_color(100, 100, 0)  # Yellow
            elif status == LEDStatus.OFF:
                self._set_color(0, 0, 0)  # Off
                
            self._current_status = status
            print(f"LED status set to: {status.value}")
            
        except Exception as e:
            print(f"Error setting LED status: {e}")
            
    def cleanup(self):
        """Clean up GPIO resources."""
        try:
            self.set_status(LEDStatus.OFF)
            self.red_pwm.stop()
            self.green_pwm.stop()
            self.blue_pwm.stop()
            GPIO.cleanup()
            print("Hardware resources cleaned up")
            
        except Exception as e:
            print(f"Error cleaning up hardware resources: {e}")

def test_led_states():
    """Test basic LED states."""
    print("Testing LED states...")
    hw = HardwareTest()
    
    try:
        # Test each status
        print("Setting LED to RUNNING (Green)")
        hw.set_status(LEDStatus.RUNNING)
        time.sleep(2)
        
        print("Setting LED to WARNING (Yellow)")
        hw.set_status(LEDStatus.WARNING)
        time.sleep(2)
        
        print("Setting LED to ERROR (Red)")
        hw.set_status(LEDStatus.ERROR)
        time.sleep(2)
        
        print("Setting LED to OFF")
        hw.set_status(LEDStatus.OFF)
        time.sleep(1)
        
        # Test color cycling
        print("\nTesting color cycle (press Ctrl+C to stop)...")
        while True:
            # Fade through some colors
            for r, g, b in [
                (100, 0, 0),    # Red
                (0, 100, 0),    # Green
                (0, 0, 100),    # Blue
                (100, 100, 0),  # Yellow
                (100, 0, 100),  # Purple
                (0, 100, 100),  # Cyan
            ]:
                hw._set_color(r, g, b)
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Cleaning up...")
        hw.cleanup()
        print("Test complete")

def test_led_fade():
    """Test LED fading effect."""
    print("Testing LED fade effect (press Ctrl+C to stop)...")
    hw = HardwareTest()
    
    try:
        while True:
            # Fade in green
            print("Fading in green...")
            for i in range(101):
                hw._set_color(0, i, 0)
                time.sleep(0.02)
            
            # Fade out green
            print("Fading out green...")
            for i in range(100, -1, -1):
                hw._set_color(0, i, 0)
                time.sleep(0.02)
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Cleaning up...")
        hw.cleanup()
        print("Test complete")

if __name__ == "__main__":
    print("RGB LED Test Script")
    print("-----------------")
    print("Connection Guide:")
    print("- Red LED   -> GPIO 17 (with 220Ω resistor)")
    print("- Green LED -> GPIO 27 (with 220Ω resistor)")
    print("- Blue LED  -> GPIO 22 (with 220Ω resistor)")
    print("- Common Cathode -> GND")
    print("-----------------")
    print("1. Test LED states (cycles through colors)")
    print("2. Test LED fade effect (green fade in/out)")
    
    choice = input("Choose a test (1 or 2): ")
    
    if choice == "1":
        test_led_states()
    elif choice == "2":
        test_led_fade()
    else:
        print("Invalid choice")
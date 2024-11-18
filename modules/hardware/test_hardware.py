# hardware/test_hardware.py
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware_controller import HardwareController, LEDStatus

def test_hardware_controller():
    """
    Test function to verify all hardware controller functionality.
    Runs through various LED states and effects.
    """
    try:
        print("Initializing hardware controller test...")
        controller = HardwareController()
        
        # Test basic states
        print("\nTesting basic LED states:")
        states = [
            (LEDStatus.RUNNING, "Running (solid green)", 2),
            (LEDStatus.WARNING, "Warning (solid yellow)", 2),
            (LEDStatus.ERROR, "Error (solid red)", 2),
            (LEDStatus.OFF, "Off", 2)
        ]
        
        for state, description, duration in states:
            print(f"\nSetting LED to {description}")
            controller.set_status(state)
            time.sleep(duration)
            
        # Test processing pulse
        print("\nTesting processing state (pulsing blue) for 5 seconds...")
        controller.set_status(LEDStatus.PROCESSING)
        time.sleep(5)
        
        # Test barcode detection pulses with different colors
        print("\nTesting barcode detection pulses...")
        colors = [
            ((0, 1, 0), "Green"),
            ((0, 0, 1), "Blue"),
            ((1, 0, 0), "Red"),
            ((1, 1, 0), "Yellow"),
            ((1, 0, 1), "Purple"),
            ((0, 1, 1), "Cyan")
        ]
        
        for color, name in colors:
            print(f"Pulsing {name}...")
            controller.pulse_success(color)
            time.sleep(1.5)  # Wait for pulse to complete
            
        # Test rapid barcode detection
        print("\nTesting rapid barcode detection simulation...")
        for _ in range(5):
            controller.pulse_success((0, 1, 0))  # Green pulses
            time.sleep(0.5)
            
        # Test buzzer
        print("\nTesting buzzer...")
        for _ in range(3):
            controller.play_barcode_sound()
            time.sleep(0.5)
            
        # Test combined effects
        print("\nTesting combined LED and buzzer (simulating barcode detection)...")
        for _ in range(3):
            controller.pulse_success((0, 1, 0))
            controller.play_barcode_sound()
            time.sleep(1)
            
        # Return to normal state
        print("\nReturning to running state (solid green)...")
        controller.set_status(LEDStatus.RUNNING)
        time.sleep(2)
        
        print("\nHardware test complete!")
        
    except Exception as e:
        print(f"Error during hardware test: {e}")
        
    finally:
        print("\nCleaning up...")
        controller.cleanup()

if __name__ == "__main__":
    print("Starting hardware controller test...")
    print("This test will demonstrate all LED states and effects.")
    print("Press Ctrl+C to exit at any time.")
    print("\nNOTE: Ensure all GPIO pins are correctly connected!")
    
    try:
        test_hardware_controller()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        try:
            controller.cleanup()
        except:
            pass
    except Exception as e:
        print(f"\nTest failed: {e}")
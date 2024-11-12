# core/camera_manager.py
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import controls
import logging
from config.settings import CameraConfig

class CameraManager:
    def __init__(self):
        """Initialize the camera system."""
        self.logger = logging.getLogger(__name__)
        self.picam2 = Picamera2()
        self._configure_camera()
        self.encoder = None
        self.output = None
        
    def _configure_camera(self):
        """Configure camera settings."""
        self.config = self.picam2.create_video_configuration(
            main={"size": CameraConfig.MAIN_RESOLUTION, "format": "RGB888"},
            lores={"size": CameraConfig.LORES_RESOLUTION, "format": "YUV420"},
            encode="main",
            buffer_count=8,
            controls={
                "FrameDurationLimits": (8333, 8333),
                "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,
                "AfMode": controls.AfModeEnum.Continuous,
                "AfSpeed": controls.AfSpeedEnum.Fast,
                "ExposureTime": CameraConfig.EXPOSURE_TIME,
                "AnalogueGain": CameraConfig.ANALOG_GAIN,
                "Sharpness": CameraConfig.SHARPNESS
            }
        )
        self.picam2.configure(self.config)
        
    def start_recording(self, output_path):
        """Start recording to the specified output path."""
        try:
            self.encoder = H264Encoder(bitrate=4000000)
            self.output = FileOutput(output_path)
            self.picam2.start_encoder(self.encoder, self.output)
            self.logger.info(f"Started recording to: {output_path}")
            return self.encoder, self.output
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            raise
            
    def stop_recording(self):
        """Stop the current recording."""
        try:
            if self.encoder:
                self.picam2.stop_encoder()
                self.encoder = None
                self.output = None
            self.logger.info("Stopped recording")
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            raise
            
    def capture_frame(self, stream="main"):
        """Capture a frame from the specified stream."""
        return self.picam2.capture_array(stream)
        
    def start(self):
        """Start the camera."""
        self.picam2.start()
        
    def stop(self):
        """Stop the camera."""
        if self.encoder:
            self.stop_recording()
        self.picam2.stop()
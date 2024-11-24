# motion_barcode_system/core/videotrack.py
"""Video tracking module for post-recording analysis."""
from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np
import time
import logging
import json
import os
from config.settings import ProcessingConfig, PathConfig

def draw_dashed_line(img, start_point, end_point, color, thickness=1, dash_length=10, gap_length=10):
    """Draw a dashed line on the image."""
    x1, y1 = start_point
    x2, y2 = end_point
    
    # Calculate line length and angle
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angle = np.arctan2(y2 - y1, x2 - x1)
    
    # Calculate x and y steps for each dash
    dx = dash_length * np.cos(angle)
    dy = dash_length * np.sin(angle)
    
    # Draw dashes
    curr_x, curr_y = x1, y1
    while np.sqrt((curr_x - x1) ** 2 + (curr_y - y1) ** 2) < length:
        # Calculate end point of current dash
        end_x = min(curr_x + dx, x2)
        end_y = min(curr_y + dy, y2)
        
        # Draw the dash
        cv2.line(img, 
                 (int(curr_x), int(curr_y)), 
                 (int(end_x), int(end_y)), 
                 color, 
                 thickness)
        
        # Move to start of next dash
        curr_x = curr_x + dx + gap_length * np.cos(angle)
        curr_y = curr_y + dy + gap_length * np.sin(angle)
        
        # Break if we've gone past the end point
        if curr_x > x2 or curr_y > y2:
            break

class FridgeDirectionDetector:
    """Detects object movement between zones."""
    
    def __init__(self, frame_height, frame_width, 
                 boundary_percent=60,  # Boundary line at 60% from top
                 buffer_percent=10):   # Buffer zone is 10% of frame height
        """
        Initialize the zone detector with configurable zones.
        
        Args:
            frame_height: Height of the video frame in pixels
            frame_width: Width of the video frame in pixels
            boundary_percent: Position of boundary line as percentage from top (0-100)
            buffer_percent: Size of buffer/transition zone as percentage of frame height
        """
        self.logger = logging.getLogger(__name__)
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Convert percentages to pixel values
        self.zone_boundary = int((boundary_percent / 100) * frame_height)
        self.zone_buffer = int((buffer_percent / 100) * frame_height)
        
        # Tracking state
        self.object_states = {}       # Final state: "IN", "OUT", or None
        self.previous_zones = {}      # Last known zone
        self.pending_movements = {}   # Track objects moving through transition
        self.detected_events = []
        
        self._setup_zones()
        
        # Log zone setup for debugging
        self.logger.info(f"Zones configured - Boundary: {self.zone_boundary}px ({boundary_percent}%), "
                        f"Buffer: {self.zone_buffer}px ({buffer_percent}%)")
        self.logger.info(f"Outside zone: 0 to {self.zone_boundary - self.zone_buffer}px")
        self.logger.info(f"Transition zone: {self.zone_boundary - self.zone_buffer} to {self.zone_boundary + self.zone_buffer}px")
        self.logger.info(f"Inside zone: {self.zone_boundary + self.zone_buffer} to {frame_height}px")
        
    def _setup_zones(self):
        """Define the zones in the frame."""
        self.zones = {
            "OUTSIDE": (0, self.zone_boundary - self.zone_buffer),
            "TRANSITION": (self.zone_boundary - self.zone_buffer, 
                         self.zone_boundary + self.zone_buffer),
            "INSIDE": (self.zone_boundary + self.zone_buffer, self.frame_height)
        }
        
    def get_zone(self, y_pos):
        """Determine which zone an object is in based on its y position."""
        for zone_name, (start, end) in self.zones.items():
            if start <= y_pos < end:
                return zone_name
        return "OUTSIDE"
        
    def detect_direction(self, track_ids, boxes, class_ids, names):
        """Detect direction of movement based on zone transitions."""
        current_tracks = set()
        transitions = {}
        
        # Process each detected object
        for track_id, box, class_id in zip(track_ids, boxes, class_ids):
            x, y = float(box[0]), float(box[1])
            current_tracks.add(track_id)
            object_type = names[int(class_id)]
            current_zone = self.get_zone(y)
            
            # Initialize new object
            if track_id not in self.object_states:
                self.object_states[track_id] = None
                self.previous_zones[track_id] = current_zone
                continue
            
            prev_zone = self.previous_zones[track_id]
            
            # Handle zone changes
            if prev_zone != current_zone:
                # Start tracking potential IN movement
                if prev_zone == "OUTSIDE" and current_zone == "TRANSITION":
                    self.pending_movements[track_id] = "IN"
                
                # Complete IN movement
                elif current_zone == "INSIDE" and (
                    track_id in self.pending_movements or prev_zone == "OUTSIDE"
                ):
                    if self.object_states[track_id] != "IN":
                        self._record_transition(track_id, "IN", object_type, time.time())
                        transitions[track_id] = ("IN", object_type)
                        self.object_states[track_id] = "IN"
                    if track_id in self.pending_movements:
                        del self.pending_movements[track_id]
                
                # Start tracking potential OUT movement
                elif prev_zone == "INSIDE" and current_zone == "TRANSITION":
                    self.pending_movements[track_id] = "OUT"
                
                # Complete OUT movement
                elif current_zone == "OUTSIDE" and (
                    track_id in self.pending_movements or prev_zone == "INSIDE"
                ):
                    if self.object_states[track_id] != "OUT":
                        self._record_transition(track_id, "OUT", object_type, time.time())
                        transitions[track_id] = ("OUT", object_type)
                        self.object_states[track_id] = "OUT"
                    if track_id in self.pending_movements:
                        del self.pending_movements[track_id]
            
            # Update previous zone
            self.previous_zones[track_id] = current_zone
        
        # Clean up state for objects no longer tracked
        self._cleanup_state(current_tracks)
        
        return transitions
        
    def _cleanup_state(self, current_tracks):
        """Remove state data for objects no longer being tracked."""
        to_remove = set(self.object_states.keys()) - current_tracks
        for track_id in to_remove:
            self.object_states.pop(track_id)
            self.previous_zones.pop(track_id)
            if track_id in self.pending_movements:
                del self.pending_movements[track_id]
            
    def _record_transition(self, track_id, direction, object_type, current_time):
        """Record a zone transition."""
        event = {
            "timestamp": current_time,
            "track_id": track_id,
            "object_type": object_type,
            "direction": direction,
        }
        self.detected_events.append(event)
        self.logger.info(f"{object_type} {track_id} moved {direction}")
        
    def get_events(self):
        """Return all detected events."""
        return self.detected_events

def draw_zones(img, zone_detector):
    """Draw zone boundaries and labels on the frame."""
    # Draw main boundary line
    cv2.line(img, (0, zone_detector.zone_boundary), 
             (zone_detector.frame_width, zone_detector.zone_boundary),
             (0, 255, 0), 2)
    
    # Draw buffer zone lines using our custom dashed line function
    draw_dashed_line(img, 
                     (0, zone_detector.zone_boundary - zone_detector.zone_buffer),
                     (zone_detector.frame_width, zone_detector.zone_boundary - zone_detector.zone_buffer),
                     (0, 255, 0), 1)
    draw_dashed_line(img,
                     (0, zone_detector.zone_boundary + zone_detector.zone_buffer),
                     (zone_detector.frame_width, zone_detector.zone_boundary + zone_detector.zone_buffer),
                     (0, 255, 0), 1)
    
    # Add zone labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "OUTSIDE", (10, zone_detector.zone_boundary - zone_detector.zone_buffer - 10),
                font, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "TRANSITION", (10, zone_detector.zone_boundary),
                font, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "INSIDE", (10, zone_detector.zone_boundary + zone_detector.zone_buffer + 25),
                font, 0.6, (0, 255, 0), 2)

def process_h264_video(video_path, model_path, boundary_percent=60, buffer_percent=10):
    """
    Process a recorded video file for object tracking and zone detection.
    
    Args:
        video_path: Path to the input video file
        model_path: Path to the YOLO model
        boundary_percent: Position of boundary line as percentage from top (0-100)
        buffer_percent: Size of buffer/transition zone as percentage of frame height
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting video processing: {video_path}")
    logger.info(f"Zone configuration - Boundary: {boundary_percent}%, Buffer: {buffer_percent}%")
    
    try:
        # Load the YOLO model
        model = YOLO(model_path, task=ProcessingConfig.YOLO_TASK)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Set up output video
        output_path = os.path.join(
            PathConfig.VIDEO_ANALYZED_DIR,
            os.path.basename(video_path).replace('.h264', '_analyzed.mp4')
        )
        os.makedirs(PathConfig.VIDEO_ANALYZED_DIR, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize tracking with configured zones
        track_history = defaultdict(lambda: [])
        zone_detector = FridgeDirectionDetector(
            frame_height=frame_height,
            frame_width=frame_width,
            boundary_percent=boundary_percent,
            buffer_percent=buffer_percent
        )
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO tracking
            results = model.track(
                source=frame,
                conf=ProcessingConfig.VIDEO_PROCESSING_CONFIDENCE,
                iou=ProcessingConfig.VIDEO_PROCESSING_IOU,
                persist=True,
                verbose=False,
                stream=True,
                vid_stride=ProcessingConfig.VIDEO_PROCESSING_STRIDE,
                max_det=ProcessingConfig.MAX_DETECTIONS,
                tracker="botsort.yaml",
                classes = [45, 46, 47, 48, 49, 50, 51]
            )
            
            # Process tracking results
            for result in results:
                annotated_frame = process_tracking_results(
                    result, zone_detector, track_history,
                    frame, frame_width
                )
                
                # Write the frame
                out.write(annotated_frame)
                
        # Save detection results
        os.makedirs(PathConfig.JSON_TRACKING_DIR, exist_ok=True)
        save_detection_results(video_path, zone_detector.get_events())
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Completed video processing: {video_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return False

def process_tracking_results(result, zone_detector, track_history, frame, frame_width):
    """Process and visualize tracking results for a single frame."""
    if result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        
        # Detect zone transitions
        transitions = zone_detector.detect_direction(track_ids, boxes, class_ids, result.names)
        
        # Get the annotated frame with YOLO detections
        annotated_frame = result.plot()
        
        # Draw tracking visualization
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
                
            # Draw tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, 
                         color=(230, 230, 230), thickness=10)
            
            # Add direction label if available
            if track_id in transitions:
                direction, object_type = transitions[track_id]
                label = f"{object_type} {track_id}: {direction}"
                cv2.putText(annotated_frame, label,
                          (int(x - w/2), int(y - h/2 - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        annotated_frame = frame
        
    # Draw zone visualization
    draw_zones(annotated_frame, zone_detector)
    
    return annotated_frame

def save_detection_results(video_path, events):
    """Save detection results to a JSON file."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        results_path = os.path.join(
            PathConfig.JSON_DIR,
            f"{base_name}_tracking_results.json"
        )
        
        results = {
            "video_file": video_path,
            "processing_time": time.time(),
            "events": events
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving detection results: {e}")
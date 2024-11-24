# motion_barcode_system/core/videotrack.py
import cv2
from ultralytics import YOLO
import numpy as np
import time
import logging
import json
import os
from config.settings import ProcessingConfig, PathConfig

class FridgeDirectionDetector:
    """Detects object movement direction relative to a boundary line using hysteresis."""
    
    def __init__(self, line_points):
        """Initialize the direction detector."""
        self.logger = logging.getLogger(__name__)
        self.line_points = line_points
        self.last_crossing = {}
        self.crossing_cooldown = 2.0
        self.detected_events = []
        
        # Hysteresis thresholds relative to the main line
        self.UPPER_THRESHOLD = -45  # Upper trigger line
        self.LOWER_THRESHOLD = 45   # Lower trigger line
        
        # Track the last known position relative to thresholds
        self.last_positions = {}
        
    def detect_direction(self, track_ids, boxes, class_ids, names):
        """Detect direction of movement for tracked objects using hysteresis thresholds."""
        current_time = time.time()
        current_directions = {}
        main_line_y = self.line_points[0][1]
        
        # Process each detected object
        for track_id, box, class_id in zip(track_ids, boxes, class_ids):
            # Get current position relative to main line
            _, y = float(box[0]), float(box[1])
            relative_pos = y - main_line_y
            object_name = names[int(class_id)]
            
            # Skip if in cooldown period
            if track_id in self.last_crossing:
                if current_time - self.last_crossing[track_id] < self.crossing_cooldown:
                    continue
            
            # Check for threshold crossings regardless of initial position
            if track_id in self.last_positions:
                last_pos = self.last_positions[track_id]
                
                # Detect crossing lower threshold moving down (IN)
                if last_pos <= self.LOWER_THRESHOLD and relative_pos > self.LOWER_THRESHOLD:
                    self._record_movement(track_id, "IN", object_name, current_time)
                    current_directions[track_id] = ("IN", object_name)
                    
                # Detect crossing upper threshold moving up (OUT)
                elif last_pos >= self.UPPER_THRESHOLD and relative_pos < self.UPPER_THRESHOLD:
                    self._record_movement(track_id, "OUT", object_name, current_time)
                    current_directions[track_id] = ("OUT", object_name)
            
            # Update last known position
            self.last_positions[track_id] = relative_pos
        
        return current_directions
        
    def _record_movement(self, track_id, direction, object_name, current_time):
        """Record a detected movement event."""
        self.last_crossing[track_id] = current_time
        
        event = {
            "timestamp": current_time,
            "track_id": track_id,
            "object_type": object_name,
            "direction": direction,
        }
        self.detected_events.append(event)
        self.logger.info(f"{object_name} {track_id} moved {direction}")
    
    def get_events(self):
        """Return all detected events."""
        return self.detected_events
    
    def cleanup_inactive(self, current_time, active_track_ids):
        """Clean up state for inactive tracks."""
        all_track_ids = set(self.last_positions.keys())
        inactive_track_ids = all_track_ids - set(active_track_ids)
        
        for track_id in inactive_track_ids:
            del self.last_positions[track_id]
            if track_id in self.last_crossing:
                del self.last_crossing[track_id]

def draw_detection_lines(frame, line_points, frame_width):
    """Draw the detection lines and hysteresis thresholds on the frame."""
    main_y = line_points[0][1]
    
    # Draw main line (solid green)
    cv2.line(frame, (0, main_y), (frame_width, main_y), (0, 255, 0), 2)
    
    # Draw upper threshold line (dashed)
    upper_y = main_y - 45
    draw_dashed_line(frame, (0, upper_y), (frame_width, upper_y), (0, 255, 0), 1)
    
    # Draw lower threshold line (dashed)
    lower_y = main_y + 45
    draw_dashed_line(frame, (0, lower_y), (frame_width, lower_y), (0, 255, 0), 1)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed line on the image."""
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    dashes = int(dist / (2 * dash_length))
    for i in range(dashes):
        start = np.array(pt1) + (i * 2 * dash_length / dist) * (np.array(pt2) - np.array(pt1))
        end = np.array(pt1) + ((i * 2 + 1) * dash_length / dist) * (np.array(pt2) - np.array(pt1))
        start = tuple(map(int, start))
        end = tuple(map(int, end))
        cv2.line(img, start, end, color, thickness)

def process_tracking_results(result, detector, track_history, frame, frame_width, line_points):
    """Process and visualize tracking results for a single frame."""
    # Get the annotated frame with YOLO detections
    annotated_frame = result.plot()
    
    if result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        
        # Detect and update directions
        current_directions = detector.detect_direction(track_ids, boxes, class_ids, result.names)
        
        # Clean up inactive tracks
        detector.cleanup_inactive(time.time(), track_ids)
        
        # Visualize tracks and labels
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            # Update track history
            if track_id not in track_history:
                track_history[track_id] = []
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            
            # Draw tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, 
                         color=(230, 230, 230), thickness=10)
            
            # Add direction label if available
            if track_id in current_directions:
                direction, object_name = current_directions[track_id]
                label = f"{object_name} {track_id}: {direction}"
                cv2.putText(annotated_frame, label,
                          (int(x - w/2), int(y - h/2 - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw detection lines with hysteresis thresholds
    draw_detection_lines(annotated_frame, line_points, frame_width)
    
    return annotated_frame

def process_h264_video(video_path, model_path, pause_event=None):
    """Process a recorded video file for object tracking and direction detection."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting video processing: {video_path}")
    
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
        
        # Define counting line points
        line_points = [(0, frame_height // 2), (frame_width, frame_height // 2)]
        
        # Initialize tracking
        track_history = {}
        detector = FridgeDirectionDetector(line_points)
        
        while cap.isOpened():
            # Check for pause if pause_event is provided
            if pause_event and pause_event.is_set():
                time.sleep(0.1)
                continue
                
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
                classes=[45, 46, 47, 48, 49, 50, 51]  # Food item classes
            )
            
            # Process tracking results
            for result in results:
                annotated_frame = process_tracking_results(
                    result, detector, track_history,
                    frame, frame_width, line_points
                )
                
                # Write the frame
                out.write(annotated_frame)
        
        # Save detection results
        os.makedirs(PathConfig.JSON_TRACKING_DIR, exist_ok=True)
        save_detection_results(video_path, detector.get_events())
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Completed video processing: {video_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return False

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
# motion_barcode_system/core/videotrack.py
"""Video processing module with improved tracking."""
import cv2
from ultralytics import YOLO
import numpy as np
import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from config.settings import PathConfig, ProcessingConfig

class FridgeDirectionDetector:
    """Detects object movement direction using both position and track history."""
    
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
        
        # Track history for each object
        self.track_histories = {}
        self.last_positions = {}
        
    def detect_direction(self, track_ids, boxes, class_ids, names, track_history):
        """Detect direction of movement using both current position and track history."""
        current_time = time.time()
        current_directions = {}
        main_line_y = self.line_points[0][1]
        
        for track_id, box, class_id in zip(track_ids, boxes, class_ids):
            _, y = float(box[0]), float(box[1])
            relative_pos = y - main_line_y
            object_name = names[int(class_id)]
            
            # First determine if we detect any movement
            detected_direction = None
            
            # Check track history
            track = track_history.get(track_id, [])
            if track:
                crossed_in = self._check_track_crossing(track, main_line_y, "IN")
                crossed_out = self._check_track_crossing(track, main_line_y, "OUT")
                if crossed_in:
                    detected_direction = "IN"
                elif crossed_out:
                    detected_direction = "OUT"
            
            # Check immediate position changes if no direction detected yet
            if not detected_direction and track_id in self.last_positions:
                last_pos = self.last_positions[track_id]
                if last_pos <= self.LOWER_THRESHOLD and relative_pos > self.LOWER_THRESHOLD:
                    detected_direction = "IN"
                elif last_pos >= self.UPPER_THRESHOLD and relative_pos < self.UPPER_THRESHOLD:
                    detected_direction = "OUT"
            
            # Now apply cooldown check AFTER determining direction
            if detected_direction:
                # Check if we're in cooldown period
                if track_id in self.last_crossing:
                    time_since_last = current_time - self.last_crossing[track_id]
                    if time_since_last < self.crossing_cooldown:
                        self.logger.debug(f"Skipping {object_name} {track_id}: in cooldown ({time_since_last:.1f}s < {self.crossing_cooldown}s)")
                        continue
                
                # If we pass cooldown check, record the movement
                self._record_movement(track_id, detected_direction, object_name, current_time)
                current_directions[track_id] = (detected_direction, object_name)
            
            # Always update last known position
            self.last_positions[track_id] = relative_pos
        
        return current_directions
    
    def _check_track_crossing(self, track, line_y, direction):
        """Check if a track history shows line crossing in the specified direction."""
        if len(track) < 2:
            return False
            
        # Convert track to numpy array for easier processing
        track_arr = np.array(track)
        
        # Get points around the line
        if direction == "IN":
            # Look for movement from above to below the line
            above_line = track_arr[:, 1] < line_y
            below_line = track_arr[:, 1] > line_y
        else:
            # Look for movement from below to above the line
            above_line = track_arr[:, 1] > line_y
            below_line = track_arr[:, 1] < line_y
        
        # Check if we have points on both sides and they're in the right order
        has_above = np.any(above_line)
        has_below = np.any(below_line)
        
        if has_above and has_below:
            # Find where the crossing happened
            transition_indices = np.where(np.diff(above_line))[0]
            if len(transition_indices) > 0:
                # Check if the transition is in the right direction
                if direction == "IN":
                    return np.any(np.diff(track_arr[transition_indices, 1]) > 0)
                else:
                    return np.any(np.diff(track_arr[transition_indices, 1]) < 0)
        
        return False
    
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
    
    def cleanup_inactive(self, current_time, active_track_ids):
        """Clean up state for inactive tracks."""
        all_track_ids = set(self.last_positions.keys())
        inactive_track_ids = all_track_ids - set(active_track_ids)
        
        for track_id in inactive_track_ids:
            del self.last_positions[track_id]
            if track_id in self.last_crossing:
                del self.last_crossing[track_id]
                
    def get_events(self):
        """Return all detected events."""
        return self.detected_events


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
        current_directions = detector.detect_direction(track_ids, boxes, class_ids, 
                                                    result.names, track_history)
        
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
            if len(track) > 50:  # Keep last 30 points
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


def process_h264_video(video_path, model_path):
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
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % ProcessingConfig.VIDEO_PROCESSING_STRIDE != 0:
                continue
            
            # Run YOLO tracking
            results = model.track(
                source=frame,
                conf=ProcessingConfig.VIDEO_PROCESSING_CONFIDENCE,
                iou=ProcessingConfig.VIDEO_PROCESSING_IOU,
                persist=True,
                verbose=False,
                stream=True,
                vid_stride=1,  # Already handled by frame_count
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
            "processing_time": datetime.now().isoformat(),
            "events": events,
            "processing_parameters": {
                "confidence_threshold": ProcessingConfig.VIDEO_PROCESSING_CONFIDENCE,
                "iou_threshold": ProcessingConfig.VIDEO_PROCESSING_IOU,
                "max_detections": ProcessingConfig.MAX_DETECTIONS,
                "processing_stride": ProcessingConfig.VIDEO_PROCESSING_STRIDE
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving detection results: {e}")
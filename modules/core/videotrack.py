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
        self.detected_events = []
        self.track_histories = {}  # Stores complete trajectories
        self.processed_tracks = set()  # Tracks we've already classified
        
        # Define zones relative to the boundary line
        self.main_line_y = line_points[0][1]
        self.ZONE_MARGIN = 50  # pixels to consider as "decisively" in a zone
        
    def detect_direction(self, track_ids, boxes, class_ids, names, track_history):
        """Detect direction by analyzing complete object trajectories."""
        current_directions = {}
        current_time = time.time()
        
        # Update track histories
        for track_id, box, class_id in zip(track_ids, boxes, class_ids):
            x, y = float(box[0]), float(box[1])
            object_name = names[int(class_id)]
            
            if track_id not in self.track_histories:
                self.track_histories[track_id] = {
                    'positions': [],
                    'object_name': object_name,
                    'processed': False,
                    'last_update': current_time
                }
            
            track_data = self.track_histories[track_id]
            track_data['positions'].append((x, y))
            track_data['last_update'] = current_time
            
            # Only process tracks we haven't classified yet
            if track_id not in self.processed_tracks:
                direction = self._analyze_trajectory(track_id)
                if direction:
                    self._record_movement(track_id, direction, object_name)
                    current_directions[track_id] = (direction, object_name)
                    self.processed_tracks.add(track_id)
        
        return current_directions
    
    def _analyze_trajectory(self, track_id):
        """
        Analyze a complete trajectory to determine movement direction.
        Returns None if direction cannot be determined yet.
        """
        track_data = self.track_histories[track_id]
        positions = track_data['positions']
        
        if len(positions) < 3:  # Reduced minimum points needed (was 5)
            return None
            
        # Instead of just using first/last points, analyze the whole path
        y_positions = [y for _, y in positions]
        max_y = max(y_positions)
        min_y = min(y_positions)
        
        # Check if the path crosses the line with sufficient margin
        crosses_up = min_y < (self.main_line_y - 20) and max_y > (self.main_line_y + 20)
        
        if crosses_up:
            # Determine direction by comparing early vs late positions
            early_y = sum(y_positions[:3]) / 3  # Average of first 3 points
            late_y = sum(y_positions[-3:]) / 3  # Average of last 3 points
            
            if early_y < late_y:
                self.logger.debug(f"Track {track_id} shows clear downward movement "
                                f"(early_y={early_y:.1f}, late_y={late_y:.1f})")
                return "IN"
            elif early_y > late_y:
                self.logger.debug(f"Track {track_id} shows clear upward movement "
                                f"(early_y={early_y:.1f}, late_y={late_y:.1f})")
                return "OUT"
        
        # If track is long but no direction determined, log for debugging
        if len(positions) > 20:
            self.logger.debug(
                f"Track {track_id} has {len(positions)} points but no clear direction. "
                f"Y-range: {min_y:.1f} to {max_y:.1f}, line at {self.main_line_y}"
            )
            
        return None
    
    def _record_movement(self, track_id, direction, object_name):
        """Record a detected movement event."""
        track_data = self.track_histories[track_id]
        
        # Convert direction to lowercase and clean up object name
        direction = direction.lower()  # "IN" -> "in"
        
        # Record simplified event format
        event = {
            "name": object_name,
            "direction": direction
        }
        
        self.detected_events.append(event)
        self.logger.info(f"{object_name} moved {direction}")
    
    def cleanup_inactive(self, current_time, active_track_ids):
        """Clean up state for inactive tracks."""
        inactive_tracks = set(self.track_histories.keys()) - set(active_track_ids)
        
        for track_id in inactive_tracks:
            track_data = self.track_histories[track_id]
            
            # Try one last time to determine direction if not processed
            if track_id not in self.processed_tracks:
                direction = self._analyze_trajectory(track_id)
                if direction:
                    self._record_movement(track_id, direction, track_data['object_name'])
            
            # Clean up
            del self.track_histories[track_id]
                
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
            
            # Update track history for visualization
            if track_id not in track_history:
                track_history[track_id] = []
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 60:  # Keep more points for visualization
                track.pop(0)
            
            # Draw trajectory
            if len(track) > 1:
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # Draw thicker white line for better visibility
                cv2.polylines(annotated_frame, [points], isClosed=False, 
                            color=(230, 230, 230), thickness=10)
            
            # Add direction label if available
            if track_id in current_directions:
                direction, object_name = current_directions[track_id]
                label = f"{object_name} {track_id}: {direction}"
                cv2.putText(annotated_frame, label,
                          (int(x - w/2), int(y - h/2 - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw detection lines
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
                tracker="botsort.yaml"
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


def save_detection_results(video_path, events, session_start):  # Add session_start parameter
    """Save detection results to a JSON file."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        results_path = os.path.join(
            PathConfig.JSON_DIR,
            f"{base_name}_tracking_results.json"
        )
        
        # Create the standardized format using the passed session_start
        results = {
            "session_start": session_start,
            "items": events
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving detection results: {e}")
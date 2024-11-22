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

class FridgeDirectionDetector:
    """Detects object movement direction relative to a boundary line."""
    
    def __init__(self, line_points):
        """Initialize the direction detector."""
        self.logger = logging.getLogger(__name__)
        self.line_points = line_points
        self.previous_positions = {}
        self.current_directions = {}
        self.object_history = defaultdict(list)
        self.last_crossing = {}
        self.crossing_cooldown = 2.0
        self.detected_events = []  # Store all detected events
        
    def get_zone(self, y_pos, y_line):
        """Determine if object is in inner or outer zone."""
        buffer = 50
        if y_pos < y_line - buffer:
            return "OUTSIDE"
        elif y_pos > y_line + buffer:
            return "INSIDE"
        return "TRANSITION"
        
    def detect_direction(self, track_ids, boxes, class_ids, names):
        """Detect direction of movement for tracked objects."""
        current_time = time.time()
        current_positions = {}
        
        # Get current positions and update history
        for track_id, box, class_id in zip(track_ids, boxes, class_ids):
            x, y = float(box[0]), float(box[1])
            object_name = names[int(class_id)]
            current_positions[track_id] = (x, y, object_name)
            
            self.object_history[track_id].append((y, current_time))
            if len(self.object_history[track_id]) > 5:
                self.object_history[track_id].pop(0)
        
        # Clean up old history
        self._cleanup_history(current_time, current_positions)
        
        # Check direction changes
        self._process_movements(current_time, current_positions)
        
        return self.current_directions
        
    def _cleanup_history(self, current_time, current_positions):
        """Clean up tracking history for inactive objects."""
        for track_id in list(self.object_history.keys()):
            if track_id not in current_positions:
                if current_time - self.object_history[track_id][-1][1] > 5.0:
                    del self.object_history[track_id]
                    if track_id in self.last_crossing:
                        del self.last_crossing[track_id]
                        
    def _process_movements(self, current_time, current_positions):
        """Process movement detection and direction changes."""
        y_line = self.line_points[0][1]
        movement_threshold = 30
        
        for track_id in current_positions:
            if len(self.object_history[track_id]) < 3:
                continue
                
            recent_y = np.mean([pos[0] for pos in self.object_history[track_id][-3:]])
            old_y = np.mean([pos[0] for pos in self.object_history[track_id][:3]])
            
            if track_id in self.last_crossing:
                if current_time - self.last_crossing[track_id] < self.crossing_cooldown:
                    continue
            
            if abs(recent_y - old_y) > movement_threshold:
                object_name = current_positions[track_id][2]
                
                if old_y < y_line and recent_y > y_line:
                    self._record_movement(track_id, "IN", object_name, current_time)
                elif old_y > y_line and recent_y < y_line:
                    self._record_movement(track_id, "OUT", object_name, current_time)
                    
    def _record_movement(self, track_id, direction, object_name, current_time):
        """Record a detected movement event."""
        self.current_directions[track_id] = (direction, object_name)
        self.last_crossing[track_id] = current_time
        
        event = {
            "timestamp": current_time,
            "track_id": track_id,
            "object_type": object_name,
            "direction": direction
        }
        self.detected_events.append(event)
        self.logger.info(f"{object_name} {track_id} moved {direction} (confidence: HIGH)")
        
    def get_events(self):
        """Return all detected events."""
        return self.detected_events

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

def process_h264_video(video_path, model_path, pause_event=None):
    """Process a recorded video file for object tracking and direction detection."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting video processing: {video_path}")
    
    try:
        # Load the YOLO model with explicit task
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
        track_history = defaultdict(lambda: [])
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

def process_tracking_results(result, detector, track_history, frame, frame_width, line_points):
    """Process and visualize tracking results for a single frame."""
    if result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        
        # Detect and update directions
        current_directions = detector.detect_direction(track_ids, boxes, class_ids, result.names)
        
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
            if track_id in current_directions:
                direction, object_name = current_directions[track_id]
                label = f"{object_name} {track_id}: {direction}"
                cv2.putText(annotated_frame, label,
                          (int(x - w/2), int(y - h/2 - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        annotated_frame = frame
        
    # Draw detection lines
    draw_detection_lines(annotated_frame, line_points, frame_width)
    
    return annotated_frame

def draw_detection_lines(frame, line_points, frame_width):
    """Draw the detection lines on the frame."""
    # Draw main line
    cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)
    
    # Draw buffer zone lines
    buffer = 50
    draw_dashed_line(frame,
                    (0, line_points[0][1] - buffer),
                    (frame_width, line_points[0][1] - buffer),
                    (0, 255, 0), 1)
    draw_dashed_line(frame,
                    (0, line_points[0][1] + buffer),
                    (frame_width, line_points[0][1] + buffer),
                    (0, 255, 0), 1)

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
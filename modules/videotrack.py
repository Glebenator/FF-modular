from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np
import time

class FridgeDirectionDetector:
    def __init__(self, line_points):
        self.line_points = line_points
        self.previous_positions = {}
        self.current_directions = {}
        self.object_history = defaultdict(list)  # Track multiple positions
        self.last_crossing = {}  # Prevent duplicate detections
        self.crossing_cooldown = 2.0  # Seconds before same object can trigger again
        
    def get_zone(self, y_pos, y_line):
        """Determine if object is in inner or outer zone"""
        buffer = 50  # Buffer zone around the line to prevent flickering
        if y_pos < y_line - buffer:
            return "OUTSIDE"
        elif y_pos > y_line + buffer:
            return "INSIDE"
        return "TRANSITION"
        
    def detect_direction(self, track_ids, boxes, class_ids, names):
        current_time = time.time()
        current_positions = {}
        
        # Get current positions and update history
        for track_id, box, class_id in zip(track_ids, boxes, class_ids):
            x, y = float(box[0]), float(box[1])
            object_name = names[int(class_id)]
            current_positions[track_id] = (x, y, object_name)
            
            # Keep track of last 5 positions for smoothing
            self.object_history[track_id].append((y, current_time))
            if len(self.object_history[track_id]) > 5:
                self.object_history[track_id].pop(0)
        
        # Clean up old history
        for track_id in list(self.object_history.keys()):
            if track_id not in current_positions:
                if current_time - self.object_history[track_id][-1][1] > 5.0:  # Remove after 5 seconds
                    del self.object_history[track_id]
                    if track_id in self.last_crossing:
                        del self.last_crossing[track_id]
        
        # Check direction changes
        y_line = self.line_points[0][1]
        for track_id in current_positions:
            if len(self.object_history[track_id]) < 3:  # Need at least 3 positions for reliable detection
                continue
                
            # Get average positions for more stability
            recent_y = np.mean([pos[0] for pos in self.object_history[track_id][-3:]])
            old_y = np.mean([pos[0] for pos in self.object_history[track_id][:3]])
            
            # Skip if we've recently detected a crossing for this object
            if track_id in self.last_crossing:
                if current_time - self.last_crossing[track_id] < self.crossing_cooldown:
                    continue
            
            # Detect significant movement across line
            movement_threshold = 30  # Minimum Y movement to consider it significant
            if abs(recent_y - old_y) > movement_threshold:
                object_name = current_positions[track_id][2]
                
                # Determine direction based on average positions
                if old_y < y_line and recent_y > y_line:
                    direction = "IN"
                    self.current_directions[track_id] = (direction, object_name)
                    self.last_crossing[track_id] = current_time
                    print(f"{object_name} {track_id} moved {direction} (confidence: HIGH)")
                    
                elif old_y > y_line and recent_y < y_line:
                    direction = "OUT"
                    self.current_directions[track_id] = (direction, object_name)
                    self.last_crossing[track_id] = current_time
                    print(f"{object_name} {track_id} moved {direction} (confidence: HIGH)")
        
        return self.current_directions

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed line on the image"""
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    dashes = int(dist / (2 * dash_length))
    for i in range(dashes):
        start = np.array(pt1) + (i * 2 * dash_length / dist) * (np.array(pt2) - np.array(pt1))
        end = np.array(pt1) + ((i * 2 + 1) * dash_length / dist) * (np.array(pt2) - np.array(pt1))
        start = tuple(map(int, start))
        end = tuple(map(int, end))
        cv2.line(img, start, end, color, thickness)

def process_h264_video(video_path, model_path):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define counting line points - horizontal line in the middle of the frame
    line_points = [(0, frame_height // 2), (frame_width, frame_height // 2)]
    
    # Store the track history
    track_history = defaultdict(lambda: [])
    
    # Initialize direction detector
    detector = FridgeDirectionDetector(line_points)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO tracking on the frame
        results = model.track(
            source=frame, 
            conf=0.3,
            iou=0.5,
            persist=True,
            verbose=True,
            stream=True,
            stream_buffer=True,
            visualize=False,
            vid_stride=5,
            max_det=3,
            tracker="botsort.yaml"
        )
        
        # Process the results generator
        for result in results:
            # Get the boxes and track IDs
            if result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.cpu().tolist()
                
                # Detect and update directions
                current_directions = detector.detect_direction(track_ids, boxes, class_ids, model.names)
                
                # Draw the tracking visualization
                annotated_frame = result.plot()
                
                # Plot the tracks and labels
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    
                    # Add direction label if available
                    if track_id in current_directions:
                        direction, object_name = current_directions[track_id]
                        label = f"{object_name} {track_id}: {direction}"
                        cv2.putText(annotated_frame, label, 
                                  (int(x - w/2), int(y - h/2 - 10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw the main detection line
                cv2.line(annotated_frame, line_points[0], line_points[1], (0, 255, 0), 2)
                
                # Draw buffer zone lines (dashed)
                buffer = 50
                draw_dashed_line(annotated_frame, 
                               (0, line_points[0][1] - buffer), 
                               (frame_width, line_points[0][1] - buffer), 
                               (0, 255, 0), 1)
                draw_dashed_line(annotated_frame, 
                               (0, line_points[0][1] + buffer), 
                               (frame_width, line_points[0][1] + buffer), 
                               (0, 255, 0), 1)
                
            else:
                annotated_frame = frame
                # Draw the detection lines even when no objects are detected
                cv2.line(annotated_frame, line_points[0], line_points[1], (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Fridge Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "recordings/videos/motion_20241111_175559.h264"  # Replace with your video path
    model_path = "../../models/yolo11n_ncnn_model"  # Replace with your model path
    process_h264_video(video_path, model_path)
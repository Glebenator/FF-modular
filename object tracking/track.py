from collections import defaultdict
import cv2
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO

class DirectionDetector:
    def __init__(self, line_points):
        self.line_points = line_points
        self.previous_positions = {}
        self.current_directions = {}  # Store current direction for each object
        
    def detect_direction(self, track_ids, boxes, class_ids, names):
        current_positions = {}
        
        # Get current positions for all tracked objects
        for track_id, box, class_id in zip(track_ids, boxes, class_ids):
            x, y = float(box[0]), float(box[1])
            object_name = names[int(class_id)]
            current_positions[track_id] = (x, y, object_name)
            
        # Check line crossings
        for track_id in current_positions:
            if track_id in self.previous_positions:
                previous_y = self.previous_positions[track_id][1]
                current_y = current_positions[track_id][1]
                object_name = current_positions[track_id][2]
                
                # Line equation
                y_line = self.line_points[0][1]  # Since it's a horizontal line
                
                # Check if object crossed the line and update direction
                if previous_y < y_line and current_y >= y_line:
                    direction = "OUT"
                    self.current_directions[track_id] = (direction, object_name)
                    print(f"{object_name} {track_id} moved {direction}")
                elif previous_y >= y_line and current_y < y_line:
                    direction = "IN"
                    self.current_directions[track_id] = (direction, object_name)
                    print(f"{object_name} {track_id} moved {direction}")
        
        # Update previous positions
        self.previous_positions = {k: (v[0], v[1]) for k, v in current_positions.items()}
        
        return self.current_directions

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1536, 864)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO model
model = YOLO("../../models/yolo11n_ncnn_model")

# Store the track history
track_history = defaultdict(lambda: [])

# Define counting line points - horizontal line in the middle of the frame
line_points = [(0, 432), (1536, 432)]

# Initialize direction detector
detector = DirectionDetector(line_points)

# Loop through the camera frames
while True:
    # Capture frame from Pi camera
    frame = picam2.capture_array()
    
    # Run YOLO tracking on the frame
    results = model.track(
        source=frame, 
        conf=0.3, 
        iou=0.5, 
        persist=True, 
        verbose=False,
        stream=True,
        stream_buffer=True,
        visualize=True, 
        vid_stride=2, 
        max_det=3, 
        tracker="botsort.yaml"
    )

    # Get the boxes and track IDs
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.cpu().tolist()
        
        # Detect and update directions
        current_directions = detector.detect_direction(track_ids, boxes, class_ids, model.names)
        
        # Draw the tracking visualization
        annotated_frame = results[0].plot()
        
        # Plot the tracks and labels
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:  # retain 30 tracks
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

        # Draw the detection line
        cv2.line(annotated_frame, line_points[0], line_points[1], (0, 255, 0), 2)
    else:
        annotated_frame = frame
        # Draw the detection line even when no objects are detected
        cv2.line(annotated_frame, line_points[0], line_points[1], (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Object Direction Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
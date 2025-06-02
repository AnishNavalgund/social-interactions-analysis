"""
Real-Time Object Detection with YOLO. 

Purpose of this is to check what all objects are identified and its accuracy. 
Helpful to then plan a pipeline to get quantifiable social interactions metrics. 

This script loads a YOLO model and,
Draws bounding boxes and class labels on each frame and displays them live.

Usage: Execute - "poetry run python scripts/object_detection.py"
"""

import cv2
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("models/yolo11l.pt")  # make sure model is downloaded in models dir 


video_path = "data/raw/video_example_peter.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video file")

class_names = model.names
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    boxes = results.boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{class_names[cls_id]} {conf:.2f}"

        # Draw bbox and print label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Show video
    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"\n totally {frame_idx} frames displayed")

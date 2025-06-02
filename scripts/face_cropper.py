"""
Face Tracking with YOLO!

This script detects and crops faces in a video using YOLO
Cropped face images are saved frame-by-frame in the output dir

"""

import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO, settings
from collections import defaultdict

def run_face_tracking(video_path, model_path, output_dir):
    # Load YOLOv11m-face model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # custom output dir
    # settings.update({'runs_dir': 'outputs'})
    model = YOLO(model_path)

    # Output path for cropped face images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running detection on {video_path}")
    results = model.track(
        source=video_path,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        device=device,
        imgsz=640,
        classes=[0],  # person class
        verbose=False
    )

    # face_log dictionary to stores the face id and the frame number and the face path
    face_log = defaultdict(list)
    frame_count = 0

    for result in results:
        frame = result.orig_img
        # print(frame)
        frame_count += 1
        frame_faces = []

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                face_crop = frame[y1:y2, x1:x2]
                face_id = int(track_id)

                # Save cropped face image
                face_path = output_dir / f"face_{face_id}_f{frame_count:04d}.jpg"
                cv2.imwrite(str(face_path), face_crop)

                face_log[face_id].append((frame_count, face_path))

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    print(f"Saved {sum(len(v) for v in face_log.values())} images of faces")
    return face_log


if __name__ == "__main__":
    video_path = "data/raw/video_example_peter.mp4"
    model_path = "models/yolov11m-face.pt"
    output_dir="outputs/new_faces"

    if os.path.exists(video_path):
        run_face_tracking(video_path, model_path, output_dir)
    else:
        print(f"No video in {video_path}")

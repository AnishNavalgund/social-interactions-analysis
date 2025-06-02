"""
Gaze Detection with L2CS-Net

This script loads a L2CS-Net model and runs gaze detection on a video file.
Draws bounding boxes on the face and prints if the person is looking at the camera or not. 
Also saves the output video with the gaze detection overlay and the Gaze timeline plot

It also prints the following metrics in the console: 
- Total joint attention time
- Percentage looking at baby
- Looking intervals

Using this script, we effectively tell how long the person is looking at the baby.
In the future, we can make it multi-modal by adding speech recognition and determine deeper insights. 

Usage: Execute - "poetry run python scripts/gaze_detection.py"
"""

import cv2
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from l2cs import Pipeline, render

# INPUTS
video_path = "data/raw/video_example_peter.mp4"
output_video_path = "outputs/new_gaze_output.mp4"
model_weights = Path("models/L2CSNet_gaze360.pkl")

# CONFIG 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yaw_thresh = 0.3  # radians
pitch_thresh = 0.3
sampling_rate = 1  # every frame
draw_label = True

# INITIALIZE GAZE PIPELINE
gaze_pipeline = Pipeline(weights=model_weights, arch="ResNet50", device=device)

# STORAGE 
timestamps = []
is_looking_list = []
looking_intervals = []

# VIDEO CAPTURE 
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VIDEO WRITER 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
inside_looking_interval = False
start_time = None

while True:
    ret, frame = cap.read()
    # print(ret.shape)
    if not ret:
        break

    frame_count += 1
    # print(frame_count)
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

    if frame_count % sampling_rate != 0:
        continue
    
    time_sec = frame_count / fps
    # print(time_sec)

    try:
        results = gaze_pipeline.step(frame)
    except ValueError:
        out_writer.write(frame)
        continue

    if len(results.pitch) == 0:
        out_writer.write(frame)
        continue

    pitch = results.pitch[0]
    yaw = results.yaw[0]
    bbox = results.bboxes[0]

    is_looking = abs(pitch) < pitch_thresh and abs(yaw) < yaw_thresh
    # print(is_looking)
    # print(pitch)

    label = "Looking at cam" if is_looking else "Not looking at cam"
    color = (0, 255, 0) if is_looking else (0, 0, 255)

    # Save timestamp & interval tracking
    timestamps.append(time_sec)
    is_looking_list.append(1 if is_looking else 0)

    if is_looking:
        if not inside_looking_interval:
            start_time = time_sec
            # print(time_sec)
            inside_looking_interval = True
    else:
        if inside_looking_interval:
            end_time = time_sec
            looking_intervals.append((start_time, end_time))
            inside_looking_interval = False

    # Print bbox & label
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if draw_label:
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Write frame
    out_writer.write(frame)

# final interval close
if inside_looking_interval:
    looking_intervals.append((start_time, frame_count / fps))


cap.release()
out_writer.release()


frame_duration = sampling_rate / fps
total_joint_attention_time = sum(is_looking_list) * frame_duration

print("\n ******* GAZE SUMMARY ******* ")

print(f"Video Duration: {timestamps[-1]:.2f} sec")
print(f"Total Joint Attention Time: {total_joint_attention_time:.2f} sec")
print(f"Percentage Looking at Baby: {100 * sum(is_looking_list) / len(is_looking_list):.1f}%")
print(f"Looking Intervals: {len(looking_intervals)} segments")

for start, end in looking_intervals:
    print(f"   - From {start:.2f}s to {end:.2f}s ({end - start:.2f}s)")

print(f"Output video saved at: {output_video_path}")

print("**************************** \n")

# --- PLOT ---
fig, ax = plt.subplots(figsize=(15, 3))
for i, t in enumerate(timestamps):
    color = 'green' if is_looking_list[i] else 'white'
    ax.bar(t, 1, width=frame_duration, color=color)

ax.set_yticks([])
ax.set_xlabel("Time (seconds)")
ax.set_title("Gaze Timeline: Green spikes = person looking at baby")
plt.tight_layout()
plt.show()
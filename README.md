# Social Interactions Analysis

## Overview and Features

The project aims to quantify social interactions of young children based on chest-mounted egocentric videos which will be helpful to understand the physcology of social development of the child. 

Due to time constraint, Gaze estimation feature is implemented along with other initial video analysis features. (eg: real-time object and face detection)

#### Features:

Gaze estimation feature helps detect when another person is looking at the child (face-to-face joint attention), for how long, and how often.

#### Computer Vision Models used: 

- L2CS-Net (gaze360)
- YOLOv11m-face
- YOLOv11l

## Repository Structure

```bash
social-interactions-analysis/
├── data/
│   └── raw/                        # Input egocentric video
├── models/                         # Pretrained L2CSNet and YOLO weights
├── notebooks/                      
│   ├── 01_gpu_check.ipynb          # Check GPU status
│   ├── 02_video_analysis.ipynb     # Video analysis
│   └── 03_gaze_model_check.ipynb   # Gaze model check
├── outputs/                        # Output gaze detection result, plot and annotated video
├── scripts/                        
│   ├── face_cropper.py             # Face cropper (helper script for 03_gaze_model_check.ipynb)
│   ├── gaze_detection.py           # Gaze detection 
│   └── object_detection.py         # Object detection
├── pyproject.toml                  # Poetry dependency manager
└── README.md                       
```

## Model Weights
Please download the weights from the following links and place them in the `models/` directory:
- L2CS-NET: https://drive.google.com/file/d/18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ/view?usp=drive_link
- YOLOv11: https://docs.ultralytics.com/models/yolo11/#performance-metrics:~:text=68.0-,YOLO11l,-640
- YOLOv11-face: https://github.com/akanametov/yolo-face#:~:text=yolov11s%2Dface.pt-,yolov11m%2Dface.pt,-yolov11l%2Dface.pt

## Usage
1. Clone the repository
```bash	
git clone https://github.com/anishknavalgund/social-interactions-analysis.git
```
2. Install the dependencies. Recommended Python version: 3.10.x
```bash
poetry install
```
3. Activate the virtual environment
```bash
source .venv/bin/activate
```
4. To run Gaze Detection,
```bash	
poetry run python scripts/gaze_detection.py
```
5. To run Object Detection,
```bash
poetry run python scripts/object_detection.py
```

## Output Summary
1. Annotated video from gaze detection: outputs/annotated_gaze_output.mp4
2. Gaze timeline plot: outputs/gaze_plot.png
3. Printable metrics when gaze detection is run: 
```bash
- Total joint attention duration
- Percentage looking at baby
- Looking intervals
```
Note: The printable metrics are copied to a gaze_metrics.txt file in the outputs/ directory along with the gaze plot. The annotated video is present in the private link - https://1drv.ms/f/c/4640ea4c5aca3f14/EppJpf1CBd9DnEo6e_QXU7kBKHiz03tBW5KJFGgn72K54A

## Challenges
- In the video, faces are partially visible in some frames. This was a challenge for the gaze detection model
- Generalization is hard i.e pretrained models (like L2CSNet or YOLO) are trained on normal camera views or general-purpose datasets, not egocentric views of children.
- Just using L2CSNet for gaze detection does not perform well. 
- Gaze estimation models expect clean, front crops, but YOLO bounding boxes often vary in size and quality. Hence, a combination of YOLO and L2CSNet needs more time and effort. 
- Handling frames with no detections required special care to prevent crashes

## Considerations:
- Detecting gaze is only part of the vision (speech and gestures can be incorporated in the future)
- To add more preprocessing steps to improve the quality of the video 
- Design own custom models 
- To try lightweight head-mounted cameras to study the gaze of the child
- To collect more egocentric data and label it for better generalization of the models. Also consider different environments. 

## Author
- *Anish Navalgund*
- *Email: anishk.navalgund@gmail.com*



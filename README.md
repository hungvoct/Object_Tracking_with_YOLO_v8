# Single-camera Multi Object Tracking using YOLOv8 and DeepSORT

## Project Overview
This repository demonstrates multi-object tracking by combining the **YOLOv8** object detector with the **DeepSORT** tracker. Given an input video, the system detects objects in each frame and assigns persistent IDs as they move across frames.

## Result Example
![Tracking Result](singlecam_tracking.PNG)

## Features
- Object detection using YOLOv8  
- Appearance-based tracking with DeepSORT  
- Color-coded bounding boxes with track IDs and confidence scores  
- Easy integration: swap in other YOLO weights or DeepSORT feature encoders  

## Prerequisites
- Python 3.8+  
- OpenCV 4.x  
- NumPy  

## Installation
Clone the repo:  
   ```bash
   git clone https://github.com/nwojke/deep_sort.git



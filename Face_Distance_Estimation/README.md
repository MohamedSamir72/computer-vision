# Depth Estimation and Eye Distance Measurement

This project combines MediaPipe's FaceMesh for facial landmark detection and the MiDaS model for depth estimation using ONNX runtime. It captures webcam input, detects eye landmarks, estimates depth, and calculates the distance between the eyes.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- ONNX
- ONNX Runtime
- NumPy

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/depth-estimation.git
    cd depth-estimation
    ```

2. **Install the required packages**:
    ```bash
    pip install opencv-python mediapipe onnx onnxruntime numpy
    ```

3. **Download the MiDaS model**:
    Download the MiDaS model and place it in the project directory. You can get the model from [here](https://github.com/isl-org/MiDaS).

## Usage

1. **Run the script**:
    ```bash
    python depth_estimation.py
    ```

## Code Explanation

### Import Libraries

```python
import cv2
import mediapipe as mp
import numpy as np
import math
import onnxruntime as ort

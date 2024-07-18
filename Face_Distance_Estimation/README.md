# Depth Estimation and Eye Distance Measurement

This project combines MediaPipe's FaceMesh for facial landmark detection and the MiDaS model for depth estimation using ONNX runtime. It captures webcam input, detects eye landmarks, estimates depth, and calculates the distance between the eyes.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
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

## Features
- Real-time face landmark detection using MediaPipe.
- Depth estimation using the MiDaS model.
- Eye center detection and distance calculation.
- Visualization of depth map and landmark detection.

## References
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
[MiDaS Model](https://pytorch.org/hub/intelisl_midas_v2/)

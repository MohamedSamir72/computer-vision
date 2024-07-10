Certainly! Below is a sample README.md file that explains the usage and details of your object detection and car counting script using YOLOv8 and OpenCV:

---

# Object Detection and Car Counting

This script performs real-time object detection using YOLOv8 and counts cars entering and exiting predefined regions based on their centroids.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Ultralytics (`pip install ultralytics`)

## Installation

1. Install dependencies:

   ```bash
   pip install opencv-python ultralytics
   ```

## Usage

1. Place your video file (`video.mp4`) in the same directory as the script.

2. Run the script:

   ```bash
   python object_detection_car_counting.py
   ```

3. To quit the application, press `q` while the window displaying the video is active.

## Functionality

- **Object Detection:** Uses YOLOv8 to detect objects in each frame of the video.
- **Car Counting:** Counts cars entering and exiting predefined regions based on their centroids.
- **Visualization:** Draws bounding boxes around detected cars, displays centroids, and indicates regions of interest (ROI) for counting.

## Reference Lines

Two reference lines are drawn horizontally on the frame to mark the regions for counting cars:

- **Cars In:** Between x-coordinates 30 and 300 (y-coordinate 350).
- **Cars Out:** Between x-coordinates 340 and 550 (y-coordinate 350).

## Output

The script displays a window showing the video with:
- Bounding boxes around detected cars.
- Centroids of each car.
- Counts of cars entering and exiting the defined regions.

## Customization

- **Adjustment of Parameters:** You can adjust the offset (`offset = 3`) for fine-tuning the detection of centroids within the ROI.

## Notes

- Ensure your environment has the necessary packages installed (`opencv-python` and `ultralytics`).
- The script assumes YOLOv8 model weights are stored in `yolov8n.pt` in the same directory as the script.

---

---

# Object Detection and Car Counting

This script performs real-time object detection using YOLOv8 and counts cars entering and exiting predefined regions based on their centroids.

## Requirements

- Python 3.x
- OpenCV 
- Ultralytics 

## Installation

1. Install dependencies:

   ```bash
   pip install opencv-python
   pip install ultralytics
   ```

## Usage

1. Place your video file (`video.mp4`) in the same directory as the script. [video](https://drive.google.com/file/d/1nIT_Aun0yGq38zOCC6SKdPanyl0TI2hH/view?usp=sharing)

2. Get the weights of YOLO model. [yolov8n.pt](https://drive.google.com/file/d/1huGZnAoj0rEBolhNv-59s-ZBshLxxRcM/view?usp=sharing)

3. Run the script:

   ```bash
   python object_detection_car_counting.py
   ```

4. To quit the application, press `q` while the window displaying the video is active.

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

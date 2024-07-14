---

# Vehicle Tracking and Counting Using OpenCV

This project demonstrates vehicle tracking and counting in a video using Python and OpenCV. The script processes each frame of a video, applies background subtraction, and identifies vehicles based on contours.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`) library
- NumPy library

## Installation

1. Install dependencies:
   ```bash
   pip install numpy opencv-python
   ```

## Usage

### Running the Script

Replace `"video.mp4"` with your video file path in the script (`video.mp4` should be located in the same directory as the script).

```python
python Object_Tracking_OpenCV.py
```

### Key Components

- **Background Subtraction**: Uses MOG2 method from OpenCV to subtract background from the video frames.

- **Image Processing**: Applies Gaussian blur and thresholding to preprocess the frames for contour detection.

- **Contour Detection**: Identifies contours in the processed frames that exceed a certain area threshold (400 pixels).

- **Vehicle Counting**: Tracks vehicles crossing a predefined reference line (`y_ref`) within the frame.

## Customization

- Adjust `y_ref_roi` and `offset` variables to fine-tune the detection area and accuracy based on your video.

- Uncomment additional `cv2.imshow()` statements (`ROI`, `Mask`, `Dilation`, `Open`) for detailed debugging and visualization.


### Notes:

- Ensure your video file (`video.mp4`) is accessible and correctly referenced in the script.
- Modify the `y_ref_roi` and `offset` parameters to suit your specific video dimensions and vehicle detection requirements.
- Provide appropriate links and credit to yourself or your organization under the Author section.
- Include any additional information or instructions specific to your project setup or deployment needs.

This template provides a structured outline for users to understand your project, set it up, and customize it as needed. Adjust the sections and details according to your project's specific requirements and preferences.

import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

### To Check the class id for car
# # Get the class names
# class_names = model.names
# # Print the class names
# for idx, name in class_names.items():
#     print(f"Class {idx}: {name}")

# Function to perform object detection on a video
def detect_objects_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to match YOLO input size (640x640)
        frame = cv2.resize(frame, (640, 640))

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes and labels on the image
        for result in results:  
            boxes = result.boxes
            for box in boxes:
                # Extract only cars
                check_class = int(box.cls.item())
                if check_class == 2:
                    # Get coordinates and dimensions of the box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw rectangle and put the class name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Object Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Test the function with the video file
detect_objects_video('video.mp4')

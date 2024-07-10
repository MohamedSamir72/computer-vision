import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Global variables
offset = 3
cars_in = 0
cars_out = 0

### To Check the class id for car
# # Get the class names
# class_names = model.names
# # Print the class names
# for idx, name in class_names.items():
#     print(f"Class {idx}: {name}")


# Get the center point of contour
def get_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1

    return (cx, cy)


def detect_objects_video(video_path):
    global cars_in
    global cars_out

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
                    
                    # Get the center point of the bounding box
                    w = x2 - x1
                    h = y2 - y1
                    cx, cy = get_center(x1, y1, w, h)
                    
                    # Check if the center of car in the reference region or not
                    if (cy < (350+offset)) and (cy > (350-offset)):
                        if (cx >= 30) and (cx <= 300):
                            cars_in+=1
                            print("in")

                        elif (cx >= 340) and (cx <= 550):
                            cars_out+=1
                            print("out")

                    # Draw rectangle and put the class name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), 2)
                    cv2.putText(frame, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    


        # Draw two reference lines
        cv2.line(frame, (30, 350), (270, 350), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (340, 350), (550, 350), (0, 0, 255), 1, cv2.LINE_AA)

        # Display the count of cars detected
        cv2.putText(frame, f'Cars In: {cars_in}', (20, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
        cv2.putText(frame, f'Cars Out: {cars_out}', (480, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)


        # Display the frame with detections
        cv2.imshow('Object Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_objects_video('video.mp4')
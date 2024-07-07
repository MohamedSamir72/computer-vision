import numpy as np
import cv2

cap = cv2.VideoCapture("video.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and display frames from the video
while cap.isOpened():
    ret, frame = cap.read()

    # To ignore the error when the video finished
    try:
        cv2.imshow("video", frame)
    except:
        print("Video finished")
        break
    
    # Adjust delay (20 milliseconds here, adjust as needed)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


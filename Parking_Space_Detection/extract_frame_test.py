import cv2

cap = cv2.VideoCapture('parking_crop.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

ret, frame = cap.read()

if ret:
    frame_path = 'test_frame.jpg'
    cv2.imwrite(frame_path, frame)
    print(f"Frame saved as {frame_path}")
else:
    print("Error: Could not read frame from video.")


cap.release()

cv2.imshow('Captured Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

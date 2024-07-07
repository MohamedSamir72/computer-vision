import numpy as np
import cv2

cap = cv2.VideoCapture("video.mp4")

# Create Background Subtractor object using MOG2
BGS = cv2.createBackgroundSubtractorMOG2()

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and display frames from the video
while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, (680, 460))
    roi = frame[240:320, 10:670]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    # Apply background subtraction
    img_sub = BGS.apply(blur)
    _, thr = cv2.threshold(img_sub, 245, 255, cv2.THRESH_BINARY)

    # Apply dilation
    dilation = cv2.dilate(thr, np.ones((5, 5)))
    
    ### Fail Trial: No good results
    # Perform opening: erosion followed by dilation
    # kernel = np.ones((3, 3), np.uint8)
    # opening_img = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(area)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(roi, (x, y), (x+w, y+h), (0,255,0), 2)


    # To ignore the error when the video finished
    try:
        cv2.imshow("video", frame)
        # cv2.imshow("ROI", roi)
        cv2.imshow("Mask", img_sub)
        # cv2.imshow("Dialtion", dilation)
        # cv2.imshow("Open", opening_img)
    except:
        print("Video finished")
        break
    
    # Adjust delay (20 milliseconds here, adjust as needed)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
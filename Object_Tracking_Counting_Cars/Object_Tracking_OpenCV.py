import numpy as np
import cv2

# Some Global variables
detection = []
y_ref_roi = 30
offset = 2
count = 0


# Get the center point of contour
def get_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1

    return (cx, cy)


cap = cv2.VideoCapture("video.mp4")

# Create Background Subtractor object using MOG2
BGS = cv2.createBackgroundSubtractorMOG2()

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and display frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    
    try:
        frame = cv2.resize(frame, (680, 460))
    except:
        pass
    
    roi = frame[240:320, 10:670]
    
    # The reference line for main frame
    y_ref = 280
    
    # The reference line for detection
    cv2.line(frame, (10, y_ref), (670, y_ref), (255,0,0), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Num. Vehicles {count}", (200, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    # Apply background subtraction
    img_sub = BGS.apply(blur)
    _, thr = cv2.threshold(img_sub, 200, 255, cv2.THRESH_BINARY)

    # Apply dilation
    dilation = cv2.dilate(thr, np.ones((5, 5)))
    
    ### Fail Trial: No good results
    # Perform opening: erosion followed by dilation
    # kernel = np.ones((3, 3), np.uint8)
    # opening_img = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 400:
            # print(area)

            # Get the points of bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center = get_center(x, y, w, h)

            # Add center point to detection points
            detection.append(center)

            # Draw a rectangle and the center point of a vehicle
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(roi, center, 1, (0,0,255), 2)


            for cx, cy in detection:
                if (cy < (y_ref_roi + offset)) and (cy > (y_ref_roi - offset)):
                    count+=1
                    detection.remove((cx, cy))
                    print(f"num cars: {count}")


    # To ignore the error when the video finished
    try:
        cv2.imshow("video", frame)
        ### All these displays for testing
        # cv2.imshow("ROI", roi)
        # cv2.imshow("Mask", img_sub)
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
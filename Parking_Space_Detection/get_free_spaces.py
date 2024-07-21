import cv2
import pandas as pd


def extract_points_from_excel(file_name):
    # Read the Excel file
    df = pd.read_excel(file_name)

    # Initialize the list to hold the points
    points = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        set_index = int(row['Set'])
        x = int(row['X'])
        y = int(row['Y'])
        
        # Ensure the points list has enough sublists
        while len(points) <= set_index:
            points.append([])
        
        # Append the point to the appropriate sublist
        points[set_index].append((x, y))

    # Remove empty sublists
    points = [sublist for sublist in points if sublist]

    return points



cap = cv2.VideoCapture("parking_crop.mp4")

# Extract points from the Excel file
points = extract_points_from_excel("points.xlsx")

while cap.isOpened():
    height = 60
    free_spaces = 0

    ret, frame = cap.read()
    h, w, _ = frame.shape

    if not ret:
        print("No frames")
        break
    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (3, 3), 1)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 11)


    # Draw parking spaces
    for pt in points:
        
        img_crop = thr[pt[0][1]:pt[1][1], pt[0][0]:pt[1][0]]
        cv2.imshow(f"{pt}", img_crop)

        # Get number of pixels that is not equal zero
        count = cv2.countNonZero(img_crop)

        # Put number of pixels in each space
        x, y = pt[0]
        cv2.putText(frame, f"{count}", (x, y+height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1)
        
        if count < 1000:
            cv2.rectangle(frame, pt[0], pt[1], (0,255,0), 2)
            free_spaces+=1
        
        else:
            cv2.rectangle(frame, pt[0], pt[1], (0,0,255), 2)

    # Put number of free spaces
    cv2.putText(frame, f"Free spaces: {free_spaces}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Image", frame)
    cv2.imshow("Blur", blur)
    cv2.imshow("Threshold", thr)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()

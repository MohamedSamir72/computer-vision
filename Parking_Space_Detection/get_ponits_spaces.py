import cv2
import pandas as pd

name_window = "Image"
points = []


# Save points to an Excel file
def save_points_to_excel(points, file_name):
    # Convert the points to a DataFrame
    points_data = []
    for i, sublist in enumerate(points):
        for point in sublist:
            points_data.append([i+1, point[0], point[1]])
    
    df = pd.DataFrame(points_data, columns=["Set", "X", "Y"])

    # Save the DataFrame to an Excel file
    df.to_excel(file_name, index=False)
    print(f"Points saved to {file_name}")

def mouse_event(event, x, y, flags, param):
    global points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0,0,255), -1)
        print(x, y)

        if len(points) == 0 or not points[-1]:  # Start a new sublist if points is empty or the last sublist is empty
            points.append([(x, y)])
        else:
            cv2.line(img, points[-1][-1], (x, y), (0,255,0), 1, cv2.LINE_AA)
            points[-1].append((x, y))

        print(points)

    if event == cv2.EVENT_RBUTTONDOWN:
        # Start a new sublist for the new set of points
        points.append([])  
        print(points)


img = cv2.imread("test_frame.jpg")

# Creating a named window
cv2.namedWindow(name_window)

# Creating a callback function
cv2.setMouseCallback(name_window, mouse_event)

while True:
    cv2.imshow(name_window,img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        points = [sublist for sublist in points if sublist]  # Remove empty sublists
        print(points)

        # Save the points to an Excel file
        save_points_to_excel(points, "points.xlsx")
        
        break
    
cv2.destroyAllWindows()



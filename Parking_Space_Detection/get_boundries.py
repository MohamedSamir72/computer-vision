import cv2
import pandas as pd

name_window = "Image"
points = []
width = 135
height = 60
spaces = 7


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
    global points, width, height, spaces
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.circle(img, (x, y), 3, (0,0,255), -1)

        for i in range(spaces):
            # Draw left column
            cv2.rectangle(img, (x, y), (x+width, y+height), (255, 255, 0), 1)
            print([(x, y), (x+width, y+height)])
            points.append([(x, y), (x+width, y+height)])

            # Draw right column
            cv2.rectangle(img, (x+width, y), (x+(2*width), y+height), (255, 255, 0), 1)
            print([(x+width, y), (x+(2*width), y+height)])
            points.append([(x+width, y), (x+(2*width), y+height)])
            
            y = y+height


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



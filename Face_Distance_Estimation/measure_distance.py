import cv2
import mediapipe as mp
import math

# Initialize MediaPipe FaceMesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to get the center of the eye given the landmark indices
def get_eye_center(landmarks, indices, image_shape):
    x = int(sum(landmarks[i].x for i in indices) / len(indices) * image_shape[1])
    y = int(sum(landmarks[i].y for i in indices) / len(indices) * image_shape[0])
    return (x, y)

# Indices for the eye contours
LEFT_EYE_INDICES = [33, 133]
RIGHT_EYE_INDICES = [362, 263]

# Function to process and display face landmarks on webcam input
def process_webcam():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(image_rgb)
            
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image_bgr = cv2.flip(image_bgr, 1)

            # Draw the eye landmarks and the line between them
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # Extract the coordinates of the eye landmarks
                    left_eye_coords = get_eye_center(face_landmarks.landmark, LEFT_EYE_INDICES, image_bgr.shape)
                    right_eye_coords = get_eye_center(face_landmarks.landmark, RIGHT_EYE_INDICES, image_bgr.shape)

                    # Draw circles at the eye landmarks
                    cv2.circle(image_bgr, left_eye_coords, 5, (0, 255, 0), -1)
                    cv2.circle(image_bgr, right_eye_coords, 5, (0, 255, 0), -1)

                    # Draw a line between the eye landmarks
                    cv2.line(image_bgr, left_eye_coords, right_eye_coords, (0, 255, 0), 2)

                    # Calculate the number of pixels between the eye landmarks
                    x_coordinates = (left_eye_coords[0] - right_eye_coords[0]) ** 2
                    y_coordinates = (left_eye_coords[1] - right_eye_coords[1]) ** 2
                    num_pixels = math.sqrt(x_coordinates + y_coordinates)

                    cv2.putText(image_bgr, f'{int(num_pixels)} pixels', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            
            cv2.imshow('MediaPipe Face Mesh', image_bgr)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Run the function to process webcam input and show face landmarks
if __name__ == "__main__":
    process_webcam()

import cv2
import mediapipe as mp
import numpy as np
import math
import onnxruntime as ort

# Indices for the eye contours
LEFT_EYE_INDICES = [33, 133]
RIGHT_EYE_INDICES = [362, 263]

# Initialize MediaPipe FaceMesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load ONNX model
midas = ort.InferenceSession('midas_small.onnx')

# Function to get the center of the eye given the landmark indices
def get_eye_center(landmarks, indices, image_shape):
    x = int(sum(landmarks[i].x for i in indices) / len(indices) * image_shape[1])
    y = int(sum(landmarks[i].y for i in indices) / len(indices) * image_shape[0])
    return (x, y)

# Preprocess input image for MiDaS model
def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

# Function to process and display face landmarks on webcam input
def process_webcam():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))

            if not success:
                print("Ignoring empty camera frame.")
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Run MiDaS model for depth estimation
            img_preprocessed = preprocess(frame)
            depth_map = midas.run(None, {'input': img_preprocessed})[0]
            depth_map = depth_map.squeeze()
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = (depth_map * 255).astype(np.uint8)
            depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

            # Draw the eye landmarks and the line between them
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # Extract the coordinates of the eye landmarks
                    left_eye_coords = get_eye_center(face_landmarks.landmark, LEFT_EYE_INDICES, img_bgr.shape)
                    right_eye_coords = get_eye_center(face_landmarks.landmark, RIGHT_EYE_INDICES, img_bgr.shape)

                    # Draw circles at the eye landmarks
                    cv2.circle(img_bgr, left_eye_coords, 5, (0, 255, 0), -1)
                    cv2.circle(img_bgr, right_eye_coords, 5, (0, 255, 0), -1)

                    # Draw a line between the eye landmarks
                    cv2.line(img_bgr, left_eye_coords, right_eye_coords, (0, 255, 0), 2)

                    # Calculate the number of pixels between the eye landmarks
                    x_coordinates = (left_eye_coords[0] - right_eye_coords[0]) ** 2
                    y_coordinates = (left_eye_coords[1] - right_eye_coords[1]) ** 2
                    num_pixels = math.sqrt(x_coordinates + y_coordinates)

                    cv2.putText(img_bgr, f'{int(num_pixels)} pixels', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # Calculate the center point between the eye landmarks
                    center_coords = ((left_eye_coords[0] + right_eye_coords[0]) // 2, 
                                     (left_eye_coords[1] + right_eye_coords[1]) // 2)

                    # Get the depth value at the center point
                    depth_value = depth_map[center_coords[1], center_coords[0]]
                    
                    cv2.circle(img_bgr, center_coords, 5, (255, 0, 0), -1)
                    cv2.putText(img_bgr, f'Depth: {depth_value}', (center_coords[0] + 10, center_coords[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    
                    distance = (6.5 * depth_value) / num_pixels
                    distance = round(distance * 3, 2)   # Tune the value

                    cv2.putText(img_bgr, f'Distance: {distance} cm', (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow('MediaPipe Face Mesh', img_bgr)
            cv2.imshow('Depth Map', depth_map_colored)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_webcam()

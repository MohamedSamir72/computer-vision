import cv2
import os

def extract_frames(video_path, output_folder, interval=30):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            img_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_folder}")


if __name__ == "__main__":
    extract_frames("video.mp4", "output_images", interval=30)
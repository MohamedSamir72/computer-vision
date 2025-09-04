import cv2
import os

def extract_frames(video_path, output_folder, interval=30, img_count=0):

    if not os.path.exists(output_folder):
        print(f"File [{output_folder}] doesn't exist")

    else:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = img_count

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                img_path = os.path.join(output_folder, f"img_{saved_count:02d}.jpg")
                cv2.imwrite(img_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()

        
        print(f"Extracted {(saved_count - img_count):02d} frames from {video_path} to ==> {output_folder} directory")

        return saved_count


if __name__ == "__main__":
    img_counter = 0

    # Create [images] directory
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)

    # Process all videos in [videos] directory
    if os.path.exists("videos"):
        for vid in os.listdir("videos"):
            if vid.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join("videos", vid)
                
                img_counter = extract_frames(video_path, output_dir, interval=20, img_count=img_counter)

    
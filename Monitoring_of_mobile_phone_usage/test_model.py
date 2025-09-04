from ultralytics import RTDETR
import cv2

# Load pretrained RT-DETR model
model = RTDETR("best_1.pt")

# Run inference in streaming mode (frame by frame)
results = model.predict(source="videos/1.mp4", stream=True, show=False, conf=0.30)

for result in results:
    if result is None:
        continue

    # Plot only filtered detections
    frame = result.plot()

    # Resize for display
    resized_frame = cv2.resize(frame, (640, 480))

    cv2.imshow("RT-DETR Detection (Filtered)", resized_frame)

    # Quit if 'q' pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


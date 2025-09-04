from ultralytics import RTDETR
import cv2

# Load pretrained RT-DETR model
model = RTDETR("best.pt")

# Run inference in streaming mode (frame by frame)
results = model.track(source="videos/1.mp4", stream=True, show=False, imgsz=320, conf=0.30)

for result in results:
    if result is None:
        continue

    # Access bounding boxes and classes
    boxes = result.boxes.xyxy.cpu().numpy()                 
    classes = result.boxes.cls.cpu().numpy().astype(int)    

    frame = result.orig_img.copy()

    persons = []
    phones = []

    # Separate persons and phones
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        if cls == 0:  # 'person' class id = 0
            persons.append((x1, y1, x2, y2))
        elif cls == 1:  # 'phone' class id = 1
            phones.append((x1, y1, x2, y2))

    # Check if phone center is inside a person box
    for (person_x1, person_y1, person_x2, person_y2) in persons:
        for (phone_x1, phone_y1, phone_x2, phone_y2) in phones:
            phone_x_center = int((phone_x1 + phone_x2) / 2)
            phone_y_center = int((phone_y1 + phone_y2) / 2)

            if person_x1 <= phone_x_center <= person_x2 and person_y1 <= phone_y_center <= person_y2:
                # Draw red bounding box around the person
                cv2.rectangle(frame, (phone_x1, phone_y1), (phone_x2, phone_y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (person_x1, person_y1), (person_x2, person_y2), (0, 0, 255), 3)
                cv2.putText(frame, "Person+Phone", (person_x1, person_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Resize for display
    resized_frame = cv2.resize(frame, (640, 480))
    cv2.imshow("RT-DETR Person with Phone", resized_frame)

    # Quit if 'q' pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

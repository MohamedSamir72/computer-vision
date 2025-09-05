from ultralytics import RTDETR
import cv2
import argparse
import torch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RT-DETR Person with Phone Detection")
    parser.add_argument("--source", type=str, default="videos/1.mp4", 
                        help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Path to model weights file")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Image size for inference")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Confidence threshold for detection")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output video (optional)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for inference (e.g., 'cuda', 'cpu'). Auto-detects if not specified")
    args = parser.parse_args()

    # Set device (CUDA if available, otherwise CPU)
    if args.device:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            device = 'cpu'
            
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Load pretrained RT-DETR model on the specified device
    model = RTDETR(args.model)
    model.to(device)

    # Initialize video writer if output path is provided
    if args.output:
        # Get video properties from the first frame
        cap = cv2.VideoCapture(args.source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Run inference in streaming mode (frame by frame)
    results = model.track(
        source=args.source, 
        stream=True, 
        show=False, 
        imgsz=args.imgsz, 
        conf=args.conf,
        device=device  # Pass device to inference
    )

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
                    # Draw green bounding box around the phone
                    cv2.rectangle(frame, (phone_x1, phone_y1), (phone_x2, phone_y2), (0, 255, 0), 3)
                    # Draw red bounding box around the person
                    cv2.rectangle(frame, (person_x1, person_y1), (person_x2, person_y2), (0, 0, 255), 3)
                    cv2.putText(frame, "Person+Phone", (person_x1, person_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save frame if output path is provided
        if args.output:
            out.write(frame)

        # Resize for display
        resized_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("RT-DETR Person with Phone", resized_frame)

        # Quit if 'q' pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release resources
    if args.output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

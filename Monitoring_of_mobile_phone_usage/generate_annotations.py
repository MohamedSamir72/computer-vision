import os
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-tiny"

# Load model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Class names
class_names = ["person", "phone"]

# os.makedirs("output_images", exist_ok=True)   # Save images with bounding boxes
os.makedirs("labels", exist_ok=True)

for img in os.listdir("images"):
    if img.lower().endswith((".jpg", ".jpeg", ".png")):
        # Load image
        image = Image.open(f"images/{img}").convert("RGB")

        # Prepare text inputs
        text_labels = [class_names]

        # Inference
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]
        draw = ImageDraw.Draw(image)

        # Prepare annotations file
        label_path = os.path.join("labels", os.path.splitext(img)[0] + ".txt")

        with open(label_path, "w") as f:
            for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
                box = [round(x, 2) for x in box.tolist()]
                print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
                
                # # Draw box
                # draw.rectangle(box, outline="red", width=3)
                # draw.text((box[0], box[1]), f"{labels}: {score:.2f}", fill="red")

                # Bounding box coordinates
                x_min, y_min, x_max, y_max = box
                w, h = image.size
                box_width = x_max - x_min
                box_height = y_max - y_min
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2

                # Normalize coordinates
                box_width /= w
                box_height /= h
                x_center /= w
                y_center /= h

                # Save to txt
                f.write(f"{labels} {x_center} {y_center} {box_width} {box_height}\n")

            # # Save results to check for all bounding boxes
            # image.save(f"output_images/{img}")
import os
import torch
import shutil
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Model setup
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Input & output dirs
input_dir = "output_images"
labels_dir = "labels"
os.makedirs(labels_dir, exist_ok=True)

# Define labels
text_labels = [["a fish"]]
class_id = 0  

def save_yolo_labels(image_path):
    image = Image.open(image_path).convert("RGB")
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
    w, h = image.size

    # Create label file
    label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    with open(label_path, "w") as f:
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            # Box in (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = box.tolist()

            # Convert to YOLO format
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Saved labels: {label_path}")

# Generate labels for all images
for file in os.listdir(input_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        save_yolo_labels(os.path.join(input_dir, file))

print("All annotations saved in 'labels/' folder (YOLO format).")

# Create a dataset folder structure compatible with Roboflow
dataset_dir = "roboflow_dataset"
images_out = os.path.join(dataset_dir, "images")
labels_out = os.path.join(dataset_dir, "labels")
os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

# Copy images and labels into the dataset folder
for file in os.listdir(input_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        shutil.copy(os.path.join(input_dir, file), images_out)

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        shutil.copy(os.path.join(labels_dir, file), labels_out)

# Zip the dataset
shutil.make_archive("roboflow_dataset", "zip", dataset_dir)
print("Dataset zipped as 'roboflow_dataset.zip' (ready to upload to Roboflow).")

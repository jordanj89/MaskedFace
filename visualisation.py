from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO("runs/detect/train3/weights/best.pt")  # Update the path

# Select a few sample images from validation set
sample_images = [
    "MaskedFace/val/images/mask-011.png",
    "MaskedFace/val/images/mask-019.png",
    "MaskedFace/val/images/mask-023.png",
]

for img_path in sample_images:
    results = model(img_path)
    results[0].show()

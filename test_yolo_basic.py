from ultralytics import YOLO
import glob
import os

model_path = 'runs/detect/surgical_detector_v17/weights/best.pt'
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

model = YOLO(model_path)
print(f"Loaded model from {model_path}")

test_images = glob.glob('datasets/surgical_data/test/images/*.jpg')
if not test_images:
    print("ERROR: No test images found.")
    exit(1)

test_image = test_images[0]
print(f"Testing on {test_image}")

results = model(test_image)

for r in results:
    boxes = r.boxes
    print(f"Found {len(boxes)} objects")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = box.conf[0]
        class_name = model.names[cls_id]
        print(f"Detected: {class_name} with confidence {conf:.2f}")


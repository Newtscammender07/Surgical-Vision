import cv2
from yolov8_surgical_monitor import SurgicalMonitor

def main():
    monitor = SurgicalMonitor('runs/detect/surgical_detector_v17/weights/best.pt', confidence_threshold=0.10)
    
    # Load test image
    image = cv2.imread('datasets/surgical_data/test/images/0_jpg.rf.fc4cb2e60ebddb31edee6163353dc202.jpg')
    if image is None:
        print("Failed to load test image.")
        return
        
    print(f"Image shape: {image.shape}")
        
    # We will just run the model directly to see the raw output of the boxes
    results = monitor.model(image, conf=0.10)
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            print(f"\n--- Box {i} ---")
            print(f"box.xyxy[0]: {box.xyxy[0]}")
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            print(f"Unpacked and numpy'd: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(f"Integer casting: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

if __name__ == "__main__":
    main()

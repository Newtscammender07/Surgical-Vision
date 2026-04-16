from ultralytics import YOLO

def train_custom_model():
    """
    Template for fine-tuning YOLOv8 on a custom dataset of surgical tools / anomalies.
    
    1. Organize your dataset in YOLO format:
        datasets/
            surgical_data/
                images/
                    train/
                    val/
                labels/
                    train/
                    val/
                data.yaml
                
    2. 'data.yaml' should contain:
        path: ../datasets/surgical_data
        train: images/train
        val: images/val
        names:
          0: surgical_tool
          1: anomaly
          2: bleeding
          ...
    """
    # Load a pretrained YOLOv8 model for transfer learning
    print("Loading pretrained YOLOv8 model for transfer learning...")
    model = YOLO('yolov8n.pt') 

    # Train the model using the custom dataset
    print("Starting training...")
    # By default, Roboflow downloads to a folder matching the project name.
    # Make sure this path points to the data.yaml file inside that downloaded folder!
    # e.g., 'Surgical-Tools-Detection-1/data.yaml'
    
    yaml_path = "../datasets/surgical_data/data.yaml"
    
    results = model.train(
        data=yaml_path, 
        epochs=15,      # Short training run for demonstration
        imgsz=640,
        batch=16,
        project='runs/detect',
        name='surgical_detector_v18'
    )
    
    print("\nTraining complete. The best weights are saved in 'runs/detect/surgical_detector_v3/weights/best.pt'")
    print("Update 'yolov8_surgical_monitor.py' line 12 to use 'runs/detect/surgical_detector_v1/weights/best.pt' instead of 'yolov8n.pt'.")

if __name__ == "__main__":
    train_custom_model()

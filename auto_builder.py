import os
import subprocess
import cv2
import numpy as np
import shutil

def draw_transparent_tool(tool_type):
    # Create empty transparent image 200x200
    img = np.zeros((200, 200, 4), dtype=np.uint8)
    
    if tool_type == "scalpel":
        # Draw a scalpel-like shape (metallic grey)
        pts = np.array([[20, 100], [100, 90], [180, 80], [185, 95], [100, 105], [20, 110]], np.int32)
        cv2.fillPoly(img, [pts], (180, 180, 180, 255))
        # Handle
        cv2.line(img, (20, 105), (100, 100), (100, 100, 100, 255), 10)
        
    elif tool_type == "injection":
        # Draw a syringe
        # Body
        cv2.rectangle(img, (50, 80), (150, 120), (230, 230, 250, 200), -1)
        # Needle
        cv2.line(img, (150, 100), (190, 100), (200, 200, 200, 255), 2)
        # Plunger
        cv2.rectangle(img, (20, 95), (50, 105), (50, 50, 50, 255), -1)
        cv2.rectangle(img, (10, 85), (20, 115), (50, 50, 50, 255), -1)
        
    elif tool_type == "forceps":
        # Draw forceps (tweezer shape)
        pts1 = np.array([[20, 90], [100, 80], [180, 95], [100, 95]], np.int32)
        pts2 = np.array([[20, 110], [100, 120], [180, 105], [100, 105]], np.int32)
        cv2.fillPoly(img, [pts1], (150, 150, 170, 255))
        cv2.fillPoly(img, [pts2], (150, 150, 170, 255))
        # Hinge
        cv2.circle(img, (30, 100), 8, (100, 100, 100, 255), -1)
        
    return img

def main():
    # 1. Ensure directories exist
    os.makedirs('dataset_builder/input/foregrounds', exist_ok=True)
    os.makedirs('dataset_builder/input/backgrounds', exist_ok=True)
    
    # 2. Generate tool images
    print("Generating transparent tools (scalpel, injection, forceps)...")
    for tool in ["scalpel", "injection", "forceps"]:
        img = draw_transparent_tool(tool)
        cv2.imwrite(f'dataset_builder/input/foregrounds/{tool}.png', img)
            
    # 3. Generate background images
    print("Generating background bases...")
    for i in range(5):
        color = (150, 200, 150) if i % 2 == 0 else (200, 180, 150)
        bg = np.full((640, 640, 3), color, dtype=np.uint8)
        # add some noise
        noise = np.random.normal(0, 15, bg.shape).astype(np.uint8)
        bg = cv2.add(bg, noise)
        cv2.imwrite(f'dataset_builder/input/backgrounds/bg_{i}.jpg', bg)
        
    # 4. Run the dataset generator
    print("Running dataset synthesis script (dataset_generator.py)...")
    subprocess.run(["python", "dataset_generator.py"], check=True)
    
    # 5. Modify train_yolov8.py to point to the new data.yaml
    train_script = 'train_yolov8.py'
    with open(train_script, 'r') as f:
        code = f.read()
    
    old_path = '="../datasets/surgical_data/data.yaml"'
    old_path_2 = 'yaml_path = "../datasets/surgical_data/data.yaml"'
    
    # We will just strictly find and replace the likely line
    import re
    new_yaml = os.path.abspath("dataset_builder/output/data.yaml").replace('\\', '/').replace('//', '/')
    
    code = re.sub(r'yaml_path\s*=\s*".*?"', f'yaml_path = "{new_yaml}"', code)
    code = re.sub(r"yaml_path\s*=\s*'.*?'", f'yaml_path = "{new_yaml}"', code)

    with open(train_script, 'w') as f:
        f.write(code)
            
    # 6. Run the trainer
    print("Starting YOLOv8 training! This will take a few minutes...")
    subprocess.run(["python", train_script], check=True)

if __name__ == '__main__':
    main()

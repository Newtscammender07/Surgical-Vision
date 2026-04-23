import os
import cv2
import numpy as np
import random
import yaml
import shutil
from pathlib import Path

# --- CONFIGURATION ---
NUM_IMAGES_TO_GENERATE = 100
FOREGROUNDS_DIR = "dataset_builder/input/foregrounds"
BACKGROUNDS_DIR = "dataset_builder/input/backgrounds"
OUTPUT_DIR = "dataset_builder/output"

# Define classes you want to add
CLASSES = ["scalpel", "injection", "forceps", "clamps", "parker"]

def setup_directories():
    """Creates necessary directories for generation."""
    dirs_to_make = [
        FOREGROUNDS_DIR, BACKGROUNDS_DIR,
        f"{OUTPUT_DIR}/images/train", f"{OUTPUT_DIR}/images/val",
        f"{OUTPUT_DIR}/labels/train", f"{OUTPUT_DIR}/labels/val"
    ]
    
    # Cleanup previous run if exists
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)
        
    print(f"Directories created! Please place:")
    print(f"  1. Transparent PNG tool images in: {FOREGROUNDS_DIR}")
    print(f"     (Name them exactly as your classes, e.g., 'scalpel.png' or 'injection1.png')")
    print(f"  2. Normal background images in: {BACKGROUNDS_DIR}")
    print(f"     (E.g., pictures of operating tables, skin, cloths)")

def rotate_image(img, angle):
    """Rotates an image (with alpha channel) by given angle without cropping."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def overlay_image_alpha(bg, fg, x, y):
    """Overlays fg onto bg at (x,y), returning new image and bounding box."""
    # Ensure BG is BGR
    if bg.shape[2] == 4:
        bg = bg[:, :, :3]
        
    # Get dimensions
    fg_h, fg_w = fg.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    # Calculate bounding box coordinates
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + fg_w), min(bg_h, y + fg_h)

    # Calculate corresponding coordinates on foreground
    fg_x1 = max(0, -x)
    fg_y1 = max(0, -y)
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    # Check if inside background
    if x1 >= bg_w or y1 >= bg_h or x2 <= 0 or y2 <= 0:
        return bg, None  # Completely outside

    fg_crop = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    alpha = fg_crop[:, :, 3] / 255.0
    alpha_rgb = np.dstack((alpha, alpha, alpha))

    bg_crop = bg[y1:y2, x1:x2]
    
    # Check for empty crops
    if bg_crop.size == 0 or fg_crop.size == 0:
        return bg, None

    # Composite
    blended = alpha_rgb * fg_crop[:, :, :3] + (1 - alpha_rgb) * bg_crop
    bg[y1:y2, x1:x2] = blended

    # Extract non-transparent bounding box for tighter fit
    non_empty_columns = np.where(alpha.max(axis=0) > 0.1)[0]
    non_empty_rows = np.where(alpha.max(axis=1) > 0.1)[0]
    
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        tight_x1 = x1 + non_empty_columns.min()
        tight_x2 = x1 + non_empty_columns.max()
        tight_y1 = y1 + non_empty_rows.min()
        tight_y2 = y1 + non_empty_rows.max()
        
        # YOLO format: center_x, center_y, width, height (normalized)
        w = (tight_x2 - tight_x1) / bg_w
        h = (tight_y2 - tight_y1) / bg_h
        cx = ((tight_x1 + tight_x2) / 2) / bg_w
        cy = ((tight_y1 + tight_y2) / 2) / bg_h
        
        return bg, (cx, cy, w, h)
    
    return bg, None

def generate_dataset():
    backgrounds = [os.path.join(BACKGROUNDS_DIR, f) for f in os.listdir(BACKGROUNDS_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    foregrounds = [os.path.join(FOREGROUNDS_DIR, f) for f in os.listdir(FOREGROUNDS_DIR) if f.endswith('.png')]

    if not backgrounds or not foregrounds:
        print("Please add images to the input folders first!")
        return

    print(f"Starting generation of {NUM_IMAGES_TO_GENERATE} synthetic images...")

    for i in range(NUM_IMAGES_TO_GENERATE):
        bg_path = random.choice(backgrounds)
        bg = cv2.imread(bg_path)
        if bg is None: continue
        bg = cv2.resize(bg, (640, 640)) # Standard YOLO size

        split = "train" if random.random() < 0.8 else "val"
        labels = []
        
        # Add 1 to 4 random tools
        num_tools = random.randint(1, 4)
        for _ in range(num_tools):
            fg_path = random.choice(foregrounds)
            fg_name = os.path.basename(fg_path).split('.')[0].lower()
            
            # Map filename to class ID
            class_id = -1
            for idx, c_name in enumerate(CLASSES):
                if c_name in fg_name:
                    class_id = idx
                    break
            
            if class_id == -1: continue # Skip if unmapped

            fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
            if fg is None or fg.shape[2] != 4: continue # Needs alpha channel
            
            # Random scaling
            scale = random.uniform(0.1, 0.4)
            fg = cv2.resize(fg, (0,0), fx=scale, fy=scale)
            
            # Random rotation
            fg = rotate_image(fg, random.randint(0, 360))

            # Random position
            bg_h, bg_w = bg.shape[:2]
            fg_h, fg_w = fg.shape[:2]
            x = random.randint(-fg_w//3, bg_w - fg_w//3)
            y = random.randint(-fg_h//3, bg_h - fg_h//3)

            bg, bbox = overlay_image_alpha(bg, fg, x, y)
            
            if bbox:
                labels.append(f"{class_id} {bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}")

        if labels:
            img_filename = f"synth_{i}.jpg"
            lbl_filename = f"synth_{i}.txt"
            
            cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{img_filename}", bg)
            with open(f"{OUTPUT_DIR}/labels/{split}/{lbl_filename}", "w") as f:
                f.write("\n".join(labels))

    # Generate data.yaml
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"Done! Dataset saved to {OUTPUT_DIR}")
    print(f"You can now train with: model.train(data='{OUTPUT_DIR}/data.yaml')")

if __name__ == "__main__":
    if not os.path.exists(FOREGROUNDS_DIR):
        setup_directories()
    else:
        generate_dataset()

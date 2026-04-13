import os
import shutil

# Paths
source_dir = r"c:\Users\mouni\Downloads\augmented reality\datasets\surgical_data\Surgical-Dataset\Surgical-Dataset"
splits_dir = os.path.join(source_dir, "Test-Train Groups")
source_images = os.path.join(source_dir, "Images")
source_labels = os.path.join(source_dir, "Labels", "label object names")

target_dir = r"c:\Users\mouni\Downloads\augmented reality\datasets\Surgical-Dataset-YOLO"
splits = {
    'train': os.path.join(splits_dir, "train-obj_detector.txt"),
    'val': os.path.join(splits_dir, "test-obj_detector.txt")
}

# Create target directories
for split in splits.keys():
    os.makedirs(os.path.join(target_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, split, "labels"), exist_ok=True)

# Process splits
for split_name, txt_file in splits.items():
    print(f"Processing {split_name} split...")
    if not os.path.exists(txt_file):
        print(f"Warning: split file {txt_file} not found.")
        continue
        
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract filename from absolute path like /home/roboticslab/.../bisturi469.jpg
        img_filename = os.path.basename(line)
        base_name = os.path.splitext(img_filename)[0]
        lbl_filename = base_name + ".txt"
        
        # Source paths
        src_img = os.path.join(source_images, img_filename)
        src_lbl = os.path.join(source_labels, lbl_filename)
        
        # Target paths
        dst_img = os.path.join(target_dir, split_name, "images", img_filename)
        dst_lbl = os.path.join(target_dir, split_name, "labels", lbl_filename)
        
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"Missing file: {img_filename} or {lbl_filename}")

print("Dataset reformatting complete!")
print("Writing data.yaml...")

# Create data.yaml mapping
yaml_content = f"""train: {os.path.join(target_dir, 'train', 'images')}
val: {os.path.join(target_dir, 'val', 'images')}

nc: 3
names: ['bisturi', 'pinca', 'tesourareta']
"""

with open(os.path.join(target_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

print(f"Saved: {os.path.join(target_dir, 'data.yaml')}")

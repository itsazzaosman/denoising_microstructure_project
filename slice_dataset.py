import os
from PIL import Image
from tqdm import tqdm


BASE_DIR = "/project/community/aiosman/diffusion_project"
INPUT_DIR = os.path.join(BASE_DIR, "dataset_split")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_low_density")

# The quadrants to slice a 128x128 image into four 64x64 blocks
# (left, upper, right, lower)
QUADRANTS = [
    (0, 0, 64, 64),     # Top-Left
    (64, 0, 128, 64),   # Top-Right
    (0, 64, 64, 128),   # Bottom-Left
    (64, 64, 128, 128)  # Bottom-Right
]

def process_directory(split_name):
    in_split_dir = os.path.join(INPUT_DIR, split_name)
    out_split_dir = os.path.join(OUTPUT_DIR, split_name)

    print(f"\nProcessing {split_name} dataset...")
    
    # Iterate through each Aspect Ratio class folder
    for class_folder in os.listdir(in_split_dir):
        in_class_path = os.path.join(in_split_dir, class_folder)
        out_class_path = os.path.join(out_split_dir, class_folder)

        if not os.path.isdir(in_class_path):
            continue

        # Create the new class directory
        os.makedirs(out_class_path, exist_ok=True)
        
        images = [f for f in os.listdir(in_class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Class {class_folder}"):
            img_path = os.path.join(in_class_path, img_name)
            
            try:
                with Image.open(img_path) as img:
                    # Slice into 4 quadrants
                    for idx, box in enumerate(QUADRANTS):
                        cropped_img = img.crop(box)
                        
                        # Scale back to 128x128 using NEAREST to keep grain edges sharp
                        resized_img = cropped_img.resize((128, 128), Image.Resampling.NEAREST)
                        
                        # Save the new image
                        base_name, ext = os.path.splitext(img_name)
                        new_name = f"{base_name}_q{idx+1}{ext}"
                        resized_img.save(os.path.join(out_class_path, new_name))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    print("Starting dataset slicing...")
    # Process both train and test folders
    process_directory("train")
    process_directory("test")
    print("\nDataset successfully sliced and expanded!")
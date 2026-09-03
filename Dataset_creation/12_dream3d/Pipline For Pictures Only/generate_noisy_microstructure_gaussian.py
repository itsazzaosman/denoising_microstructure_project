#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import sys

# Define your input (clean) and output (noisy) directories
INPUT_DIR = Path(r"C:\Users\Ahmed Alhassan\OneDrive\Desktop\Pipline For Pictures Only\Ni_dataset_new\ni_clean_maps")
OUTPUT_DIR = Path(r"C:\Users\Ahmed Alhassan\OneDrive\Desktop\Pipline For Pictures Only\Ni_dataset_new\ni_g_noisy_maps")

def apply_experimental_noise(img):
    """Applies Gaussian noise (std=25) and reduces contrast."""
    # 1. Reduce Contrast (Simulates poor exposure)
    contrast_factor = 0.6 
    img_float = img.astype(np.float32)
    img_float = img_float * contrast_factor + (128 * (1 - contrast_factor))
    
    # 2. Inject Gaussian Noise (mean=0, std=25)
    noise = np.random.normal(0, 25, img_float.shape)
    noisy_img = img_float + noise
    
    # 3. Clip back to standard 8-bit image range [0, 255]
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def main():
    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR}")
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Grab all PNGs in the input directory
    map_files = list(INPUT_DIR.glob("*.png"))
    total_files = len(map_files)
    
    print(f"Found {total_files} maps. Starting noise injection...")

    for i, file_path in enumerate(map_files, 1):
        img = cv2.imread(str(file_path))
        if img is None:
            continue

        # Apply noise and save to output directory
        noisy_img = apply_experimental_noise(img)
        output_path = OUTPUT_DIR / file_path.name
        cv2.imwrite(str(output_path), noisy_img)

        # Print progress every 500 images
        if i % 500 == 0 or i == total_files:
            print(f"Processed {i}/{total_files} maps...")

    print("Noise injection complete!")

if __name__ == "__main__":
    main()
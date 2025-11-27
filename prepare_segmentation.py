import openslide
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm


SLIDE_PATH = "CMU-1.svs"
OUTPUT_DIR = "segmentation_data"
IMG_DIR = f"{OUTPUT_DIR}/images"
MASK_DIR = f"{OUTPUT_DIR}/masks"
PATCH_SIZE = 256
NUM_PATCHES = 200 

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def generate_dataset():
    slide = openslide.OpenSlide(SLIDE_PATH)
    w, h = slide.dimensions
    print(f"Slide loaded: {w}x{h}")

    count = 0
    for y in range(h//4, h//4*3, PATCH_SIZE):
        for x in range(w//4, w//4*3, PATCH_SIZE):
            if count >= NUM_PATCHES: break
            
            # 1. Read Raw Image
            patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
            patch_np = np.array(patch)
            hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if np.mean(mask) > 50: 
                patch.save(f"{IMG_DIR}/patch_{count}.png")
                mask_img = Image.fromarray(mask)
                mask_img.save(f"{MASK_DIR}/patch_{count}.png")
                
                count += 1
        if count >= NUM_PATCHES: break
    
    print(f"✅ Generated {count} Image-Mask pairs in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()
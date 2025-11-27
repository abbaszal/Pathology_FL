import openslide
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import time
import os

SLIDE_PATH = "CMU-1.svs"   
OUTPUT_DIR = "local_features" 
PATCH_SIZE = 256
BATCH_SIZE = 64          
NUM_WORKERS = 4        


class WSIPatchDataset(Dataset):
    def __init__(self, slide_path, patch_size=256, transform=None):
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.transform = transform
        
        # Open slide
        self.slide = openslide.OpenSlide(slide_path)
        self.width, self.height = self.slide.dimensions
        self.coords = self._filter_tissue()
        print(f"Slide: {slide_path}")
        print(f"Dimensions: {self.width}x{self.height}")
        print(f"Total Patches Found (Tissue only): {len(self.coords)}")

    def _filter_tissue(self):
        coords = []
        thumb = self.slide.get_thumbnail((self.width // 100, self.height // 100)).convert('HSV')
        thumb_np = np.array(thumb)
        for y in range(0, self.height, self.patch_size):
            for x in range(0, self.width, self.patch_size):

                thumb_x = int(x / 100)
                thumb_y = int(y / 100)
                if thumb_y >= thumb_np.shape[0] or thumb_x >= thumb_np.shape[1]:
                    continue
                
                pixel = thumb_np[thumb_y, thumb_x]
                saturation = pixel[1]
        
                if saturation > 20: 
                    coords.append((x, y))
        return coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = self.slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert('RGB')
        
        if self.transform:
            patch = self.transform(patch)
        return patch


def get_model():

    full_model = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(full_model.children())[:-1])
    
    feature_extractor.eval()
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    print(f"Feature Extractor (2048-dim) loaded on {device}")
    return feature_extractor, device


def process_wsi(slide_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.basename(slide_path).replace(".svs", ".pt").replace(".tif", ".pt")
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"Skipping {slide_path}, already processed.")
        return

    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    dataset = WSIPatchDataset(slide_path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model, device = get_model()
    
    all_features = []
    
    print("Starting Feature Extraction...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            features = model(batch)
            features = features.flatten(start_dim=1)
            
            all_features.append(features.cpu())
            
    # Combine all batches
    if len(all_features) > 0:
        full_bag = torch.cat(all_features, dim=0)
        print(f"Extraction Complete. Shape: {full_bag.shape}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        # Save to disk
        torch.save(full_bag, save_path)
        print(f"Saved to {save_path}")
    else:
        print("No tissue found on this slide.")

if __name__ == "__main__":
    process_wsi(SLIDE_PATH)
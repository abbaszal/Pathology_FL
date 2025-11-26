import os
from datasets import load_dataset

dataset = load_dataset("owkin/camelyon16-features", split="Phikon_train")

print(f"Download complete. Dataset shape: {dataset.shape}")
print(f"Column names found: {dataset.column_names}")
save_path = "./real_data"
dataset.save_to_disk(save_path)

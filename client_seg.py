import flwr as fl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
from unet import SimpleUNet

# 1. Image Dataset Class
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, hospital_id, total_hospitals):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))
        chunk_size = len(self.files) // total_hospitals
        start = hospital_id * chunk_size
        end = start + chunk_size
        self.files = self.files[start:end]
        print(f"Hospital {hospital_id}: Training on {len(self.files)} images.")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("L") 
        
        img = self.transform(img)
        mask = self.transform(mask)
        mask = (mask > 0.5).float()
        
        return img, mask

# 2. Client
class SegClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.model = SimpleUNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        dataset = SegDataset("./segmentation_data/images", "./segmentation_data/masks", hospital_id, 2)
        self.loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        epoch_loss = 0.0
        for images, masks in self.loader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, masks)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return self.get_parameters(config={}), len(self.loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_sum = 0.0
        
        with torch.no_grad():
            for images, masks in self.loader:
                images, masks = images.to(self.device), masks.to(self.device)
                preds = self.model(images)
                loss_sum += self.criterion(preds, masks).item()
                
        return loss_sum / len(self.loader), len(self.loader), {}

if __name__ == "__main__":
    h_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SegClient(h_id))
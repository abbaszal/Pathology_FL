import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
import sys
import os


def load_hospital_data(hospital_id, total_hospitals=2):
    local_data = []
    if hospital_id == 0:
        print(f"Hospital {hospital_id}: Checking for LOCALLY processed raw WSI features...")
        feature_path = "./local_features/CMU-1.pt"
        
        if os.path.exists(feature_path):
            try:
                features = torch.load(feature_path)
                if features.dim() == 1: 
                    features = features.unsqueeze(0) 
                
                if features.shape[1] > 768:
                    features = features[:, :768]
                label = torch.tensor([0.0])
                
                local_data.append((features, label))
                print(f"✅ Hospital 0: Loaded 1 Raw Slide (CMU-1). Shape: {features.shape}")
                return local_data
            except Exception as e:
                print(f"❌ Hospital 0 Error loading local file: {e}")
        else:
            print(f"⚠️ Hospital 0: Local file {feature_path} not found. Using dummy data fallback.")

    print(f"Hospital {hospital_id}: Loading standard Camelyon16 dataset...")
    try:
        dataset = load_from_disk("./real_data")
        total_size = len(dataset)
        chunk_size = total_size // total_hospitals
        start = hospital_id * chunk_size
        end = start + chunk_size
        
        subset = dataset.select(range(start, end))
        print(f"Hospital {hospital_id}: Loading {len(subset)} slides from disk...")
        
        for item in subset:
            features = torch.tensor(item['features']).float()
            if features.dim() == 1:
                features = features.unsqueeze(0)
            if features.shape[1] > 768:
                features = features[:, :768]
                
            label = torch.tensor([float(item['label'])])
            local_data.append((features, label))
            
    except Exception as e:
        print(f"❌ Hospital {hospital_id} Error loading dataset: {e}")
        
    return local_data


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=768): 
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        A = self.attention(x)
        A = torch.softmax(A, dim=0)
        M = torch.mm(A.transpose(0, 1), x)
        return self.classifier(M)

class PathologyClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id
        self.model = AttentionMIL() 
        self.data = load_hospital_data(hospital_id)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
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
        count = 0
        
        for features, label in self.data:
            if features.size(0) == 0: continue 
            
            self.optimizer.zero_grad()
            pred = self.model(features)
            loss = self.criterion(pred.view(-1), label)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
            
        print(f"Hospital {self.hospital_id} finished training. Avg Loss: {epoch_loss/max(count, 1):.4f}")
        return self.get_parameters(config={}), len(self.data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval() 
        
        loss_sum = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, label in self.data:

                if features.dim() == 1: features = features.unsqueeze(0)
                if features.shape[1] > 768: features = features[:, :768]
                

                preds = self.model(features)
                loss = self.criterion(preds.view(-1), label)
                loss_sum += loss.item()
                
 
                predicted_label = (preds > 0.5).float()
                if predicted_label == label:
                    correct += 1
                total += 1

        if total == 0: return 0.0, 0, {"accuracy": 0.0}
        
        avg_loss = loss_sum / total
        accuracy = correct / total
        print(f"Hospital {self.hospital_id} Evaluation: Loss {avg_loss:.4f} | Acc {accuracy:.4f}")
        
        return avg_loss, total, {"accuracy": accuracy}

if __name__ == "__main__":
    h_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Starting Client for Hospital {h_id}...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PathologyClient(h_id))
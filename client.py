import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
import sys



# 1. Load Data logic
def load_hospital_data(hospital_id, total_hospitals=2):
    dataset = load_from_disk("./real_data")
    

    total_size = len(dataset)
    chunk_size = total_size // total_hospitals
    start = hospital_id * chunk_size
    end = start + chunk_size
    
    subset = dataset.select(range(start, end))
    
    local_data = []
    print(f"Hospital {hospital_id}: Loading {len(subset)} slides...")
    
    for item in subset:
        features = torch.tensor(item['features']).float()
        
        # FIX 1: Ensure it's 2D (Patches, Features)
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        # FIX 2: SLICE THE DATA
        # The data is 771 dim (768 features + 3 coords). We only want the first 768.
        if features.shape[1] > 768:
            features = features[:, :768]
            
        label = torch.tensor([float(item['label'])])
        local_data.append((features, label))
        
    return local_data


# 2. Model
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

        # x shape: [Num_Patches, Input_Dim]

        A = self.attention(x)
        A = torch.softmax(A, dim=0)
        M = torch.mm(A.transpose(0, 1), x)
        return self.classifier(M)



# 3. Client

class PathologyClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.model = AttentionMIL() # Phikon features are 768 dim
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

        # Train for 1 epoch for demo speed
        for features, label in self.data:
            if features.size(0) == 0: continue # Skip empty slides

            

            self.optimizer.zero_grad()
            pred = self.model(features)
            loss = self.criterion(pred.view(-1), label)
            loss.backward()
            self.optimizer.step()

            

        return self.get_parameters(config={}), len(self.data), {}



    def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            self.model.eval() # Switch to evaluation mode
            
            loss_sum = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, label in self.data:
                    # Same data fixing logic as training
                    if features.dim() == 1: features = features.unsqueeze(0)
                    if features.shape[1] > 768: features = features[:, :768]
                    
                    # Predict
                    preds = self.model(features)
                    loss = self.criterion(preds.view(-1), label)
                    loss_sum += loss.item()
                    
                    # Calculate accuracy
                    predicted_label = (preds > 0.5).float()
                    if predicted_label == label:
                        correct += 1
                    total += 1

            if total == 0: return 0.0, 0, {"accuracy": 0.0}
            
            avg_loss = loss_sum / total
            accuracy = correct / total
            
            return avg_loss, total, {"accuracy": accuracy}



if __name__ == "__main__":

    h_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PathologyClient(h_id))

# Federated Learning for Digital Pathology (Camelyon16)
  
This project implements a **Federated Learning** pipeline for detecting breast cancer metastasis in **Whole Slide Images (WSIs)**.  

---

##  Medical Data & Modality

| Attribute | Description |
|-----------|-------------|
| **Dataset** | Camelyon16 (Cancer Metastases in Lymph Nodes Challenge) |
| **Modality** | Histopathology *(H&E Stained Whole Slide Images)* |
| **Domain** | Digital Pathology – Breast Cancer Metastasis Detection |
| **Source** | Sentinel lymph node slides from breast cancer patients |
| **Data Scale** | Gigapixel WSIs (100,000+ pixels wide) |

To overcome the storage and memory constraints of gigapixel images, this project uses **Phikon (ViT-based)** precomputed feature embeddings, converting TB-scale WSIs into manageable **768-dimensional feature vectors**.

---

##  Technical Architecture

- **Federated Learning Orchestration:** Flower (FLWR)  
- **Deep Learning Framework:** PyTorch  
- **Model Architecture:** Attention-based Multiple Instance Learning (MIL)  
- **Data Handling:** Hugging Face Datasets (`owkin/camelyon16-features`)  
 

---


```mermaid
graph TD

    HF[Hugging Face Hub: owkin/camelyon16-features]
    DL_Script[download_.py]
    LocalDisk["./real_data - Phikon Embeddings"]

    Server[server.py - Aggregator]

    subgraph Hospital_0
        Client0[client.py 0]
        Loader0[Data Loader: 771 to 768 dim]
        Model0[AttentionMIL Model]
    end

    subgraph Hospital_1
        Client1[client.py 1]
        Loader1[Data Loader: 771 to 768 dim]
        Model1[AttentionMIL Model]
    end

    %% Phase 1: Setup & Data Prep
    HF -->|Downloads Features| DL_Script
    DL_Script -->|Saves to Disk| LocalDisk

    %% Phase 2: Federated Network
    LocalDisk -->|Reads Shard 0| Loader0
    LocalDisk -->|Reads Shard 1| Loader1

    Loader0 --> Model0
    Loader1 --> Model1

    Server <--> |REC: Global Params| Client0
    Server <--> |REC: Global Params| Client1

    Client0 -.-> |Send Updated Weights| Server
    Client1 -.-> |Send Updated Weights| Server
```


##  How to Run

### Install Dependencies
```bash
pip install torch torchvision flwr datasets huggingface_hub
````

---

### 1️⃣ Download & Prepare Data

```bash
python download_.py
```

---

### 2️⃣ Start the Federated Server

```bash
python server.py
```

---

### 3️⃣ Start Federated Clients (Hospitals) for example: 2 hospitals

```bash
# Terminal 2 - Hospital 0
python client.py 0

# Terminal 3 - Hospital 1
python client.py 1
```


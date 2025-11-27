# Federated Learning for Digital Pathology (Classification & Segmentation)

This project implements a comprehensive **Federated Learning (FL)** framework for **Digital Pathology**. It demonstrates a multi-task pipeline capable of handling **heterogeneous data sources** (Raw WSI processing vs. Pre-computed features) and performing both **Slide-Level Classification** and **Semantic Segmentation**.

---

## 🔬 Medical Data & Tasks

### 🧪 Task 1: WSI Classification (Metastasis Detection)
**Goal:** Detect breast cancer metastasis in lymph nodes  
**Method:** Attention-based Multiple Instance Learning (MIL)

**Data Handling (Heterogeneous):**
| Hospital | Type | Data Format | Processing |
|----------|------|-------------|------------|
| A (Research) | Raw WSI | .svs Gigapixel Slides | Custom ResNet50 feature extraction |
| B (Standard) | Pre-computed | Phikon ViT embeddings (768-dim) | Direct feature loading |

---

### 🧬 Task 2: Semantic Segmentation (Tissue Detection)
**Goal:** Pixel-wise segmentation of tissue vs. background  
**Method:** U-Net trained on 256×256 image patches  
**Data:** Synthetic ground truth masks generated using HSV thresholding

| Attribute | Description |
|-----------|-------------|
| Dataset | Camelyon16 & CMU-1 (OpenSlide Test Data) |
| Modality | Histopathology (H&E Stained Whole Slide Images) |
| Scale | Gigapixel WSIs (46,000+ pixels width) |
| Privacy | Data remains local; only model weights are shared via gRPC |

---

## 🏗 Technical Architecture

### 🔄 System Workflow (Mermaid Diagram)

```mermaid
graph TD
    classDef storage fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef script fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef server fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    subgraph "Phase 1: Data Engineering"
        RawWSI(("Raw Slide (.svs)\nCMU-1")):::storage
        HF_Hub(("HuggingFace Hub\nPhikon Features")):::storage
        
        script1(wsi_processor.py):::script
        script2(prepare_segmentation.py):::script
        script3(download_fixed.py):::script

        RawWSI -->|Extract Patches + ResNet50| script1
        RawWSI -->|Extract Images + Masks| script2
        HF_Hub -->|Download Embeddings| script3

        LocalFeats(("./local_features\n2048-dim Vectors")):::storage
        SegData(("./segmentation_data\nImages & Masks")):::storage
        StdFeats(("./real_data\nPhikon 768-dim")):::storage

        script1 --> LocalFeats
        script2 --> SegData
        script3 --> StdFeats
    end

    subgraph "Phase 2: Federated Network"
        Server(server.py\nFedAvg Aggregator):::server

        subgraph "Hospital 0 (Research Node)"
            Client0(client.py 0):::script
            Loader0[Loader: Raw Local Data]
            Slice0[Slice: 2048 -> 768 dim]
        end

        subgraph "Hospital 1 (Standard Node)"
            Client1(client.py 1):::script
            Loader1[Loader: Standard Dataset]
        end

        LocalFeats --> Loader0
        Loader0 --> Slice0 --> Client0
        StdFeats --> Loader1 --> Client1

        Server <==>|gRPC: Global Weights| Client0
        Server <==>|gRPC: Global Weights| Client1
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


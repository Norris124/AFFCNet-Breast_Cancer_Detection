
## README.md 

```markdown
# AFFC-Net: Adaptive Feature Fusion Convolutional-Graph Network for Histopathological Image Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3%2B-green)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the complete implementation of **AFFC-Net**, a dual-branch
architecture that integrates a ResNet-50 convolutional backbone with a superpixel-based
Graph Attention Network (GAT) for histopathological image classification. The two
feature streams are fused through a CBAM-style Adaptive Feature Fusion (AFF) module
before a GraphSAGE classifier produces the final prediction.

Additionally, this work introduces a **SUARA2-inspired two-step hierarchical graph
aggregation** strategy that reduces per-step neighbourhood complexity from O(N) to
O(√N), enabling efficient distributed multi-GPU training.

---

## Key Results

| Dataset   | Classes | Samples | Weighted F1 | Accuracy |
|-----------|---------|---------|-------------|----------|
| LC25000   | 5       | 25,000  | **97.44%**  | 97.44%   |
| BRACS     | 7       | 4,168   | **55.03%**  | 56.86%   |
| BreaKHis  | 2       | 7,909   | **89.23%**  | 89.22%   |

Multi-GPU SUARA2 (P=4) achieves **97.09% F1** on LC25000 with a theoretical
**1.35× communication speedup** at 120 GPUs.

---

## Architecture

```
Input Image (224×224×3)
        │
   ┌────┴────┐
   │         │
CNN Branch  GNN Branch
ResNet-50   SLIC Superpixels
(layer3)    → GAT (3 layers)
[B,196,1024]→ TopKPool
             [B,196,1024]
   │         │
   └────┬────┘
        │
   AFF Module
   (μ · CNN + (1−μ) · GNN)
   + CBAM Channel & Spatial Attention
        │
   GraphSAGE Classifier
   (49-node 7×7 grid graph)
        │
   Class Logits
```

---

## Repository Structure

```
AFFC-Net-Histopathology/
│
├── LC25000/
│   ├── 1_cnn_feature_extraction_lc25000.ipynb   # ResNet-50 → [N,196,1024]
│   ├── 2_gnn_feature_extraction_lc25000.ipynb   # SLIC+GAT → [N,196,1024]
│   └── 3_affcnet_training_lc25000.ipynb         # AFF + GraphSAGE training
│
├── BRACS/
│   ├── 5_bracs_cnn_feature_extraction.ipynb
│   ├── 6_bracs_gnn_feature_extraction.ipynb
│   └── 7_affcnet_training_bracs.ipynb
│
├── BreaKHis/
│   ├── 9_cnn_feature_extraction_breakhis.ipynb
│   ├── 10_gnn_feature_extraction_breakhis.ipynb
│   └── 11_affcnet_training_breakhis.ipynb
│
├── XAI/
│   ├── 12_XAI_BRACS.ipynb                       # Grad-CAM + GAT attention
│   ├── 13_XAI_LC25000.ipynb
│   └── 14_XAI_BreakHis.ipynb
│
├── SUARA/
│   ├── AFFC_SUARA_bracs_experiment.ipynb        # SUARA2 distributed training
│   ├── AFFC_SUARA2_breakhis_v5.ipynb
│   └── AFFC_SUARA2_lc25000_v5.ipynb
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AFFC-Net-Histopathology.git
cd AFFC-Net-Histopathology

# 2. Create and activate a virtual environment (recommended)
python -m venv affcnet_env
source affcnet_env/bin/activate        # Linux/Mac
affcnet_env\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch Geometric (match your CUDA version)
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f \
  https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## Datasets

Download and place datasets in the following locations before running notebooks:

| Dataset  | Source | Place at |
|----------|--------|----------|
| LC25000  | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) | `lung_colon_image_set/` |
| BRACS    | [BRACS Portal](https://www.bracs.icar.cnr.it/) | `BRACS_Dataset/BRACS_RoI/latest_version/` |
| BreaKHis | [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis) | `BreaKHis_v1/` |

> **Note:** Dataset folders are excluded from this repository via `.gitignore`
> due to file size. Only notebooks and code are tracked.

---

## Usage

Run notebooks in order within each dataset folder:

**Step 1 — CNN Feature Extraction**
```
1_cnn_feature_extraction_*.ipynb
```
Extracts ResNet-50 spatial features → saves `*_cnn_features_v2.pt`

**Step 2 — GNN Feature Extraction**
```
2_gnn_feature_extraction_*.ipynb
```
Builds SLIC superpixel graphs, trains GAT, extracts node features
→ saves `*_gnn_features_v2.pt` and `*_labels.pt`

**Step 3 — AFFC-Net Training**
```
3_affcnet_training_*.ipynb
```
Loads cached CNN + GNN features, trains AFF + GraphSAGE classifier
→ saves best model checkpoint and evaluation metrics

**Step 4 — Explainability (Optional)**
```
XAI/12_XAI_*.ipynb
```
Generates Grad-CAM saliency maps and GAT attention overlays

**Step 5 — Distributed Training with SUARA2 (Optional)**
```
SUARA/AFFC_SUARA2_*.ipynb
```
Runs multi-GPU experiments with hierarchical graph aggregation

---

## Experimental Results Detail

### LC25000 — 5-Class Classification

| Model Variant         | Precision | Recall  | F1      | Accuracy |
|-----------------------|-----------|---------|---------|----------|
| AFFC-Net w/o GNN      | 20.56%    | 20.51%  | 20.51%  | 20.51%   |
| AFFC-Net 3-Layer      | 97.36%    | 97.36%  | 97.36%  | 97.36%   |
| AFFC-Net + SUARA2     | 97.45%    | 97.44%  | 97.44%  | 97.44%   |
| SUARA2 P=4 (Best)     | 97.11%    | 97.09%  | 97.09%  | 97.09%   |

### BRACS — 7-Class Classification

| Model Variant         | F1      | Accuracy |
|-----------------------|---------|----------|
| AFFC-Net 3-Layer      | 55.03%  | 56.86%   |
| SUARA2 P=2 (Best)     | 57.76%  | 59.94%   |

### BreaKHis — Binary Classification

| Model Variant         | F1      | Accuracy |
|-----------------------|---------|----------|
| AFFC-Net 5-Layer      | 89.39%  | 89.39%   |
| AFFC-Net + SUARA2     | 89.23%  | 89.22%   |

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU       | NVIDIA H200 / A100 / V100 (≥16 GB VRAM recommended) |
| RAM       | ≥32 GB system RAM |
| Storage   | ≥50 GB (datasets + cached features) |
| CUDA      | 11.8 or 12.1 |

> All experiments were conducted on NVIDIA H200 GPUs (MIG 3g.71gb partitions).
> Single-GPU experiments run on any GPU with ≥16 GB VRAM.

---



---

**Quick tip:** Once your README is on GitHub, you can edit it directly in the browser by clicking the pencil icon on the file — no need to push from your local machine for small text changes.

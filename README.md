# TRIM: Temporal Risk Variance Minimization for OOD Generalization on Dynamic Graphs

This repository provides the official PyTorch implementation of:

**TRIM: Temporal Risk Variance Minimization for OOD Generalization on Dynamic Graphs**

*Liting Wang, Da Li, Zhiyun Lin*

> 📄 Paper link will be updated upon publication.

---

## 📌 Overview

TRIM is a framework for **out-of-distribution (OOD) generalization on dynamic graphs**. By minimizing the variance of temporal risks across time, TRIM learns time-invariant representations that generalize well to unseen distributions in evolving graph structures.

The repository supports both **link prediction** and **node classification** tasks on multiple dynamic graph benchmarks.

---

## ⚙️ Dependencies

- CUDA == 11.3
- Python >= 3.9
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.24.3
- SciPy >= 1.16.0
- scikit-learn >= 1.2.2
- tqdm >= 4.65.0

See [`requirements.txt`](./requirements.txt) for the full list.

---

## 🔧 Installation

We recommend using a fresh conda environment:

```bash
# Create and activate environment
conda create -n trim python=3.9 -y
conda activate trim

# Install PyTorch (with CUDA 11.3)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric and dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-geometric>=2.3.0

# Install remaining dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

Run the following command:

```bash
python main.py --dataset dataset_name
```

### Supported Datasets

**Link Prediction**

- `collab`
- `act`
- `synthetic` (0.4, 0.6, 0.8)

**Node Classification**

- `Aminer`
- `dymotif_data`

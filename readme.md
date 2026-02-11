# Word-Level Bangla Sign Language Recognition

An attention-based approach to Word-Level Bangla Sign Language (BdSL) Recognition using Graph Neural Networks and Spatial-Temporal Transformers.

## Overview

This project implements and benchmarks multiple deep learning architectures for skeleton-based sign language recognition on the **BdSLW60** dataset (60 word-level Bangla sign classes). We compare Graph Neural Network (GNN) approaches against the state-of-the-art **SLGTFormer** (Spatial-Local-Global Transformer) architecture.

## Architecture

The project explores the following models:

| Model | Type | Description |
|-------|------|-------------|
| **SLGTFormer** | Transformer | Spatial-Local-Global Transformer with LGRPE, TTSA, and PAF |
| **ST-GCN (Vanilla)** | GNN | Standard Spatial-Temporal Graph Convolution Network |
| **Attention GNN** | GNN | ST-GCN + Dilated TCN + Spatial/Temporal Attention |
| **Adaptive GNN** | GNN | Dynamic Adjacency Matrix Learning (CTR-GCN Style) + MS-TCN |
| **GNN + LSTM** | Hybrid | Graph Convolution + Bidirectional LSTM |
| **GNN + Transformer** | Hybrid | Graph Convolution + Transformer Encoder |
| **Pose LSTM** | Baseline | Raw Keypoint LSTM (No Graph Structure) |

## Results

### Video Dataset

| Model | Top-1 Acc (%) | Top-5 Acc (%) |
|-------|:---:|:---:|
| **SLGTFormer (Bone)** | **99.41** | **99.95** |
| SLGTFormer (Joint) | 95.49 | 99.46 |
| ST-GCN (Vanilla) | 94.15 | 99.19 |
| Adaptive GNN | 61.01 | 85.66 |
| Attention GNN | 58.22 | 86.90 |
| GNN + LSTM | 41.68 | 76.58 |
| Pose LSTM | 29.97 | 68.53 |
| GNN + Transformer | 24.17 | 58.97 |

### Image Dataset

| Model | Top-1 Acc (%) | Top-5 Acc (%) |
|-------|:---:|:---:|
| **Attention GNN** | **84.00** | **92.24** |
| SLGTFormer (Bone) | 77.94 | 91.08 |
| SLGTFormer (Joint) | 62.01 | 87.81 |
| GNN + Transformer | 13.48 | 40.16 |
| Pose LSTM | 12.25 | 36.96 |
| ST-GCN (Vanilla) | 11.37 | 35.13 |
| GNN + LSTM | 11.30 | 37.92 |

### Ablation Study (SLGTFormer Components)

| Configuration | Video Top-1 (%) | Image Top-1 (%) |
|---------------|:---:|:---:|
| Full SLGTFormer | 95.49 | 62.01 |
| w/o LGRPE | 95.33 | 62.01 |
| w/o TTSA | 13.10 | 56.36 |
| w/o PAF | 58.22 | 39.28 |

## Project Structure

```
├── config/                    # YAML configuration files for all experiments
├── data/                      # Dataset directory (not tracked)
│   ├── bdsl/                  # Video skeleton data
│   └── bdsl_img/              # Image skeleton data
├── feeders/                   # Data loading and augmentation
├── graph/                     # Graph structure definitions (27-keypoint skeleton)
├── model/                     # Model implementations
│   ├── twin_attention.py      # SLGTFormer (Main Model)
│   ├── grpe_attention.py      # LGRPE + PAF spatial attention
│   ├── attention.py           # MHSA / RPE-MHSA modules
│   ├── twins_attention_utils.py # TwinSVT (TTSA) module
│   ├── st_gcn_vanilla.py      # Vanilla ST-GCN baseline
│   ├── attention_gnn.py       # Attention GNN
│   ├── adaptive_gnn.py        # Adaptive GNN (CTR-GCN Style)
│   ├── gnn_lstm.py            # GNN + Bi-LSTM hybrid
│   ├── gnn_transformer.py     # GNN + Transformer hybrid
│   └── pose_lstm.py           # Pose LSTM baseline
├── preprocessing/             # Data preprocessing scripts
├── scripts/                   # Training and experiment runner scripts (.bat)
├── tools/                     # Utilities (monitoring, visualization, fusion)
├── main.py                    # Main training and evaluation entry point
├── experiments.yaml           # Full experiment registry
└── results_final.csv          # Experiment results log (not tracked)
```

## Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU

### Installation

```bash
# Clone the repository
git clone https://github.com/imranrimon/Word-Level-Bangla-Sign-Language-Recognition.git
cd Word-Level-Bangla-Sign-Language-Recognition

# Create conda environment
conda env create -f environment.yml
conda activate bdsl_graph

# Install dependencies
pip install timm einops scipy
```

### Data Preparation

1. Download the BdSLW60 dataset
2. Run preprocessing to extract skeleton keypoints:
   ```bash
   python preprocessing/preprocess_bdsl.py
   ```
3. For bone modality, generate bone data:
   ```bash
   python preprocessing/generate_bone_data.py
   ```

## Training

### Single Experiment
```bash
python main.py --config config/bdsl.yaml
```

### Full Experiment Suite
```bash
python tools/run_experiments.py --config experiments.yaml
```

### Ablation Study
```bash
scripts/run_ablations.bat
```

## Key Findings

1. **SLGTFormer dominates on video data** — The Bone modality variant achieves near-perfect 99.41% accuracy
2. **GNNs excel on static images** — Attention GNN (84%) outperforms SLGTFormer (62%) on single-frame recognition
3. **TTSA is critical** — Removing Two-Stream Temporal Attention causes catastrophic performance loss (95% → 13%)
4. **Simple GNNs > Complex GNNs** — Vanilla ST-GCN (94%) outperforms Adaptive/Attention GNNs (~61%) on video data

## Citation

```bibtex
@article{rimon2026bdsl,
  title={Word-Level Bangla Sign Language Recognition Using Attention Graph Neural Network},
  author={Rimon, Imran},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

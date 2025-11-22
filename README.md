<div align="center">

# LDCT-MOSNet: Medical Image Segmentation for BIBM 2025

**Author:** Zongqi Xu  
**Affiliation:** SouthWest Minzu University  
**Updated:** 2025/07/31  
**Status:** Manuscript under review (BIBM 2025)

</div>

---

## Overview

LDCT-MOSNet targets low-dose CT (LDCT) multi-organ segmentation for chronic obstructive pulmonary disease (COPD) population screening. The project investigates how lightweight DeepLab variants and custom loss designs perform under multi-center, low-dose, and hardware-diverse scenarios. We provide training/evaluation pipelines, comparison baselines, and utilities for reproducible experimentation on large-scale clinical datasets.

## Dataset

- **Source:** Multi-center COPD LDCT program run by a Class III Grade A hospital in Henan, China and its medical alliance (≥12 regional hospitals, covering rural, county, and tertiary centers).
- **Cohort:** 2,000 subjects with expert annotations covering 29 organ/tissue labels.  
  - 1,067 cases include complete scanner metadata (vendor, model, dose parameters).  
  - Ethics approval: *2024HL-040*.
- **Challenges:** Strong inter-center heterogeneity, dose bias, registration difficulty, class imbalance.
- **External Validation:** LIDC-IDRI (lung cancer lesions) and TCIA cohorts for cross-dataset generalization.

## Implemented Models

### Proposed / Improved Variants

- DeepLabV3+ with MobileNetV2 backbone  
- DSDeepLab (improved) + MobileNetV2  
- DenseASSP-DeeplabV3  
- Multi-Scale Hierarchical Feature Reuse DeepLabV3  
- DeepLabV3 with composite loss (dynamic edge awareness + multi-scale confidence weighting)

### Baseline Comparisons (GitHub references)

- AgileFormer
- MMCV-based baselines
- nnUNet
- ParaTransCNN
- SACNet
- Swin-Transformer segmentation
- U-Net / U-Net++
- ARU-Net

## Environment

| Component | Version / Requirement |
|-----------|----------------------|
| Python    | 3.8                  |
| CUDA      | 12.1                 |
| PyTorch   | 2.1.0                |
| Hardware  | 8 × NVIDIA RTX 3090 (24 GB VRAM each) |

> **Tip:** Adjust `requirements.txt` or conda environment file according to the backbone you plan to train; some comparison models may require additional dependencies (e.g., mmcv, timm, einops).

## Repository Structure

```
├── logs/                 # Training & inference logs
├── nets/                 # Model definitions (DeepLab variants, custom modules)
├── utils/                # Data loaders, metrics, schedulers, visualization helpers
├── segment_anything/     # SAM/SAM2 utilities (optional)
├── model_data/           # Pretrained weights & checkpoints
├── VOCdevkit/            # Dataset root (default structure, replace with your LDCT data)
├── train.py              # Main training script (switch configs for each model)
├── train_mininet.py      # MiniNet training script (rename from README: previously second train.py)
├── predict.py            # Inference / evaluation entry point
└── parameters_count.py   # Parameter counting utility
```

> Make sure to rename duplicated scripts appropriately (e.g., `train_mininet.py`) to avoid ambiguity when switching between backbones.

## Quick Start

1. **Prepare dataset**  
   - Organize LDCT volumes and labels under `VOCdevkit/` following Pascal VOC-style folder hierarchy, or adapt the data loader paths inside `utils/`.
   - For external validation, place LIDC-IDRI/TCIA data in separate folders and update the config.

2. **Install environment**
   ```bash
   conda create -n ldct-mosnet python=3.8
   conda activate ldct-mosnet
   pip install -r requirements.txt  # customize for specific baseline
   ```

3. **Train**
   ```bash
   python train.py \
     --model deeplabv3_plus_mobilenetv2 \
     --dataset VOCdevkit --num_classes 29 \
     --batch_size 16 --epochs 120 \
     --loss composite_edge_confidence
   ```

4. **Evaluate / Predict**
   ```bash
   python predict.py \
     --weights logs/best_model.pth \
     --input  VOCdevkit/val.txt \
     --save_dir outputs/
   ```

5. **Compare baselines**  
   - Integrate third-party repos (AgileFormer, nnUNet, etc.) under `nets/` or as submodules.  
   - Keep logs per model (`logs/<model_name>/`) for reproducibility and reporting.

## Results & Manuscript Status

- Extensive experiments across improved DeepLab variants and open-source baselines.  
- Cross-dataset validation on LIDC-IDRI and TCIA confirms strong generalization of LDCT-MOSNet under domain shift.  
- Full paper is under review for **IEEE BIBM 2025**; benchmark tables and ablation studies will be released upon acceptance.

## Citation

> Zongqi Xu, *LDCT-MOSNet: Lightweight DeepLab Variants with Composite Loss for Multi-Organ Segmentation in Low-Dose CT*, submitted to IEEE International Conference on Bioinformatics & Biomedicine (BIBM), 2025.

---

For collaboration or dataset access inquiries, please contact **Zongqi Xu (SouthWest Minzu University)**. Feel free to open issues or pull requests for bug fixes, new backbones, or training tricks.

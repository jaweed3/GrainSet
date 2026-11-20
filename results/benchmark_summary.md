# GrainSet Rice Quality Inspection - Benchmark Results

This repository contains experiments on cereal grain inspection using deep learning models for Edge AI deployment.

## 1. Baseline Model: SE-ResNet50 (Squeeze-and-Excitation ResNet-50)

- **Dataset**: Full Rice Dataset (31,000 images, 8 inspection categories)
  - Training: 24,767 images
  - Validation: 3,095 images
  - Testing: 3,100 images (un-augmented original test set)
- **Classes**:
  - `0_NOR`: Normal (2,000 test samples)
  - `1_F&S`: Fusarium & Shriveled (150 test samples)
  - `2_SD`: Sprouted (150 test samples)
  - `3_MY`: Moldy (150 test samples)
  - `4_AP`: Attacked by Pests (150 test samples)
  - `5_BN`: Broken (150 test samples)
  - `6_UN`: Unripe (150 test samples)
  - `7_IM`: Impurities (200 test samples)
- **Training Setup**:
  - Hardware: NVIDIA GeForce RTX 4060 (8GB VRAM)
  - Optimizer: Adam (`lr=1.2e-3`, `step` scheduler)
  - Batch Size: 64
  - Input Resolution: 224x224
  - Epochs: 50 (~117 minutes)
  - Tracking: Weights & Biases (W&B)

## 2. Evaluation Results on Test Set (3,100 Images)

| Metric | Our Baseline (SE-ResNet50) | Paper Benchmark (Fan et al., 2023) |
| :--- | :---: | :---: |
| **Accuracy** | **96.42%** | 95.70% |
| **Macro F1-Score** | **94.48%** | 91.90% |
| **Weighted F1-Score** | **96.59%** | 94.10% |

### Per-Class Detailed Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| 0_NOR (Normal) | 99.95% | 95.90% | 0.9788 | 2,000 |
| 1_F&S (Fusarium/Shriveled) | 97.35% | 98.00% | 0.9767 | 150 |
| 2_SD (Sprouted) | 69.81% | 98.67% | 0.8177 | 150 |
| 3_MY (Moldy) | 92.67% | 92.67% | 0.9267 | 150 |
| 4_AP (Attacked by Pests) | 91.93% | 98.67% | 0.9518 | 150 |
| 5_BN (Broken) | 97.95% | 95.33% | 0.9662 | 150 |
| 6_UN (Unripe) | 92.41% | 97.33% | 0.9481 | 150 |
| 7_IM (Impurities) | 98.52% | 100.00% | 0.9926 | 200 |
| **Overall / Macro Avg** | **92.57%** | **97.07%** | **94.48%** | **3,100** |
| **Weighted Avg** | **97.07%** | **96.42%** | **96.59%** | **3,100** |

## 3. Confusion Matrix

```
               Predicted:
               NOR   F&S    SD    MY    AP    BN    UN    IM   [Total]
NOR (True)  : 1918     0    56     7    11     1     7     0    (2000)
F&S (True)  :    0   147     2     0     0     0     1     0    (150)
SD  (True)  :    0     2   148     0     0     0     0     0    (150)
MY  (True)  :    0     1     5   139     0     1     4     0    (150)
AP  (True)  :    0     1     0     0   148     1     0     0    (150)
BN  (True)  :    0     0     0     3     1   143     0     3    (150)
UN  (True)  :    1     0     1     1     1     0   146     0    (150)
IM  (True)  :    0     0     0     0     0     0     0   200    (200)
```

## 4. Next Phase: Lightweight Models for Edge AI

- MobileNetV4 / MobileNetV3
- EfficientNet-B0 / EfficientNet-Lite
- Model quantization (INT8) and mobile deployment

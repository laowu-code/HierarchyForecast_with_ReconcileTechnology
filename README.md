
# MFFDM-WLS: Coherent Wind Speed Forecasting

This repository contains the official implementation of the **MFFDM-WLS** framework, a multi-granularity feature-based coherent forecasting method designed for temporal hierarchical wind speed time series.

## 📖 Overview

Wind speed forecasting at multiple time scales (e.g., 10 min, 30 min, 60 min) is critical for wind farm planning and grid stability. However, traditional methods often produce inconsistent forecasts across these different granularities.

**MFFDM-WLS** addresses this by:

1. 
**MFFDM**: A deep learning base forecaster that fuses features across granularities using innovative interaction modules.


2. 
**WLS Reconciliation**: A Weighted Least Squares technique that adjusts forecasts to ensure hierarchical consistency while improving accuracy, especially at coarser granularities.



---

## 🏗️ Methodological Framework

The project is divided into two major stages: **Base Forecasting** and **Forecast Reconciliation**.

### 1. Multi-granularity Feature Fusion-based Deep Model (MFFDM)

MFFDM serves as the base forecaster to capture complex patterns within and across time scales.

* 
**BUSA (Bottom-Up Self-Attention)**: Facilitates the transmission of fine-grained information to improve coarse-grained features.


* 
**TDAD (Top-Down Adaptive Decomposition)**: Allows coarse-grained trends to adaptively refine fine-grained forecasts.


* 
**Feature Enhancement**: Utilizes **SENet** (Squeeze-and-Excitation Network) and **Residual Blocks** to extract robust representations at each level.


* 
**Multi-Task Loss**: Supports deterministic and probabilistic forecasting using Quantile Loss (QL), Negative Log Gaussian Loss (NLGL), and Negative Log Laplace Loss (NLLL).



### 2. Weighted Least Squares (WLS) Reconciliation

To ensure that forecasts at different levels satisfy the "average-aggregation" constraint, we employ optimal reconciliation techniques.

* 
**Consistency**: Ensures the average of fine-grained forecasts equals the corresponding coarse-grained forecast.


* 
**Weighted Adjustment**: WLS accounts for error variances at different levels, significantly enhancing the accuracy of middle and top-level forecasts.


---

## ✨ Key Features

* 🚀 **Multi-Granularity Interaction**: Jointly learns features from multiple time scales rather than treating them independently.


* 📊 **Coherent Forecasts**: Guarantees hierarchical consistency for both deterministic (point) and probabilistic (interval) predictions.


* ⚖️ **Optimal Reconciliation**: Outperforms standard approaches like Bottom-Up (BU) or Ordinary Least Squares (OLS) by utilizing the WLS estimator.


## 📜 Citation

If you find this code or research helpful, please cite our paper:

```bibtex
@article{wang2025mffdm,
  title={MFFDM-WLS: A multi-granularity feature-based coherent forecasting method for temporal hierarchical wind speed time series},
  author={Wang, Yun and Duan, Xiaocong and Zhang, Fan and Wu, Guang and Zou, Runmin and Wan, Jie and Hu, Qinghua},
  journal={Applied Energy},
  volume={400},
  pages={126615},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.apenergy.2025.126615}
}
[cite_start]``` [cite: 391, 392, 396, 397]

```
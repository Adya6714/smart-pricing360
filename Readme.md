# Smart Product Pricing 360 - Complete Documentation

**Multimodal Price Prediction System**  
**Hackathon Submission 2025**

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Solution Overview](#2-solution-overview)
3. [Model Architecture](#3-model-architecture)
4. [Performance](#4-performance)
5. [Environment Setup](#5-environment-setup)
6. [Running the Code](#6-running-the-code)
7. [Output Files](#7-output-files)
8. [Hardware Requirements](#8-hardware-requirements)
9. [Troubleshooting](#9-troubleshooting)
10. [GCP Cloud Setup (Optional)](#10-gcp-cloud-setup-optional)
11. [Project Structure](#11-project-structure)

---

## 1. Executive Summary

Multimodal ensemble system combining text embeddings (e5-small-v2 with attention pooling), vision embeddings (CLIP with TTA), handcrafted features, and gradient boosting models to predict product prices. 

**Key Results:**
- **OOF SMAPE:** 49.77
- **Holdout SMAPE:** 50.25
- **Final Submission:** `outputs/test_preds/ensemble_improved.csv`

**Innovations:**
- Attention pooling for text (vs. mean pooling)
- Test-time augmentation for images  
- Fold-local feature scaling (prevents CV leakage)
- LightGBM-Tweedie for direct price modeling
- Two-stage ensemble optimization

---

## 2. Solution Overview

### Approach
- **Type:** Multimodal Ensemble (6 models)
- **Text:** e5-small-v2 with attention pooling (384-dim)
- **Image:** CLIP ViT-B/32 with test-time augmentation (512-dim)
- **Features:** 20 handcrafted features (quantities, text stats, brand indicators)
- **Models:** 3 neural heads + XGBoost + LightGBM + LightGBM-Tweedie

### Pipeline Variants

We provide 4 notebook variants for different use cases:

| Notebook | Description | Time | SMAPE |
|----------|-------------|------|-------|
| `01_mm_all.ipynb` | **Full pipeline** - Best performance | ~6-8 hrs | 49.77 |
| `01_mm_fast_v2.ipynb` | Fast variant - 3 folds, smaller model | ~2-3 hrs | ~52-54 |
| `02_mm_streamlined.ipynb` | Streamlined - Simplified architecture | ~4-5 hrs | ~51-53 |
| `03_mm_minimal.ipynb` | Minimal - LightGBM only | ~1-2 hrs | ~53-55 |

**Recommendation:** Use `01_mm_all.ipynb` for final submission.

---

## 3. Model Architecture

```
Text (e5-small) + Images (CLIP) + 20 Features
    ↓
[Text Head] [Image Head] [Fusion Head] [XGBoost] [LightGBM] [LGB-Tweedie]
    ↓
Weighted Ensemble (optimized via grid search)
    ↓
Final Predictions
```

### Individual Models

**1. Text Head**
- Input: e5-small-v2 embeddings (384-dim) + handcrafted features (20-dim)
- Architecture: 4-layer MLP with BatchNorm and Dropout
- Training: 5-fold CV, QuantileLoss, early stopping on SMAPE

**2. Image Head**  
- Input: CLIP ViT-B/32 image embeddings (512-dim)
- Architecture: 4-layer MLP with BatchNorm
- Enhancement: Test-time augmentation (original + horizontal flip)

**3. Fusion Head**
- Input: CLIP text (512-dim) + CLIP image (512-dim) + features (20-dim)
- Architecture: 5-layer deep MLP
- Hidden dimensions: 768 → 384 → 192 → 96 → 1

**4-6. Gradient Boosting**
- XGBoost: 2000 trees, log-price target
- LightGBM: 2000 trees, log-price target
- LightGBM-Tweedie: 4000 trees, raw price target (robust for skewed distributions)

### Training Details

- **Cross-Validation:** 5-fold stratified + 10% holdout
- **Loss Function:** Quantile loss (median regression, q=0.5)
- **Early Stopping:** SMAPE-based (patience 7-8)
- **Optimizer:** AdamW with Cosine annealing
- **Learning Rate:** 1e-3 for neural nets, 0.03-0.05 for GBMs
- **Regularization:** L2 weight decay, dropout (0.2-0.3), gradient clipping

### Feature Engineering (20 features)

**Package Indicators:**
- `has_pack`, `has_multipack`, `has_qty`, `pack_count`

**Weight/Volume:**
- `has_oz`, `has_lb`, `has_kg`, `has_ml`, `has_l`
- `content_mass_g`, `content_vol_ml`, `est_units`

**Numeric:**
- `num_count`, `max_number`, `min_number`, `sum_numbers`

**Text Statistics:**
- `text_len`, `word_count`, `avg_word_len`, `num_digits`

**Category:**
- `is_premium`, `is_budget`, `is_food`, `size_indicator`, `has_image`

---

## 4. Performance

### Cross-Validation Results

| Model | OOF SMAPE | Holdout SMAPE |
|-------|-----------|---------------|
| Text Head | 51.39 | 51.10 |
| Image Head | 63.48 | 72.90 |
| Multimodal Fusion | 52.91 | 54.94 |
| XGBoost | 53.55 | 53.78 |
| LightGBM | 54.02 | 54.34 |
| LightGBM Tweedie | 57.30 | 57.16 |
| **Ensemble (optimized)** | **49.77** | **50.25** |

### Ensemble Weights (Optimized)

After grid search with local refinement:
- Text Head: 52.4%
- Multimodal Fusion: 33.3%
- XGBoost: 14.3%
- Others: 0%

---

## 5. Environment Setup

### Prerequisites

- **Python:** 3.10+
- **Hardware:** 8GB+ RAM (16GB recommended)
- **Storage:** ~5GB for models and cache
- **Device:** Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU

### Installation Steps

#### Option 1: Using pip (Recommended)

```bash
# 1. Clone repository
cd smart_pricing

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas numpy scikit-learn xgboost lightgbm pillow tqdm python-dotenv

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from src.utils.device import setup_device; setup_device(verbose=True)"
```

#### Option 2: Using Poetry

```bash
# 1. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. Install dependencies
cd smart_pricing
poetry install

# 3. Activate environment
poetry shell
```

### Environment Configuration

The notebooks **auto-detect the best available device**:
- **CUDA** (GCP/cloud GPUs) - highest priority
- **MPS** (Apple Silicon) - fallback
- **CPU** - last resort

No manual configuration needed! The code automatically adjusts batch sizes and optimizations based on your hardware.

### Data Setup

```bash
# Place your data files
mkdir -p dataset
# Copy train.csv and test.csv to dataset/

# Verify data
ls -lh dataset/
# Should show: train.csv, test.csv
```

---

## 6. Running the Code

### Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate  # or: poetry shell

# 2. Start Jupyter
jupyter notebook

# 3. Open: notebooks/01_mm_all.ipynb

# 4. Run all cells sequentially (Cell 1 → Cell 12)
```

### Execution Flow

| Cell Range | Description | Time | Notes |
|------------|-------------|------|-------|
| **Cell 1-2** | Setup, device detection, data loading | ~1 min | Auto-detects GPU/MPS/CPU |
| **Cell 3-4** | Feature engineering, image mapping | ~2 min | Extracts 20 features |
| **Cell 5** | Text embeddings (e5-small-v2) | ~15 min | Cached after first run |
| **Cell 6** | Image embeddings (CLIP with TTA) | ~90 min | Cached after first run |
| **Cell 7-10** | Model training (5-fold CV) | ~3-4 hrs | Text, Image, Fusion, GBMs |
| **Cell 11** | Ensemble optimization | ~15 min | Grid search + refinement |
| **Cell 12** | Generate final predictions | ~1 min | Post-processing + save |

**Total Time:** ~6-8 hours on first run, ~4-5 hours on subsequent runs (embeddings cached)

### Running on Different Hardware

**Apple Silicon (M1/M2/M3):**
```bash
# Just run the notebook - automatically uses MPS
jupyter notebook notebooks/01_mm_all.ipynb
```

**NVIDIA GPU (Local or Cloud):**
```bash
# Install CUDA-compatible PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Run notebook - automatically uses CUDA
jupyter notebook notebooks/01_mm_all.ipynb
```

**CPU Only:**
```bash
# Just run - will be slower (~3-4x) but works
jupyter notebook notebooks/01_mm_all.ipynb
```

### Running as Python Script (Non-interactive)

```bash
# Convert notebook to script
jupyter nbconvert --to python notebooks/01_mm_all.ipynb

# Run in background
nohup python notebooks/01_mm_all.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

---

## 7. Output Files

### Final Submission File

**Location:** `outputs/test_preds/ensemble_improved.csv`

**Format:**
```csv
sample_id,price
100179,13.18
245611,13.30
146263,22.61
...
```

### Additional Output Files

```
outputs/
├── oof/                    # Out-of-fold predictions
│   ├── txt_improved.npy
│   ├── img_improved.npy
│   ├── mm_improved.npy
│   ├── xgb.npy
│   ├── lgb.npy
│   └── lgb_tweedie.npy
├── test_preds/             # Test predictions
│   ├── txt_improved.csv
│   ├── img_improved.csv
│   ├── mm_improved.csv
│   ├── xgb.csv
│   ├── lgb.csv
│   ├── lgb_tweedie.csv
│   ├── stack_ridge.csv
│   └── ensemble_improved.csv  ← **SUBMIT THIS**
└── reports/
    └── ensemble_improved.json  # Ensemble weights and scores
```

---

## 8. Hardware Requirements

### Minimum Requirements

- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 10GB free space
- **GPU:** Not required (but recommended)

### Recommended Setup

- **CPU:** 8+ cores
- **RAM:** 16GB+
- **Storage:** 20GB+ SSD
- **GPU:** 
  - Apple M1/M2/M3 (MPS)
  - NVIDIA T4/V100/A100 (CUDA)

### Performance Comparison

| Hardware | Full Training Time | Cost |
|----------|-------------------|------|
| **Apple M3** | 6-8 hours | Free (local) |
| **GCP T4 GPU** | 4-6 hours | $2-3 (GCP credits) |
| **GCP V100 GPU** | 2-3 hours | $6-8 (GCP credits) |
| **CPU Only** | 24-36 hours | Free (slow) |

---

## 9. Troubleshooting

### Common Issues

**Issue 1: `ModuleNotFoundError: No module named 'src'`**

**Solution:** The notebook auto-adds project root to Python path in Cell 1. Make sure you're running from the project directory.

```bash
cd /path/to/smart_pricing
jupyter notebook notebooks/01_mm_all.ipynb
```

---

**Issue 2: `RuntimeError: MPS doesn't support float64`**

**Solution:** Already handled! Notebook auto-converts all tensors to float32.

---

**Issue 3: Out of Memory**

**Solution:** Reduce batch sizes:

```python
# In training cells, modify:
bs=256  # instead of 512
batch_size=32  # instead of 64 for images
```

---

**Issue 4: Image download stalls**

**Solution:** Already handled! Notebook works with partial images. It proceeds with 72% train images available.

---

**Issue 5: `FileNotFoundError: dataset/train.csv`**

**Solution:** Make sure dataset files are in the correct location:

```bash
ls -l dataset/
# Should show: train.csv, test.csv
```

---

**Issue 6: CUDA not detected on GPU machine**

**Solution:**
```bash
# Verify GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Test
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 10. GCP Cloud Setup (Optional)

If you want faster training using Google Cloud Platform's $300 free credits:

### Quick GCP Setup

```bash
# 1. Install gcloud CLI
brew install --cask google-cloud-sdk  # macOS
# Or download from: https://cloud.google.com/sdk/docs/install

# 2. Initialize and authenticate
gcloud init
gcloud auth login

# 3. Create GPU instance (T4)
gcloud compute instances create smart-pricing-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --metadata="install-nvidia-driver=True"

# 4. SSH into instance
gcloud compute ssh smart-pricing-gpu --zone=us-central1-a

# 5. On GCP instance - clone your repo
git clone git@github.com:Adya6714/smart-pricing360.git
cd smart-pricing360

# 6. Install dependencies
pip install -r requirements.txt

# 7. Run training
python notebooks/01_mm_all.py  # or use Jupyter

# 8. Download results
exit  # back to local
gcloud compute scp smart-pricing-gpu:~/smart-pricing360/outputs/test_preds/ensemble_improved.csv . \
    --zone=us-central1-a

# 9. Stop instance to save credits
gcloud compute instances stop smart-pricing-gpu --zone=us-central1-a
```

### Cost Estimate

- **T4 GPU:** ~$0.45/hour
- **Full training:** 6 hours = ~$3
- **Storage (200GB):** ~$4/month
- **Total:** ~$7 out of $300 free credits

### Device Auto-Detection

The code automatically detects and uses the best available device:
- On GCP → Uses CUDA (GPU)
- On M3 Mac → Uses MPS (Apple Silicon)
- On CPU → Falls back to CPU

**No code changes needed when switching between environments!**

---

## 11. Project Structure

```
smart_pricing/
├── dataset/
│   ├── train.csv              # Training data (75,000 samples)
│   ├── test.csv               # Test data (75,000 samples)
│   ├── sample_test.csv        # Sample test input
│   └── sample_test_out.csv    # Sample expected output
│
├── notebooks/
│   ├── 01_mm_all.ipynb        # Full pipeline (RECOMMENDED)
│   ├── 01_mm_fast_v2.ipynb    # Fast variant (3-fold)
│   ├── 02_mm_streamlined.ipynb # Streamlined variant
│   └── 03_mm_minimal.ipynb    # Minimal variant (LightGBM only)
│
├── src/
│   ├── data/
│   │   └── load.py            # Data loading utilities
│   ├── features/
│   │   └── text_tabular.py    # Feature engineering
│   ├── training/
│   │   └── metrics.py         # SMAPE metric
│   └── utils/
│       ├── device.py          # Universal device detection
│       ├── seed.py            # Reproducibility
│       └── utils.py           # Image download utilities
│
├── outputs/                   # Generated during training
│   ├── oof/                   # Out-of-fold predictions (.npy)
│   ├── test_preds/            # Test predictions (.csv)
│   │   └── ensemble_improved.csv  ← **SUBMIT THIS**
│   └── reports/               # Ensemble reports (.json)
│
├── data/processed/            # Generated during training
│   ├── images/                # Downloaded product images
│   └── *_cache/               # Cached embeddings
│
├── .hf_cache/                 # Hugging Face model cache
├── .venv/                     # Python virtual environment
│
├── pyproject.toml             # Poetry dependencies
├── poetry.lock                # Locked dependencies
├── .gitignore                 # Git ignore rules
├── .python-version            # Python version (3.12.5)
└── Documentation_template.md  # This file
```

### Key Files

- **Main Pipeline:** `notebooks/01_mm_all.ipynb`
- **Final Output:** `outputs/test_preds/ensemble_improved.csv`
- **Dependencies:** `pyproject.toml` (Poetry) or install manually
- **Utilities:** `src/` folder (auto-imported by notebooks)

---

## 12. Reproducibility

### Seeding

All random operations are seeded for reproducibility:
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- Scikit-learn: `random_state=42`
- XGBoost/LightGBM: `random_state=42`

### Deterministic Results

The pipeline is fully deterministic. Running twice on the same hardware with the same data will produce identical results.

---

## 13. Contact & Support

**Repository:** https://github.com/Adya6714/smart-pricing360

**For issues:**
1. Check this documentation
2. Review notebook comments
3. Check `training.log` for errors
4. Verify device detection: `python -c "from src.utils.device import setup_device; setup_device(verbose=True)"`

---

## 14. Quick Reference

### Essential Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pandas numpy scikit-learn xgboost lightgbm pillow tqdm python-dotenv

# Run
jupyter notebook notebooks/01_mm_all.ipynb

# Output location
ls outputs/test_preds/ensemble_improved.csv

# Clean cache (if needed)
rm -rf .hf_cache/ data/processed/
```

### Time Estimates

- **Setup:** 5 minutes
- **First run:** 6-8 hours (embeddings cached)
- **Subsequent runs:** 4-5 hours
- **Fast variant:** 2-3 hours

---

## ✅ Submission Checklist

- [ ] Environment set up and dependencies installed
- [ ] Dataset files (`train.csv`, `test.csv`) in `dataset/` folder
- [ ] Ran `notebooks/01_mm_all.ipynb` successfully
- [ ] Output file generated: `outputs/test_preds/ensemble_improved.csv`
- [ ] Verified output format (2 columns: `sample_id`, `price`)
- [ ] Checked output has 75,000 rows (one per test sample)

**Final submission file:** `outputs/test_preds/ensemble_improved.csv`

---

**This solution is fully reproducible. Simply follow the setup instructions and run the notebook sequentially.**

**Good luck with your hackathon submission! 🚀**

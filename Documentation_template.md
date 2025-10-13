# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** October 2025

---

## 1. Executive Summary

Multimodal ensemble system combining text embeddings (e5-small-v2 with attention pooling), vision embeddings (CLIP with TTA), handcrafted features, and gradient boosting models to predict product prices. Achieves robust performance through fold-local scaling and two-stage ensemble optimization.

---

## 2. Solution Overview

### Approach
- **Type:** Multimodal Ensemble (6 models)
- **Text:** e5-small-v2 with attention pooling (384-dim)
- **Image:** CLIP ViT-B/32 with test-time augmentation (512-dim)
- **Features:** 20 handcrafted features (quantities, text stats, brand indicators)
- **Models:** 3 neural heads + XGBoost + LightGBM + LightGBM-Tweedie

### Key Innovations
1. Attention pooling for text (vs. mean pooling)
2. Test-time augmentation for images (original + horizontal flip)
3. Fold-local feature scaling (prevents CV leakage)
4. LightGBM-Tweedie for direct price modeling
5. Two-stage ensemble optimization (coarse + local refinement)

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

**Models:**
- **Text Head:** e5 embeddings + features → 4-layer MLP
- **Image Head:** CLIP image → 4-layer MLP  
- **Fusion:** CLIP text + image + features → 5-layer MLP
- **GBMs:** Full feature set (embeddings + handcrafted)

**Training:**
- 5-fold stratified CV + 10% holdout
- Loss: Quantile loss (median regression)
- Early stopping: SMAPE-based (patience 7-8)
- Optimizer: AdamW with Cosine annealing

---

## 4. Performance

**OOF SMAPE:** [Your score]  
**Holdout SMAPE:** [Your score]

---

## 5. Setup Instructions

### Quick Start

```bash
# 1. Clone and navigate to project
cd smart_pricing

# 2. Create environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers pandas numpy scikit-learn xgboost lightgbm pillow tqdm python-dotenv

# 3. Place data files
mkdir -p dataset
# Copy train.csv and test.csv to dataset/

# 4. Configure environment
cat > .env << EOF
DEVICE=mps
TRAIN_CSV=dataset/train.csv
TEST_CSV=dataset/test.csv
IMG_DIR=data/processed/images
OUT_DIR=outputs
HF_HOME=.hf_cache
EOF

# 5. Run notebook
jupyter notebook notebooks/01_mm_all.ipynb
```

### Running the Notebook

Execute cells sequentially (1-12):
- **Cells 1-2:** Setup and data loading
- **Cells 3-6:** Feature extraction and embeddings (~30 min)
- **Cells 7-10:** Model training (~2-3 hours)
- **Cells 11-12:** Ensemble optimization and final predictions (~15 min)

**Total Runtime:** ~4-5 hours on Apple M1/M2

### Output

Final submission: `outputs/test_preds/ensemble_improved.csv`

Format:
```csv
sample_id,price
100179,XX.XX
100180,XX.XX
```

---

## 6. Hardware Notes

- **Apple Silicon:** Set `DEVICE=mps`
- **NVIDIA GPU:** Set `DEVICE=cuda` (install CUDA-compatible PyTorch)
- **CPU Only:** Set `DEVICE=cpu` (10x slower)

### Memory Requirements
- **Minimum:** 8GB RAM
- **Recommended:** 16GB+ RAM
- **Storage:** ~5GB for models and cache

---

## 7. Troubleshooting

**ModuleNotFoundError: 'src'**
→ Notebook auto-adds project root to path (Cell 1)

**Out of memory**
→ Reduce batch sizes: `batch_size=64` for text, `batch_size=32` for images

**Image download stalls**
→ Already handled - notebook works with partial images (72% train available)

**MPS errors (Apple Silicon)**
→ Notebook auto-converts all tensors to float32

---

## 8. Key Files

- `notebooks/01_mm_all.ipynb` - Main pipeline
- `src/data/load.py` - Data loading with fold creation
- `src/training/metrics.py` - SMAPE metric
- `src/utils/seed.py` - Reproducible seeding
- `outputs/test_preds/ensemble_improved.csv` - **Final submission**

---

## Appendix: Feature List (20 total)

**Package:** `has_pack`, `has_multipack`, `has_qty`, `pack_count`  
**Weight/Volume:** `has_oz`, `has_lb`, `has_kg`, `has_ml`, `has_l`, `content_mass_g`, `content_vol_ml`, `est_units`  
**Numeric:** `num_count`, `max_number`, `min_number`, `sum_numbers`  
**Text:** `text_len`, `word_count`, `avg_word_len`, `num_digits`  
**Category:** `is_premium`, `is_budget`, `is_food`, `size_indicator`, `has_image`

---

**Note:** This solution is fully reproducible. Simply follow the setup instructions and run the notebook sequentially.

# Notebook Naming Conventions

This document outlines the naming conventions for outputs from each notebook.

## Overview

Each notebook has its own isolated output directories to prevent conflicts:

| Notebook | Output Directory | Cache Directory | Purpose |
|----------|-----------------|-----------------|---------|
| `01_mm_all.ipynb` | `outputs_01_mm_all/` | `data/processed/cache_01_mm_all_*/` | Full multimodal pipeline with all models |
| `01_mm_fast_v2.ipynb` | `outputs_01_mm_fast_v2/` | `data/processed/cache_01_mm_fast_v2/` | Fast version (3-fold, simpler models) |
| `02_mm_streamlined.ipynb` | `outputs_02_mm_streamlined/` | `data/processed/cache_02_mm_streamlined/` | Streamlined version with smart features |
| `03_mm_minimal.ipynb` | `outputs_03_mm_minimal/` | `data/processed/cache_03_mm_minimal/` | Minimal LightGBM-only version |

## Directory Structure

### 01_mm_all (Full Pipeline)

```
outputs_01_mm_all/
├── oof/                     # Out-of-fold predictions
│   ├── txt_improved.npy
│   ├── img_improved.npy
│   ├── mm_improved.npy
│   ├── xgb.npy
│   ├── lgb.npy
│   └── lgb_tweedie.npy
├── test_preds/              # Test set predictions
│   ├── txt_improved.csv
│   ├── img_improved.csv
│   ├── mm_improved.csv
│   ├── xgb.csv
│   ├── lgb.csv
│   ├── lgb_tweedie.csv
│   ├── stack_ridge.csv
│   └── ensemble_improved.csv  ← FINAL SUBMISSION
├── reports/                 # Metrics and ensemble weights
│   └── ensemble_improved.json
└── models/                  # Saved model checkpoints (if needed)

data/processed/
├── cache_01_mm_all_text/    # Text embeddings cache (e5-small-v2)
│   ├── e5_train_attn.npy
│   ├── e5_test_attn.npy
│   ├── e5_holdtr_attn.npy
│   └── e5_holdva_attn.npy
└── cache_01_mm_all_clip/    # CLIP embeddings cache
    ├── clip_txt_train.npy
    ├── clip_txt_test.npy
    ├── clip_img_train_tta.npy
    └── clip_img_test_tta.npy
```

### 01_mm_fast_v2 (Fast Version)

```
outputs_01_mm_fast_v2/
├── oof_v2/
│   └── fast_v2.npy
└── test_preds_v2/
    ├── fast_v2.csv
    └── blended_v2.csv  ← SUBMISSION

data/processed/cache_01_mm_fast_v2/
├── txt_train_v2.npy
├── txt_test_v2.npy
├── img_train_v2.npy
└── img_test_v2.npy
```

### 02_mm_streamlined (Streamlined Version)

```
outputs_02_mm_streamlined/
├── oof/
│   ├── text.npy
│   ├── image.npy
│   ├── fusion.npy
│   └── lgb.npy
└── test_preds/
    └── streamlined_ensemble.csv  ← SUBMISSION

data/processed/cache_02_mm_streamlined/
├── text_train.npy
├── text_test.npy
├── img_train.npy
└── img_test.npy
```

### 03_mm_minimal (Minimal Version)

```
outputs_03_mm_minimal/
└── submission_minimal.csv  ← SUBMISSION
```

## File Naming Rules

### Model Output Files

**OOF (Out-of-Fold) Predictions:**
- Format: `{model_name}.npy`
- Saved as numpy arrays for ensemble optimization

**Test Predictions:**
- Format: `{model_name}.csv`
- Always has columns: `sample_id`, `price`
- No index column

**Final Submissions:**
- `01_mm_all`: `ensemble_improved.csv`
- `01_mm_fast_v2`: `blended_v2.csv`
- `02_mm_streamlined`: `streamlined_ensemble.csv`
- `03_mm_minimal`: `submission_minimal.csv`

### Cache Files

**Text Embeddings:**
- Format: `{encoder_name}_{split}_*.npy`
- Examples: `e5_train_attn.npy`, `txt_train_v2.npy`

**Image Embeddings:**
- Format: `{model_name}_img_{split}_*.npy`
- Examples: `clip_img_train_tta.npy`, `img_train_v2.npy`

## Quick Setup

Run the provided script to create all directories:

```bash
python organize_outputs.py
```

This will create all required output and cache directories with proper naming.

## Benefits of This Structure

1. **No Conflicts:** Each notebook has isolated outputs
2. **Easy Comparison:** Can compare results across different approaches
3. **Cache Reuse:** Each notebook caches its own embeddings
4. **Clear Organization:** Easy to find specific notebook's results
5. **Version Control:** Can track which notebook produced which results

## Notes

- All notebooks share the same input data (`dataset/train.csv`, `dataset/test.csv`)
- All notebooks share the same image directory (`data/processed/images`)
- HuggingFace cache (`.hf_cache`) is shared across notebooks
- Each notebook maintains its own embeddings cache to avoid version conflicts


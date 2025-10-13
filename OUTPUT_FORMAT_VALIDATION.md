# Output Format Validation

## Input Files Structure

### `test.csv` / `sample_test.csv`
```
sample_id,catalog_content,image_link
100179,"Item Name: ...",https://m.media-amazon.com/images/I/71hoAn78AWL.jpg
217392,"Item Name: ...",https://m.media-amazon.com/images/I/91GB1wC6ObL.jpg
...
```

**Columns:**
- `sample_id`: Unique identifier for each test sample
- `catalog_content`: Product description and details
- `image_link`: URL to product image

## Output Files Structure

### `sample_test_out.csv` (Expected Format)
```
sample_id,price
217392,62.080007781501635
209156,17.189762720779267
262333,96.5014102539953
...
```

**Columns:**
- `sample_id`: Must match the test sample IDs exactly
- `price`: Predicted price as a float value

## How the Notebook Generates Predictions

### 1. Individual Model Predictions
Each model saves predictions in the correct format:

```python
pd.DataFrame({
    "sample_id": test_df.sample_id,
    "price": np.clip(predictions, 0.0, None)
}).to_csv(OUT_DIR/"test_preds"/"model_name.csv", index=False)
```

**Saved files:**
- `outputs/test_preds/txt_improved.csv` - Text model predictions
- `outputs/test_preds/img_improved.csv` - Image model predictions  
- `outputs/test_preds/mm_improved.csv` - Multimodal fusion predictions
- `outputs/test_preds/xgb.csv` - XGBoost predictions
- `outputs/test_preds/lgb.csv` - LightGBM predictions
- `outputs/test_preds/lgb_tweedie.csv` - LightGBM Tweedie predictions
- `outputs/test_preds/stack_ridge.csv` - Ridge stacker predictions

### 2. Final Ensemble Prediction
The final weighted ensemble is saved as:

```python
final = sub[["sample_id", "price"]].copy()
final.to_csv(OUT_DIR/"test_preds"/"ensemble_improved.csv", index=False)
```

**File:** `outputs/test_preds/ensemble_improved.csv`

## Validation Checklist

✅ **Sample ID Matching**: All output files use `test_df.sample_id` which comes directly from the test CSV, ensuring sample IDs match exactly

✅ **Column Names**: Both columns are named `sample_id` and `price` as required

✅ **No Index Column**: All CSV files use `index=False` to prevent adding an index column

✅ **Float Precision**: Prices are saved as float values (native pandas float format)

✅ **No Missing Values**: All test samples get predictions (handled by model inference on full test set)

✅ **Positive Values**: All predictions are clipped to >= 0 using `np.clip(predictions, 0.0, None)`

## How to Test with Sample Files

If you want to test with the sample files before running the full dataset:

```python
# At the top of the notebook, replace these lines:
TRAIN_CSV = str(project_root / "dataset/sample_test.csv")  # Use sample for testing
TEST_CSV = str(project_root / "dataset/sample_test.csv")

# Then the output will match sample_test_out.csv format exactly
```

## Key Points

1. **Order Preserved**: The predictions maintain the same order as the test CSV because we iterate through `test_df` directly

2. **All Samples Covered**: Every sample_id in test.csv will have a corresponding prediction

3. **Format Matches**: The output format exactly matches `sample_test_out.csv`:
   - 2 columns: `sample_id`, `price`
   - No header issues
   - No index column
   - Float precision for prices

4. **File Location**: Final submission is saved to:
   ```
   /Users/adyasrivastava/Downloads/smart_pricing/outputs/test_preds/ensemble_improved.csv
   ```



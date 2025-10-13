"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def read_csvs(train_path, test_path):
    """
    Read train and test CSV files and perform basic preprocessing
    
    Args:
        train_path (str): Path to train.csv
        test_path (str): Path to test.csv
    
    Returns:
        tuple: (train_df, test_df)
    """
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Basic info
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Check for required columns
    required_train_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
    required_test_cols = ['sample_id', 'catalog_content', 'image_link']
    
    for col in required_train_cols:
        if col not in train_df.columns:
            raise ValueError(f"Required column '{col}' not found in train data!")
    
    for col in required_test_cols:
        if col not in test_df.columns:
            raise ValueError(f"Required column '{col}' not found in test data!")
    
    # Add helper flags
    train_df['has_image'] = ~train_df['image_link'].isna()
    test_df['has_image'] = ~test_df['image_link'].isna()
    
    # Clean catalog_content
    train_df['catalog_content'] = train_df['catalog_content'].fillna("").astype(str)
    test_df['catalog_content'] = test_df['catalog_content'].fillna("").astype(str)
    
    # Handle missing prices in train (if any)
    if train_df['price'].isna().any():
        print(f"⚠️  Warning: {train_df['price'].isna().sum()} missing prices in train data!")
        train_df = train_df[~train_df['price'].isna()].reset_index(drop=True)
    
    # Price statistics
    print(f"\nPrice statistics:")
    print(f"  Min: ${train_df['price'].min():.2f}")
    print(f"  Max: ${train_df['price'].max():.2f}")
    print(f"  Mean: ${train_df['price'].mean():.2f}")
    print(f"  Median: ${train_df['price'].median():.2f}")
    print(f"\nImages available:")
    print(f"  Train: {train_df['has_image'].sum()}/{len(train_df)} ({100*train_df['has_image'].mean():.1f}%)")
    print(f"  Test: {test_df['has_image'].sum()}/{len(test_df)} ({100*test_df['has_image'].mean():.1f}%)")
    
    return train_df, test_df


def make_folds(df, n_folds=5, seed=42):
    """
    Create stratified K-folds based on log-transformed prices
    
    Args:
        df (pd.DataFrame): Training dataframe with 'price' column
        n_folds (int): Number of folds
        seed (int): Random seed
    
    Returns:
        pd.DataFrame: Dataframe with 'fold' column added
    """
    df = df.copy()
    df['fold'] = -1
    
    # Log transform prices for better stratification
    y_log = np.log1p(df['price'].clip(lower=0.0))
    
    # Create bins for stratification
    try:
        y_bins = pd.qcut(y_log, q=n_folds, labels=False, duplicates='drop')
    except ValueError:
        # If qcut fails, use cut instead
        y_bins = pd.cut(y_log, bins=n_folds, labels=False)
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_bins)):
        df.loc[val_idx, 'fold'] = fold
    
    # Verify folds
    print(f"\nCreated {n_folds} stratified folds:")
    for fold in range(n_folds):
        fold_df = df[df['fold'] == fold]
        print(f"  Fold {fold}: {len(fold_df)} samples, "
              f"mean price: ${fold_df['price'].mean():.2f}, "
              f"median: ${fold_df['price'].median():.2f}")
    
    return df


def extract_value_unit(catalog_content):
    """
    Extract numeric value and unit from catalog content
    
    Example: "Value: 10.5 Unit: Ounce" -> (10.5, "Ounce")
    
    Args:
        catalog_content (str): Product catalog text
    
    Returns:
        tuple: (value, unit) or (None, None) if not found
    """
    import re
    
    # Look for "Value: X.XX Unit: YYY" pattern
    value_match = re.search(r'Value:\s*([0-9.]+)', catalog_content)
    unit_match = re.search(r'Unit:\s*([A-Za-z\s]+)', catalog_content)
    
    value = float(value_match.group(1)) if value_match else None
    unit = unit_match.group(1).strip() if unit_match else None
    
    return value, unit


def add_basic_features(df):
    """
    Add basic engineered features to dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    df = df.copy()
    
    # Text length features
    df['text_len'] = df['catalog_content'].str.len()
    df['word_count'] = df['catalog_content'].str.split().str.len()
    
    # Extract value and unit
    df[['value', 'unit']] = df['catalog_content'].apply(
        lambda x: pd.Series(extract_value_unit(x))
    )
    
    # Count bullet points
    df['bullet_count'] = df['catalog_content'].str.count('Bullet Point')
    
    # Check for product description
    df['has_description'] = df['catalog_content'].str.contains('Product Description', case=False)
    
    return df

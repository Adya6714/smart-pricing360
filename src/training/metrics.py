"""
Metrics for price prediction
"""
import numpy as np


def smape_np(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    
    Formula:
    SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|)/2)
    
    Args:
        y_true (np.ndarray): Ground truth prices
        y_pred (np.ndarray): Predicted prices
    
    Returns:
        float: SMAPE value (percentage between 0-200, lower is better)
    """
    y_true = np.array(y_true).astype(np.float64)
    y_pred = np.array(y_pred).astype(np.float64)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape = np.mean(numerator / denominator) * 100.0
    
    return smape


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true).astype(np.float64)
    y_pred = np.array(y_pred).astype(np.float64)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

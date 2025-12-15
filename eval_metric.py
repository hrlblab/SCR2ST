import os
import numpy as np
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch


def calculate_pcc(pred, target):
    target = target.float()
    pred = pred.float()

    x = pred - pred.mean(dim=0, keepdim=True)
    y = target - target.mean(dim=0, keepdim=True)

    covariance = (x * y).sum(dim=0)
    bessel_corrected_variance_x = (x ** 2).sum(dim=0)
    bessel_corrected_variance_y = (y ** 2).sum(dim=0)

    pcc = covariance / torch.sqrt(bessel_corrected_variance_x * bessel_corrected_variance_y + 1e-8)
    return pcc.mean()


def calculate_pcc_per_row(gt, predict):
    """
    Calculate PCC for each row in each patch.

    Parameters:
    - gt: Ground truth data, numpy array of shape (num_patches, num_rows, num_columns).
    - predict: Predicted data, numpy array of shape (num_patches, num_rows, num_columns).

    Returns:
    - pcc_per_row: PCC for each row in each patch, numpy array of shape (num_patches, num_rows).
    """
    num_patches, num_rows, num_columns = gt.shape
    pcc_per_row = np.zeros((num_patches, num_rows))

    for patch_index in range(num_patches):
        for row_index in range(num_rows):
            pcc, _ = pearsonr(gt[patch_index, row_index, :], predict[patch_index, row_index, :])
            pcc_per_row[patch_index, row_index] = pcc

    return pcc_per_row


def calculate_overall_pcc(gt, predict):
    """
    Calculate overall PCC across all patches, rows, and columns.

    Parameters:
    - gt: Ground truth data, numpy array that can be flattened.
    - predict: Predicted data, numpy array that can be flattened.

    Returns:
    - overall_pcc: Overall Pearson correlation coefficient.
    """
    # Flatten the arrays to make them 1D
    flattened_gt = gt.flatten()
    flattened_predict = predict.flatten()

    # Calculate PCC
    overall_pcc, _ = pearsonr(flattened_gt, flattened_predict)

    return overall_pcc


def compute_pcc_per_sample(X, Y):
    """
    Compute Pearson Correlation Coefficient (PCC) for each sample.

    Args:
    - X: numpy array, shape (n_samples, n_features)
    - Y: numpy array, shape (n_samples, n_features)

    Returns:
    - pcc_values: numpy array, shape (n_samples,)
    """
    # Compute mean for each sample
    print(X.shape, Y.shape)
    X_mean = np.mean(X, axis=1, keepdims=True)  # shape: (n_samples, 1)
    Y_mean = np.mean(Y, axis=1, keepdims=True)  # shape: (n_samples, 1)

    # Center the data
    X_centered = X - X_mean  # shape: (n_samples, n_features)
    Y_centered = Y - Y_mean  # shape: (n_samples, n_features)

    # Compute numerator: covariance for each sample
    numerator = np.sum(X_centered * Y_centered, axis=1)  # shape: (n_samples,)

    # Compute denominator: product of standard deviations for each sample
    denominator = np.sqrt(np.sum(X_centered ** 2, axis=1) * np.sum(Y_centered ** 2, axis=1))  # shape: (n_samples,)

    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)

    # Compute PCC for each sample
    pcc_values = numerator / denominator

    return pcc_values



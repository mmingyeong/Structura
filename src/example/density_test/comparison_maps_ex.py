#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparison of density maps computed by FFTKDE and DensityCalculator

This script computes and compares the density maps obtained from two different methods
by applying several quantitative metrics and visualization techniques:
    - RMSE and MAE (voxel-wise error)
    - Pearson's correlation coefficient
    - Histograms and Cumulative Distribution Functions (CDFs)
    - Relative error map
    - Structural Similarity Index (SSIM) averaged over slices
    - Voxel-wise difference map (for a selected slice)
    
Before running, ensure that the two density maps are saved as "density_fft.npy" and "density_dc.npy",
and that they are defined on the same grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  # Requires scikit-image

def compute_rmse(a, b):
    """Compute the Root Mean Square Error between two arrays."""
    return np.sqrt(np.mean((a - b) ** 2))

def compute_mae(a, b):
    """Compute the Mean Absolute Error between two arrays."""
    return np.mean(np.abs(a - b))

def compute_pearson(a, b):
    """Compute the Pearson correlation coefficient between two arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    # np.corrcoef returns the correlation matrix
    return np.corrcoef(a_flat, b_flat)[0, 1]

def plot_histograms(a, b, bins=50):
    """Plot histograms of the two density maps for comparison."""
    plt.figure(figsize=(10, 5))
    plt.hist(a.flatten(), bins=bins, alpha=0.5, label="FFTKDE", color="blue")
    plt.hist(b.flatten(), bins=bins, alpha=0.5, label="DensityCalculator", color="orange")
    plt.xlabel("Density Value")
    plt.ylabel("Frequency")
    plt.title("Histogram Comparison")
    plt.legend()
    plt.show()

def plot_cdfs(a, b):
    """Plot the cumulative distribution functions (CDFs) of the two density maps."""
    a_flat = np.sort(a.flatten())
    b_flat = np.sort(b.flatten())
    cdf_a = np.arange(len(a_flat)) / float(len(a_flat))
    cdf_b = np.arange(len(b_flat)) / float(len(b_flat))
    
    plt.figure(figsize=(10, 5))
    plt.plot(a_flat, cdf_a, label="FFTKDE", color="blue")
    plt.plot(b_flat, cdf_b, label="DensityCalculator", color="orange")
    plt.xlabel("Density Value")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF Comparison")
    plt.legend()
    plt.show()

def compute_relative_error(a, b, epsilon=1e-10):
    """Compute the element-wise relative error between two arrays."""
    return np.abs(a - b) / (np.abs(b) + epsilon)

def compute_average_ssim(a, b):
    """
    Compute the average Structural Similarity Index (SSIM) for 3D data by applying SSIM on each 2D slice.
    Here we assume the third axis represents the 'z' direction.
    """
    ssim_vals = []
    for i in range(a.shape[2]):
        # ssim returns a value comparing the two 2D slices.
        s, _ = ssim(a[:, :, i], b[:, :, i], full=True)
        ssim_vals.append(s)
    return np.mean(ssim_vals)

def main():
    # Load density maps computed by FFTKDE and DensityCalculator.
    # These files must be pre-saved as "density_fft.npy" and "density_dc.npy".
    density_fft = np.load("density_fft.npy")
    density_dc = np.load("density_dc.npy")
    
    # Ensure the density maps have the same shape.
    if density_fft.shape != density_dc.shape:
        raise ValueError("The two density maps must have the same shape for a voxel-wise comparison.")
    
    # Compute RMSE and MAE.
    rmse_val = compute_rmse(density_fft, density_dc)
    mae_val = compute_mae(density_fft, density_dc)
    print(f"RMSE: {rmse_val:.6f}")
    print(f"MAE: {mae_val:.6f}")
    
    # Compute Pearson's correlation coefficient.
    pearson_val = compute_pearson(density_fft, density_dc)
    print(f"Pearson Correlation: {pearson_val:.6f}")
    
    # Plot histograms.
    plot_histograms(density_fft, density_dc, bins=50)
    
    # Plot cumulative distribution functions.
    plot_cdfs(density_fft, density_dc)
    
    # Compute relative error.
    rel_error = compute_relative_error(density_fft, density_dc)
    mean_rel_error = np.mean(rel_error)
    print(f"Mean Relative Error: {mean_rel_error:.6f}")
    
    # Compute average SSIM (averaged over 2D slices along z-axis).
    avg_ssim = compute_average_ssim(density_fft, density_dc)
    print(f"Average SSIM: {avg_ssim:.6f}")
    
    # Visualize voxel-wise difference map for the middle slice.
    mid_slice = density_fft.shape[2] // 2
    diff_map = density_fft[:, :, mid_slice] - density_dc[:, :, mid_slice]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(diff_map, cmap="bwr", interpolation="nearest")
    plt.colorbar(label="Difference")
    plt.title("Voxel-wise Difference Map (Middle Slice)")
    plt.xlabel("Y axis index")
    plt.ylabel("X axis index")
    plt.show()

if __name__ == '__main__':
    main()

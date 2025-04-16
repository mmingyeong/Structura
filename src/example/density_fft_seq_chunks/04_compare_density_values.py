#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: compare_density_values_full.py
#
# Description:
#   This script loads two 3D density maps computed by different methods (KDE-based and FFT-based)
#   from HDF5 files and compares their raw density values. It performs several types of analyses:
#
#    1) Computes statistical measures (mean, median, std, correlation, mean and std absolute difference)
#       between the two datasets.
#
#    2) Produces histograms of the KDE and FFT density distributions and the histogram of their differences.
#
#    3) Computes the ratio (KDE+1)/(FFT+1) and its logarithm (log10), and plots their histograms.
#
#    4) Generates a 2D hexbin scatter plot of log10-transformed density values from both datasets.
#
#    5) Compares the absolute mean density with a reference value (1816).
#
#    6) Computes and plots PDF diagrams of the density maps. In this version, the PDF diagram is computed
#       using the logarithm of density values to improve the visibility of regions near zero.
#
#   In addition, all computed statistical results are saved to a text file.
#   All figures are saved in PNG and PDF formats.
#
#   Density maps are assumed to be on the same grid with grid_spacing of 0.82 cMpc/h.

import os
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Append the parent directory (src) to the Python module search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from logger import logger  # 사용자 정의 로거 모듈
from kernel import KernelFunctions

def load_3d_data(hdf5_file, candidate_keys):
    """
    Load 3D density data from an HDF5 file using one of the candidate dataset keys.
    
    Parameters:
        hdf5_file (str): Path to the HDF5 file.
        candidate_keys (list): List of dataset keys to search for in the file.
    
    Returns:
        tuple: (density, box_size, grid_spacing)
               density (ndarray): 3D numpy array of the density data.
               box_size (float): Box size (in cMpc/h) from file attributes, default is 205.0.
               grid_spacing (float): Grid spacing (in cMpc/h) from file attributes, default is 0.82.
    """
    if not os.path.exists(hdf5_file):
        logger.error("HDF5 file does not exist: %s", hdf5_file)
        return None, None, None

    logger.info("Loading HDF5 file: %s", hdf5_file)
    with h5py.File(hdf5_file, "r") as f:
        density = None
        for key in candidate_keys:
            if key in f:
                density = f[key][:]
                logger.info("Dataset loaded with key: %s, shape=%s", key, density.shape)
                break
        if density is None:
            logger.error("No valid dataset found in %s. Tried keys: %s", hdf5_file, ", ".join(candidate_keys))
            return None, None, None

        box_size = f.attrs.get("box_size", 205.0)
        grid_spacing = f.attrs.get("grid_spacing", 0.82)
        logger.info("File attributes: box_size=%.2f, grid_spacing=%.2f", box_size, grid_spacing)
    return density, box_size, grid_spacing

def analyze_density_difference(kde_data, fft_data):
    """
    Analyze the difference between two 3D density datasets.
    
    Parameters:
        kde_data (ndarray): 3D numpy array for the KDE-based density map.
        fft_data (ndarray): 3D numpy array for the FFT-based density map.
    
    Returns:
        dict: A dictionary containing statistical measures:
              - mean_kde, mean_fft
              - median_kde, median_fft
              - std_kde, std_fft
              - correlation_coefficient between the two datasets
              - mean_absolute_difference and std_absolute_difference
    """
    flat_kde = kde_data.ravel()
    flat_fft = fft_data.ravel()
    
    stats = {
        "mean_kde": np.mean(flat_kde),
        "mean_fft": np.mean(flat_fft),
        "median_kde": np.median(flat_kde),
        "median_fft": np.median(flat_fft),
        "std_kde": np.std(flat_kde),
        "std_fft": np.std(flat_fft),
        "correlation_coefficient": np.corrcoef(flat_kde, flat_fft)[0, 1],
        "mean_absolute_difference": np.mean(np.abs(flat_kde - flat_fft)),
        "std_absolute_difference": np.std(np.abs(flat_kde - flat_fft))
    }
    return stats

###########################################################################
# UPDATED PLOT_HISTOGRAMS FUNCTION
###########################################################################
def plot_histograms(kde_data, fft_data, stats, results_dir):
    """
    Plot histograms of the density distributions for KDE and FFT data,
    and a histogram of the differences (KDE - FFT), with options to
    truncate outliers at the 99th percentile and use log scale for better visibility.

    Parameters:
        kde_data (ndarray): 3D density data from the KDE method.
        fft_data (ndarray): 3D density data from the FFT method.
        stats (dict): Statistical measures computed from the datasets.
        results_dir (str): Directory where the plot images will be saved.
    
    Returns:
        tuple: Paths to the saved plot images (PNG and PDF).
    """
    flat_kde = kde_data.ravel()
    flat_fft = fft_data.ravel()
    diff_data = flat_kde - flat_fft
    
    # 99th percentile cutoff to remove extreme outliers
    kde_99 = np.percentile(flat_kde, 99)
    fft_99 = np.percentile(flat_fft, 99)
    max_val = max(kde_99, fft_99)
    # 음수가 있을 수 있으면 아래와 같이 min_val도 잡을 수 있음 (필요시 조정).
    min_val = min(np.percentile(flat_kde, 1), np.percentile(flat_fft, 1), 0)
    
    # 차분 분포에서의 99th percentile (절댓값 기준)
    diff_99 = np.percentile(np.abs(diff_data), 99)
    diff_min = -diff_99
    diff_max = diff_99
    
    plt.figure(figsize=(12, 8))

    # -------------------------
    # (1) KDE와 FFT 분포 비교
    # -------------------------
    plt.subplot(2, 1, 1)
    bins = 100
    # log=True로 y축 로그 스케일 적용, range로 x축을 제한
    plt.hist(
        flat_kde, bins=bins, alpha=0.5, label="KDE Density",
        color='blue', range=(min_val, max_val), log=True
    )
    plt.hist(
        flat_fft, bins=bins, alpha=0.5, label="FFT Density",
        color='orange', range=(min_val, max_val), log=True
    )
    plt.xlabel("Density Value")
    plt.ylabel("Frequency (log scale)")
    plt.title("Density Distribution Comparison (clipped & log-scale)")
    plt.legend()

    # ---------------------------------
    # (2) KDE - FFT 분포 비교 (차분값)
    # ---------------------------------
    plt.subplot(2, 1, 2)
    plt.hist(
        diff_data, bins=bins, alpha=0.7, color='green',
        range=(diff_min, diff_max), log=True
    )
    plt.xlabel("Difference (KDE - FFT)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Histogram of Density Differences (clipped & log-scale)")
    
    # Embed statistical information
    stat_text = (
        f"Correlation Coefficient: {stats['correlation_coefficient']:.4f}\n"
        f"Mean KDE: {stats['mean_kde']:.4f}, Mean FFT: {stats['mean_fft']:.4f}\n"
        f"Median KDE: {stats['median_kde']:.4f}, Median FFT: {stats['median_fft']:.4f}\n"
        f"Std KDE: {stats['std_kde']:.4f}, Std FFT: {stats['std_fft']:.4f}\n"
        f"Mean Abs Diff: {stats['mean_absolute_difference']:.4f}, "
        f"Std Abs Diff: {stats['std_absolute_difference']:.4f}"
    )
    plt.gcf().text(0.75, 0.5, stat_text, bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(results_dir, f"{current_time_str}_density_histograms.png")
    pdf_path = os.path.join(results_dir, f"{current_time_str}_density_histograms.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    logger.info("Histogram plot saved as PNG: %s", png_path)
    logger.info("Histogram plot saved as PDF: %s", pdf_path)
    return png_path, pdf_path

def analyze_ratio(kde_data, fft_data):
    """
    Compute the ratio (KDE+1) / (FFT+1) and its logarithm (log10) from the density data.
    
    Parameters:
        kde_data (ndarray): 3D density data from the KDE method.
        fft_data (ndarray): 3D density data from the FFT method.
    
    Returns:
        tuple: (ratio, stats_ratio) where
            - ratio is a 1D array of computed ratios,
            - stats_ratio is a dictionary with mean and std of ratio and log10(ratio).
    """
    flat_kde = kde_data.ravel()
    flat_fft = fft_data.ravel()
    ratio = (flat_kde + 1) / (flat_fft + 1)
    log_ratio = np.log10(ratio)
    stats_ratio = {
        "mean_ratio": np.mean(ratio),
        "std_ratio": np.std(ratio),
        "mean_log_ratio": np.mean(log_ratio),
        "std_log_ratio": np.std(log_ratio)
    }
    return ratio, stats_ratio

def plot_ratio_histogram(ratio, stats_ratio, results_dir):
    """
    Plot histograms of the ratio (KDE+1)/(FFT+1) and its log10 transformation.
    
    Parameters:
        ratio (ndarray): 1D array of computed ratios.
        stats_ratio (dict): Dictionary with statistics for ratio and log10(ratio).
        results_dir (str): Directory to save the plots.
    
    Returns:
        tuple: Paths to the saved ratio histogram images (PNG and PDF).
    """
    plt.figure(figsize=(12, 5))
    
    # Ratio histogram
    plt.subplot(1, 2, 1)
    plt.hist(ratio, bins=100, color='gray', alpha=0.7)
    plt.xlabel("Ratio ((KDE+1)/(FFT+1))")
    plt.ylabel("Frequency")
    plt.title("Ratio Distribution")
    
    # log10(ratio) histogram
    plt.subplot(1, 2, 2)
    log_ratio = np.log10(ratio)
    plt.hist(log_ratio, bins=100, color='purple', alpha=0.7)
    plt.xlabel("log10(Ratio)")
    plt.ylabel("Frequency")
    plt.title("log10(Ratio) Distribution")
    
    stat_text = (
        f"Mean(Ratio): {stats_ratio['mean_ratio']:.4f}\n"
        f"Std(Ratio): {stats_ratio['std_ratio']:.4f}\n"
        f"Mean(log10(Ratio)): {stats_ratio['mean_log_ratio']:.4f}\n"
        f"Std(log10(Ratio)): {stats_ratio['std_log_ratio']:.4f}\n"
    )
    plt.gcf().text(0.75, 0.5, stat_text, bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(results_dir, f"{current_time_str}_ratio_hist.png")
    pdf_path = os.path.join(results_dir, f"{current_time_str}_ratio_hist.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    logger.info("Ratio histogram saved as PNG: %s", png_path)
    logger.info("Ratio histogram saved as PDF: %s", pdf_path)
    return png_path, pdf_path

def plot_hexbin_scatter(kde_data, fft_data, results_dir, max_points=500000):
    """
    Create a 2D hexbin scatter plot to compare log10-transformed density values from KDE and FFT data.
    
    Parameters:
        kde_data (ndarray): 3D density data from the KDE method.
        fft_data (ndarray): 3D density data from the FFT method.
        results_dir (str): Directory to save the plot.
        max_points (int): Maximum number of random points to sample for the plot.
    
    Returns:
        tuple: Paths to the saved hexbin plot images (PNG and PDF).
    """
    flat_kde = kde_data.ravel()
    flat_fft = fft_data.ravel()
    n_points = flat_kde.size
    
    if n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        flat_kde = flat_kde[indices]
        flat_fft = flat_fft[indices]
    
    x_vals = np.log10(flat_kde + 1)
    y_vals = np.log10(flat_fft + 1)
    
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x_vals, y_vals, gridsize=100, cmap='inferno', mincnt=1)
    cbar = plt.colorbar(hb)
    cbar.set_label("Bin count")
    plt.xlabel("log10(KDE + 1)")
    plt.ylabel("log10(FFT + 1)")
    plt.title("2D Hexbin: log10(KDE + 1) vs. log10(FFT + 1)")
    
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(results_dir, f"{current_time_str}_hexbin_kde_fft.png")
    pdf_path = os.path.join(results_dir, f"{current_time_str}_hexbin_kde_fft.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    logger.info("Hexbin plot saved as PNG: %s", png_path)
    logger.info("Hexbin plot saved as PDF: %s", pdf_path)
    return png_path, pdf_path

def compare_reference_density(kde_data, fft_data, ref_value=1816):
    """
    Compare the absolute mean density of each density map with a reference value.
    
    Parameters:
        kde_data (ndarray): 3D density data from the KDE method.
        fft_data (ndarray): 3D density data from the FFT method.
        ref_value (float): Reference absolute mean density value.
    
    Returns:
        tuple: (mean_abs_kde, mean_abs_fft)
    """
    flat_kde = kde_data.ravel()
    flat_fft = fft_data.ravel()
    mean_abs_kde = np.mean(np.abs(flat_kde))
    mean_abs_fft = np.mean(np.abs(flat_fft))
    diff_kde = mean_abs_kde - ref_value
    diff_fft = mean_abs_fft - ref_value
    print(f"Reference absolute mean density: {ref_value}")
    print(f"KDE absolute mean density: {mean_abs_kde:.2f} (Difference: {diff_kde:.2f})")
    print(f"FFT absolute mean density: {mean_abs_fft:.2f} (Difference: {diff_fft:.2f})")
    return mean_abs_kde, mean_abs_fft

def plot_pdf_diagrams(kde_data, fft_data, results_dir):
    """
    Compute normalized histograms (PDF diagrams) of the density map data and
    compare the PDF diagrams for KDE and FFT density maps.
    
    In this version, the PDF is computed from the logarithm of density values
    (i.e., log10(density + 1)) to improve the visibility when values are clustered near 0.
    
    Parameters:
        kde_data (ndarray): 3D density data from the KDE method.
        fft_data (ndarray): 3D density data from the FFT method.
        results_dir (str): Directory to save the PDF diagram plots.
    
    Returns:
        tuple: Paths to the saved PDF diagram plots (PNG and PDF).
    """
    flat_kde = kde_data.ravel()
    flat_fft = fft_data.ravel()
    bins = 100

    # Apply log10 transformation with offset to handle zero or near-zero densities.
    kde_log = np.log10(flat_kde + 1)
    fft_log = np.log10(flat_fft + 1)
    
    # Compute histograms (PDF) in the log-transformed space.
    hist_kde, bin_edges = np.histogram(kde_log, bins=bins, density=True)
    hist_fft, _ = np.histogram(fft_log, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, hist_kde, label="KDE PDF (log-scale)", color='blue', lw=2)
    plt.plot(bin_centers, hist_fft, label="FFT PDF (log-scale)", color='orange', lw=2)
    plt.xlabel("log10(Density + 1)")
    plt.ylabel("Probability Density")
    plt.title("PDF Diagram (log10 scale) Comparison")
    plt.legend()
    plt.tight_layout()
    
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(results_dir, f"{current_time_str}_pdf_diagram_log.png")
    pdf_path = os.path.join(results_dir, f"{current_time_str}_pdf_diagram_log.pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"PDF diagram saved as PNG: {png_path}")
    print(f"PDF diagram saved as PDF: {pdf_path}")
    return png_path, pdf_path

def save_stats_to_text(stats, mean_abs_kde, mean_abs_fft, ref_value, results_dir):
    """
    Save the density comparison statistics and reference comparison results to a text file.
    
    Parameters:
        stats (dict): Dictionary containing statistical measures from analyze_density_difference.
        mean_abs_kde (float): Mean absolute density for the KDE dataset.
        mean_abs_fft (float): Mean absolute density for the FFT dataset.
        ref_value (float): Reference density value.
        results_dir (str): Directory to save the text file.
    """
    text_file_path = os.path.join(results_dir, "density_comparison_stats.txt")
    
    lines = []
    lines.append("Density Comparison Statistics:\n")
    lines.append(f"mean_kde: {stats['mean_kde']:.4f}\n")
    lines.append(f"mean_fft: {stats['mean_fft']:.4f}\n")
    lines.append(f"median_kde: {stats['median_kde']:.4f}\n")
    lines.append(f"median_fft: {stats['median_fft']:.4f}\n")
    lines.append(f"std_kde: {stats['std_kde']:.4f}\n")
    lines.append(f"std_fft: {stats['std_fft']:.4f}\n")
    lines.append(f"correlation_coefficient: {stats['correlation_coefficient']:.4f}\n")
    lines.append(f"mean_absolute_difference: {stats['mean_absolute_difference']:.4f}\n")
    lines.append(f"std_absolute_difference: {stats['std_absolute_difference']:.4f}\n")
    lines.append("\n")
    
    diff_kde = mean_abs_kde - ref_value
    diff_fft = mean_abs_fft - ref_value
    lines.append("Reference density comparison:\n")
    lines.append(f"Reference absolute mean density: {ref_value}\n")
    lines.append(f"KDE absolute mean density: {mean_abs_kde:.2f} (Difference: {diff_kde:.2f})\n")
    lines.append(f"FFT absolute mean density: {mean_abs_fft:.2f} (Difference: {diff_fft:.2f})\n")
    
    with open(text_file_path, "w") as f:
        f.writelines(lines)
    
    logger.info("Density comparison statistics saved to text file: %s", text_file_path)
    print(f"Density comparison statistics saved to: {text_file_path}")

def main():
    """
    Main function to compare the raw density values of two 3D density maps
    (one from a KDE-based method and one from an FFT-based method) using various methods.
    
    Steps:
      1) Load the two density maps from HDF5 files.
      2) Compute statistical measures (mean, median, std, correlation, etc.).
      3) Plot histograms of the density distributions and their differences.
      4) Compute and plot the ratio (KDE+1)/(FFT+1) and its log10 version.
      5) Generate a 2D hexbin scatter plot of log10-transformed density values.
      6) Compare the absolute mean density with a reference value (1816).
      7) Compute and plot the PDF diagrams of the density maps using a log-transformed scale.
      8) Save all plots as PNG and PDF, log/print analysis results, and write statistics to a text file.
    """
    resolution = 0.82
    kernel = KernelFunctions.triangular

    start_time = time.time()
    
    # Set file paths
    kde_hdf5 = f"/home/users/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks/final_snapshot-99_kde_density_map_{kernel.__name__}_dx{resolution}.hdf5"
    fft_hdf5 = f"/home/users/mmingyeong/structura/Structura/src/example/density_fft_seq_chunks/final_snapshot-99_fft_density_map_{kernel.__name__}_dx{resolution}.hdf5"
    
    candidate_keys_kde = ["density", "kde_density", "density_map", "density_kde"]
    candidate_keys_fft = ["density", "fft_density", "density_map", "density_fft"]
    
    # Load density maps
    kde_data, kde_box, kde_spacing = load_3d_data(kde_hdf5, candidate_keys_kde)
    fft_data, fft_box, fft_spacing = load_3d_data(fft_hdf5, candidate_keys_fft)
    
    if kde_data is None or fft_data is None:
        logger.error("Failed to load one or both density datasets. Exiting.")
        return
    
    grid_spacing = kde_spacing if kde_spacing is not None else 0.82
    logger.info("Using grid_spacing = %.2f", grid_spacing)
    
    # Compute statistical measures
    stats = analyze_density_difference(kde_data, fft_data)
    logger.info("Density comparison statistics:")
    for key, value in stats.items():
        logger.info("%s: %.4f", key, value)
    print("Density comparison statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # Compare absolute mean density with reference value
    ref_density = 1816
    mean_abs_kde, mean_abs_fft = compare_reference_density(kde_data, fft_data, ref_value=ref_density)
    
    # Create results directory
    results_dir = os.path.join(os.getcwd(), "statics_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # (1) Save statistics to a text file
    save_stats_to_text(stats, mean_abs_kde, mean_abs_fft, ref_density, results_dir)
    
    # (2) Plot histograms of density distributions and their differences
    plot_histograms(kde_data, fft_data, stats, results_dir)
    
    # (3) Ratio analysis plots
    ratio, stats_ratio = analyze_ratio(kde_data, fft_data)
    logger.info("Ratio analysis statistics:")
    for key, value in stats_ratio.items():
        logger.info("%s: %.4f", key, value)
    print("Ratio analysis statistics:")
    for key, value in stats_ratio.items():
        print(f"{key}: {value:.4f}")
    plot_ratio_histogram(ratio, stats_ratio, results_dir)
    
    # (4) 2D Hexbin scatter plot
    plot_hexbin_scatter(kde_data, fft_data, results_dir, max_points=500000)
    
    # (5) PDF diagram comparison using log-transformed density values
    plot_pdf_diagrams(kde_data, fft_data, results_dir)
    
    elapsed_time = time.time() - start_time
    logger.info("Total execution time: %.2f seconds", elapsed_time)
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

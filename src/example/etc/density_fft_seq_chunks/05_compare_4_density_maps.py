#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-04-10
# @Filename: compare_4_density_pdf_only.py
#
# Description:
#   This script loads four 3D density maps (FFT and KDE, each in default and triangular versions),
#   computes their PDF diagrams (using log10-transformed density values), and overlays a text annotation 
#   indicating the computed average density values together with a reference average density (1816) for comparison.
#
#   The output is a single figure with the PDF diagrams of all four datasets and embedded text displaying the numerical averages.
#

import os
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 사용 중인 사용자 정의 로거 및 기타 모듈의 상대경로를 추가 (필요한 경우)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from logger import logger  # 사용자 정의 로거 모듈 (없으면 주석처리 가능)

def load_3d_data(hdf5_file, candidate_keys):
    """
    Load 3D density data from an HDF5 file using one of the candidate dataset keys.
    
    Parameters:
        hdf5_file (str): Path to the HDF5 file.
        candidate_keys (list): List of dataset keys to search for in the file.
    
    Returns:
        tuple: (density, box_size, grid_spacing)
               density (ndarray): 3D numpy array of the density data.
               box_size (float): Box size from file attributes (default 205.0).
               grid_spacing (float): Grid spacing from file attributes (default 0.82).
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

def main():
    """
    Main function to load 4 density maps, compute their PDF diagrams (after applying a log10 transformation
    to the density values), and display the average density of each dataset together with a reference value (1816)
    in the figure as text.
    """
    start_time = time.time()
    
    # Define file paths for each density map
    fft_default_path = "/home/users/mmingyeong/structura/Structura/src/example/density_fft_seq_chunks/final_snapshot-99_fft_density_map.hdf5"
    fft_tri_path     = "/home/users/mmingyeong/structura/Structura/src/example/density_fft_seq_chunks/final_snapshot-99_fft_density_map_triangular_dx0.82.hdf5"
    kde_default_path = "/home/users/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks/final_snapshot-99_kde_density_map.hdf5"
    kde_tri_path     = "/home/users/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks/final_snapshot-99_kde_density_map_triangular_dx0.82.hdf5"
    
    candidate_keys = ["density", "kde_density", "fft_density", "density_map"]
    
    # Load datasets
    fft_default, _, _ = load_3d_data(fft_default_path, candidate_keys)
    fft_tri, _, _     = load_3d_data(fft_tri_path, candidate_keys)
    kde_default, _, _ = load_3d_data(kde_default_path, candidate_keys)
    kde_tri, _, _     = load_3d_data(kde_tri_path, candidate_keys)
    
    # Check if any dataset is missing using identity checks
    if any(x is None for x in [fft_default, fft_tri, kde_default, kde_tri]):
        logger.error("One or more density datasets failed to load. Exiting.")
        return

    # Flatten the 3D data into 1D arrays for histogram and mean calculations
    flat_fft_default = fft_default.ravel()
    flat_fft_tri     = fft_tri.ravel()
    flat_kde_default = kde_default.ravel()
    flat_kde_tri     = kde_tri.ravel()
    
    # Compute mean densities for each dataset
    mean_fft_default = np.mean(flat_fft_default)
    mean_fft_tri     = np.mean(flat_fft_tri)
    mean_kde_default = np.mean(flat_kde_default)
    mean_kde_tri     = np.mean(flat_kde_tri)
    
    # Set a reference density value for comparison
    ref_density = 1816
    
    # Prepare PDF diagram using log10 transformation (log10(density + 1))
    # Combine all data to obtain common histogram bin edges
    all_data = np.concatenate([flat_fft_default, flat_fft_tri, flat_kde_default, flat_kde_tri])
    log_all = np.log10(all_data + 1)
    bins = 100
    edges = np.histogram_bin_edges(log_all, bins=bins)
    
    # Compute histograms (PDFs) with density=True to yield a probability density
    hist_fft_default, _ = np.histogram(np.log10(flat_fft_default+1), bins=edges, density=True)
    hist_fft_tri, _     = np.histogram(np.log10(flat_fft_tri+1),     bins=edges, density=True)
    hist_kde_default, _ = np.histogram(np.log10(flat_kde_default+1), bins=edges, density=True)
    hist_kde_tri, _     = np.histogram(np.log10(flat_kde_tri+1),     bins=edges, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Create a figure for the PDF diagram only.
    plt.figure(figsize=(12, 8))
    plt.plot(bin_centers, hist_fft_default, label="FFT", lw=2, color='blue')
    plt.plot(bin_centers, hist_fft_tri,     label="FFT (triangular)", lw=2, color='cyan')
    plt.plot(bin_centers, hist_kde_default, label="KDE", lw=2, color='orange')
    plt.plot(bin_centers, hist_kde_tri,     label="KDE (triangular)", lw=2, color='red')
    plt.xlabel("log10(Density + 1)")
    plt.ylabel("Probability Density")
    plt.title("PDF Diagram Comparison for 4 Density Maps")
    plt.legend()
    
    # Create text annotation with the computed mean densities and reference density.
    textstr = (
        f"Mean Densities:\n"
        f"FFT: {mean_fft_default:.2f}\n"
        f"FFT (triangular): {mean_fft_tri:.2f}\n"
        f"KDE: {mean_kde_default:.2f}\n"
        f"KDE (triangular): {mean_kde_tri:.2f}\n\n"
        f"Reference Mean Density: {ref_density}"
    )
    
    # Place the text annotation inside the figure; adjust position as needed.
    plt.gcf().text(0.68, 0.5, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Create results directory if it does not exist
    results_dir = os.path.join(os.getcwd(), "statics_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the resulting figure as PNG and PDF.
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path_png = os.path.join(results_dir, f"{current_time_str}_pdf_diagram_only.png")
    figure_path_pdf = os.path.join(results_dir, f"{current_time_str}_pdf_diagram_only.pdf")
    plt.savefig(figure_path_png, bbox_inches="tight")
    plt.savefig(figure_path_pdf, bbox_inches="tight")
    plt.show()
    
    elapsed_time = time.time() - start_time
    logger.info("Total execution time: %.2f seconds", elapsed_time)
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-04-09
# @Filename: 03_visualize_kde_hdf5_snapshot99.py
#
# Description:
#   This script loads a 3D KDE density map from an HDF5 file, computes a 2D projection by summing
#   along the 0th axis, applies a log10 transformation to enhance dynamic range, and uses the Visualizer class
#   functions to generate and save the image plot in PNG and PDF formats.
#   The axes are labeled as x, y, z in physical units (cMpc/h) with grid spacing 0.82, and the total execution
#   time is logged and printed.

import os
import sys
import time
import h5py
import numpy as np
from datetime import datetime

# Append the parent directory (src) to the Python module search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Define a results directory for image outputs.
IMG_RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(IMG_RESULTS_DIR, exist_ok=True)

from logger import logger
from kernel import KernelFunctions
from visualization import Visualizer

def main():
    """
    Main function to load a 3D KDE density map from an HDF5 file,
    compute its 2D projection by summing along the 0th axis, apply a log10 transformation,
    and generate/save the resulting image plot using the Visualizer class.
    
    The labels are set assuming axes 0,1,2 correspond to x, y, z with units cMpc/h,
    grid spacing is 0.82, and projection is done along axis 0.
    
    Returns:
        None
    """
    res=0.41
    kernel=KernelFunctions.uniform

    start_time = time.time()
    
    # HDF5 file path (3D KDE density map)
#    ==========================================================================================================================================
    hdf5_path = f"/home/users/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks/final_snapshot-99_kde_density_map_{kernel.__name__}_dx{res}.hdf5"
#    ==========================================================================================================================================
    if not os.path.exists(hdf5_path):
        logger.error("HDF5 file not found: %s", hdf5_path)
        return
    
    # Candidate keys for the 3D density dataset.
    candidate_keys = ["density", "kde_density", "density_map", "density_kde"]
    
    # Load the 3D density data from the HDF5 file.
    logger.info("Opening HDF5 file: %s", hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        density = None
        for key in candidate_keys:
            if key in f:
                density = f[key][:]
                logger.info("Dataset found with key: %s", key)
                break
        if density is None:
            logger.error("None of the candidate keys found in HDF5 file: %s", ", ".join(candidate_keys))
            return
        # Get grid spacing and box size from attributes; use default values if not available.
        box_size = f.attrs.get("box_size", 205.0)   # in cMpc/h
        grid_spacing = f.attrs.get("grid_spacing", 0.82)  # in cMpc/h
        logger.info("Data shape: %s, box_size: %.2f, grid_spacing: %.2f", density.shape, box_size, grid_spacing)
    
    # Projection: Sum the 3D density map along axis 0 (x-axis) to obtain a 2D density map.
    density_2d = np.sum(density, axis=0)
    logger.info("Computed 2D projection (summed over axis 0); resulting shape: %s", density_2d.shape)
    
    # Apply log10 transformation for visualization (avoid log(0) by adding 1).
    hist = np.log10(density_2d + 1)
    
    # Compute bin edges for the 2D map using grid_spacing.
    # For projection along axis 0, the remaining axes are y and z.
    ny, nz = density_2d.shape
    edges1 = np.linspace(0, ny * grid_spacing, ny + 1)  # y-axis
    edges2 = np.linspace(0, nz * grid_spacing, nz + 1)  # z-axis
    
    # Set parameters for plotting.
    # Axes: with projection along axis 0, x-axis is removed; remaining are y and z.
    xlabel = "Y (cMpc/h)"
    ylabel = "Z (cMpc/h)"
    title = f"KDE Projection (log10) along Axis 0 {kernel.__name__}, {res}"
    
    # For annotation purposes, define x_range, x_center, etc.
    # Here, since projection is over full axis 0, we use the full box_size.
    x_min = 0.0
    x_max = box_size
    x_center = box_size / 2.0
    x_range = box_size  # Thickness of the projection (full x-dimension)
    sampling_rate = 1  # No down-sampling
    bins = ny  # Number of bins equals the cell count along the projected axis (should be 250)
    
    # input_folder is directory of the HDF5 file.
    input_folder = os.path.dirname(hdf5_path)
    
    # Initialize Visualizer instance (using GPU acceleration as specified).
    viz = Visualizer(use_gpu=True)
    
    try:
        # Use Visualizer.create_image_plot to generate and save the image.
        saved_files = viz.create_image_plot(
            hist=hist,
            edges1=edges1,
            edges2=edges2,
            results_folder=IMG_RESULTS_DIR,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            projection_axis=0,
            x_range=x_range,
            x_center=x_center,
            sampling_rate=sampling_rate,
            x_min=x_min,
            x_max=x_max,
            input_folder=input_folder,
            results_dir=IMG_RESULTS_DIR,
            bins=bins,
            box_size=box_size,
            scale="log10",
            cmap="cividis"
        )
        
        for fmt, path in saved_files.items():
            logger.info("%s file saved: %s", fmt, path)
        logger.info("Visualization processing completed successfully.")
    
    except Exception as e:
        logger.error("An error occurred during visualization: %s", e)
    
    # Calculate and log the total execution time.
    elapsed_time = time.time() - start_time
    logger.info("Total execution time: %.2f seconds", elapsed_time)
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

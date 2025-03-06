#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: visualization_ex.py

import sys
import os
import time  # For execution time measurement

# Append the parent directory (src) to the Python module search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization import Visualizer
from data_loader import DataLoader
from config import RESULTS_DIR, OUTPUT_DATA_PATHS
from logger import logger

start_time = time.time()  # Record the start time

# ------------------------------------------------------------------
# Configuration: The initial condition data are provided in ckpc/h units.
# The plot will utilize a thickness of 10 cMpc/h, i.e., the filtering range
# is set from 100 to 110 cMpc/h.
# Note: 1 cMpc/h is equivalent to 1000 ckpc/h.
# ------------------------------------------------------------------
npy_folder = OUTPUT_DATA_PATHS["TNG300_ICS"]
logger.info(f"Input directory (npy): {npy_folder}")
logger.info(f"Output directory: {RESULTS_DIR}")

# Verify the existence of .npy files in the input directory.
if os.path.exists(npy_folder) and os.path.isdir(npy_folder):
    file_list = os.listdir(npy_folder)
    npy_files = [f for f in file_list if f.endswith(".npy")]

    if npy_files:
        logger.debug(f"Detected {len(npy_files)} .npy files in the input directory.")
        logger.debug(f"Example file: {npy_files[0]}")
    else:
        logger.warning("No .npy files found in the input directory!")
else:
    logger.error(f"Input directory does not exist: {npy_folder}")

# ------------------------------------------------------------------
# Filtering range configuration: Convert 100 ~ 110 cMpc/h to ckpc/h.
# ------------------------------------------------------------------
x_min = 100 * 1000  # 100 cMpc/h -> 100000 ckpc/h
x_max = 110 * 1000  # 110 cMpc/h -> 110000 ckpc/h
x_thickness = x_max - x_min  # Thickness of 10 cMpc/h (i.e., 10000 ckpc/h)
x_center = (x_min + x_max) / 2  # Center of the range
logger.debug(f"Filtering range configured: x_min={x_min}, x_max={x_max}, x_thickness={x_thickness}, x_center={x_center}")

# Load data using DataLoader.
logger.info("Loading data using DataLoader...")
loader = DataLoader(npy_folder)
sampling_rate = 1  # Utilize the entire dataset (sampling rate = 1)
positions = loader.load_all_chunks(x_min=x_min, x_max=x_max, sampling_rate=sampling_rate)
logger.debug(f"Dataset shape: {positions.shape}")
logger.info("Data loaded successfully.")

# ------------------------------------------------------------------
# Visualization: Compute 2D histogram and generate an image plot.
# ------------------------------------------------------------------
logger.info("Computing 2D histogram and generating image plot...")
viz = Visualizer(use_gpu=True)  # Utilize GPU acceleration

# When the 'bins' parameter is omitted, the optimal bin number is computed automatically.
# bins = 205  # for 1.00 cMpc/h resolution
bins = 410  # for 0.5 cMpc/h resolution
# List of scale transformations to be applied.
scales = ["log10", "log2", "ln", "sqrt", "linear"]

# For demonstration purposes, use the first scale in the list.
scale = scales[0]
try:
    # Compute the 2D histogram.
    hist, edges1, edges2 = viz.compute_2d_histogram(positions, bins=bins, scale=scale)

    # Define parameters for the plot.
    title = f"2D_Histogram_{scale}_0.5cMpc_h_bin"
    xlabel = "Y (ckpc/h)"
    ylabel = "Z (ckpc/h)"
    box_size = 205  # in cMpc/h

    # Generate the image plot (the data_unit parameter defaults to "ckpc/h").
    saved_files = viz.create_image_plot(
        hist=hist,
        edges1=edges1,
        edges2=edges2,
        results_folder=RESULTS_DIR,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        projection_axis=0,
        x_range=x_thickness,
        x_center=x_center,
        sampling_rate=sampling_rate,
        x_min=x_min,
        x_max=x_max,
        input_folder=npy_folder,
        results_dir=RESULTS_DIR,
        bins=bins,
        box_size=box_size,
        scale=scale,
    )

    # Log the file paths of the saved outputs.
    for fmt, path in saved_files.items():
        logger.info(f"{fmt} file saved: {path}")
    logger.info(f"Processing for scale '{scale}' completed successfully.")
except Exception as e:
    logger.error(f"Error occurred for scale '{scale}': {e}")

# Measure and log the total execution time.
end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
logger.info(f"Total execution time: {formatted_time} ({elapsed_time:.2f} seconds)")
logger.info("Plot generated successfully.")

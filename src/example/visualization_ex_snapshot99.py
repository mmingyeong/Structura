#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: visualization_ex_snapshot99.py

import sys
import os
import time  # For measuring execution time

# Append the parent directory (src) to the Python module search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization import Visualizer
from data_loader import DataLoader
from config import RESULTS_DIR, OUTPUT_DATA_PATHS
from logger import logger

start_time = time.time()  # Record the starting time of the execution

# ------------------------------------------------------------------
# Configuration: Snapshot99 (z = 0) data are provided in cMpc/h units.
# The plot will be generated for a slice with a thickness of 10 cMpc/h,
# corresponding to a filtering range from 100 to 110 cMpc/h.
# ------------------------------------------------------------------
npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
logger.info("Input directory (npy): %s", npy_folder)
logger.info("Output directory: %s", RESULTS_DIR)

# Verify the presence of .npy files within the input directory.
if os.path.exists(npy_folder) and os.path.isdir(npy_folder):
    file_list = os.listdir(npy_folder)
    npy_files = [f for f in file_list if f.endswith(".npy")]

    if npy_files:
        logger.debug("Detected %d .npy file(s) in the input directory.", len(npy_files))
        logger.debug("Example file: %s", npy_files[0])
    else:
        logger.warning("No .npy files were found in the input directory.")
else:
    logger.error("The input directory does not exist: %s", npy_folder)

# ------------------------------------------------------------------
# Define the filtering range.
# For z = 0 data (in cMpc/h), the x-range is set from 100 to 110 cMpc/h.
# ------------------------------------------------------------------
x_min = 100  # cMpc/h
x_max = 110  # cMpc/h
x_thickness = x_max - x_min  # Thickness of 10 cMpc/h
x_center = (x_min + x_max) / 2  # Center at 105 cMpc/h
logger.debug("Filtering range set to: %.2f - %.2f cMpc/h (Center: %.2f cMpc/h, Thickness: %.2f cMpc/h)",
             x_min, x_max, x_center, x_thickness)

# Load the data using DataLoader.
logger.info("Loading data...")
loader = DataLoader(npy_folder)
sampling_rate = 1  # Utilize the full dataset (sampling rate = 1)
positions = loader.load_all_chunks(x_min=x_min, x_max=x_max, sampling_rate=sampling_rate)
logger.debug("Loaded data shape: %s", positions.shape)
logger.info("Data loaded successfully.")

# ------------------------------------------------------------------
# Visualization: Compute a 2D histogram and generate an image plot.
# ------------------------------------------------------------------
logger.info("Computing 2D histogram and generating image plot...")
viz = Visualizer(use_gpu=True)  # Enable GPU acceleration

# Set the number of bins; here, 410 bins yield a resolution of 0.5 cMpc/h.
bins = 410
# Define the list of scale transformations to be applied.
scales = ["log10", "log2", "ln", "sqrt", "linear"]

# For demonstration purposes, select the first scale from the list.
scale = scales[0]
try:
    # Compute the 2D histogram.
    hist, edges1, edges2 = viz.compute_2d_histogram(positions, bins=bins, scale=scale)

    # Define plot parameters.
    title = f"2D_Histogram_{scale}_0.5cMpc_h_bin"
    xlabel = "Y (ckpc/h)"
    ylabel = "Z (ckpc/h)"
    box_size = 205  # in cMpc/h

    # Generate the image plot. Note: The data_unit parameter defaults to "ckpc/h".
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

    # Log the paths of the saved files.
    for fmt, path in saved_files.items():
        logger.info("%s file saved: %s", fmt, path)
    logger.info("Processing for scale '%s' completed successfully.", scale)
except Exception as e:
    logger.error("An error occurred for scale '%s': %s", scale, e)

# Measure and log the total execution time.
end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
logger.info("Total execution time: %s (%.2f seconds)", formatted_time, elapsed_time)
logger.info("Plot generated successfully.")

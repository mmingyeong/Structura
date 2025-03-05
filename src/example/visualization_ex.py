#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: visualization_ex.py

import sys
import os
import time  # For execution time measurement

# Add the parent directory (src) to the Python module path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization import Visualizer
from data_loader import DataLoader
from config import RESULTS_DIR, OUTPUT_DATA_PATHS, LBOX_CMPCH, LBOX_CKPCH
from logger import logger
from utils import set_x_range

start_time = time.time()  # Record start time

# Configuration (all units in cMpc/h)
npy_folder = OUTPUT_DATA_PATHS["NPY"]
logger.info(f"📁 Input (npy) folder: {npy_folder}")
logger.info(f"📁 Output folder: {RESULTS_DIR}")

# Log file formats in the input folder.
if os.path.exists(npy_folder) and os.path.isdir(npy_folder):
    file_list = os.listdir(npy_folder)
    npy_files = [f for f in file_list if f.endswith('.npy')]
    
    if npy_files:
        logger.info(f"📂 Detected {len(npy_files)} .npy files in the input folder.")
        logger.info(f"📄 Example file: {npy_files[0]}")
    else:
        logger.warning("⚠️ No .npy files found in the input folder!")
else:
    logger.error(f"❌ Input folder does not exist: {npy_folder}")

# Define filtering and simulation settings.
x_center = 100      # X-axis center (cMpc/h)
x_thickness = 10    # X-range thickness (cMpc/h)

# Convert X-range (cMpc/h → ckpc/h)
x_min, x_max = set_x_range(center_cMpc=x_center, thickness_cMpc=x_thickness, 
                           lbox_cMpch=LBOX_CMPCH, lbox_ckpch=LBOX_CKPCH)
logger.info(f"🔹 Filtering X range: {x_min:.2f} - {x_max:.2f} ckpc/h "
            f"({x_min/1000:.3f} - {x_max/1000:.3f} cMpc/h)")

# Load data.
logger.info("🔹 Loading data...")
loader = DataLoader(npy_folder)
sampling_rate = 1  # 0.1% sampling (i.e. 0.001 fraction)
positions = loader.load_all_chunks(x_min=x_min, x_max=x_max, sampling_rate=sampling_rate)
logger.info(f"✅ Data loaded successfully. Shape: {positions.shape}")

# Visualization: Compute 2D histogram.
logger.info("🔹 Computing 2D histogram and generating image plot...")
viz = Visualizer(use_gpu=True)  # Use GPU acceleration

# Compute the 2D histogram with default parameters (bins=500, projection_axis=0, scale="log10").
hist, edges1, edges2 = viz.compute_2d_histogram(positions)

# Define required plot parameters.
# Note: title은 필수 인자이므로, 적절한 문자열을 지정합니다.
title = None
xlabel = "Y (ckpc/h)"
ylabel = "Z (ckpc/h)"

# Create and save the image plot using only the required arguments.
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
    lbox_cMpc=LBOX_CMPCH,
    lbox_ckpch=LBOX_CKPCH,
    x_min=x_min,
    x_max=x_max,
    input_folder=npy_folder,
    results_dir=RESULTS_DIR
)

# Log the saved file paths.
for fmt, path in saved_files.items():
    logger.info(f"✅ {fmt} file saved: {path}")

# Measure and log total execution time.
end_time = time.time()
elapsed_time = end_time - start_time  # Total execution time in seconds.
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
logger.info(f"⏳ Total execution time: {formatted_time} ({elapsed_time:.2f} seconds)")
logger.info("✅ Plot generated successfully!")

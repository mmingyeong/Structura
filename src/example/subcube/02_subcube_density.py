#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_subcube_density.py

This script loads subcube data from a numpy file, computes a 3D density map using the 
DensityCalculator class with a Gaussian kernel and a fixed bandwidth (h=1), and saves 
the resulting density map to an npy file. Detailed logging, execution time reporting, and 
profiling (cProfile) are included.

Author: Mingyeong
Date: 2025-04-02
"""

import os
import logging
import numpy as np
import sys
import time
import cProfile

# Add parent directory (src) to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configure logging to both terminal and a file.
logger = logging.getLogger("SubcubeLogger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler: log file for persistent logging
fh = logging.FileHandler("subcube_density.log", mode="a")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

from density import DensityCalculator
from kernel import KernelFunctions

def main():
    # Record start time
    start_time = time.time()
    logger.info("Start time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    
    logger.info("Starting density computation for subcube with periodic boundaries.")
    
    # 1. Load subcube data (expected to be an array of particle coordinates with shape (N, 3))
    cube_data_file = 'subcube_npy/subcube_0000.npy'
    logger.info("Loading cube data from file: %s", cube_data_file)
    try:
        particles = np.load(cube_data_file)
    except Exception as e:
        logger.error("Error loading cube data from %s: %s", cube_data_file, e)
        sys.exit(f"Error loading cube data from {cube_data_file}: {e}")
    
    logger.info("Cube data loaded successfully. Shape: %s", particles.shape)
    
    # 2. Set grid bounds explicitly to reflect the periodic subcube domain.
    # For periodic boundary conditions, we enforce that the subcube has a physical size of 56.
    # This accounts for a base cube size of 50 with an overlap of 3 on each side.
    subcube_size = 56.0
    grid_bounds = {
        'x': (0.0, subcube_size),
        'y': (0.0, subcube_size),
        'z': (0.0, subcube_size)
    }
    logger.info("Grid bounds set for periodic boundary conditions: %s", grid_bounds)
    
    # 3. Define grid spacing for the subcube such that the number of grid cells equals the subcube size.
    # That is, we set the number of grid cells along each axis to be int(subcube_size) (e.g., 56),
    # which results in a grid spacing of subcube_size / 56 = 1.0 in each dimension.
    n_grid_sub = int(subcube_size)  # Number of grid cells per axis equals the subcube size.
    grid_spacing = (subcube_size / n_grid_sub, subcube_size / n_grid_sub, subcube_size / n_grid_sub)
    logger.info("Subcube grid spacing set to: %s (grid count = %d, subcube_size = %.1f)", 
                grid_spacing, n_grid_sub, subcube_size)
    
    # 4. Instantiate the DensityCalculator with the loaded particles and defined grid parameters.
    density_calculator = DensityCalculator(particles, grid_bounds, grid_spacing)
    logger.info("DensityCalculator instantiated with grid_bounds: %s and grid_spacing: %s", grid_bounds, grid_spacing)
    
    # 5. Compute the density map using the Gaussian kernel with fixed bandwidth h=1.
    # Note: The current implementation in DensityCalculator does not explicitly apply the
    # minimum image convention for periodic boundaries. To fully enforce periodicity, the
    # distance computation should adjust differences as:
    #       diff = diff - L * rint(diff / L)
    # where L is the box size. Ensure that such an adjustment is applied within the kernel
    # function or during the distance computation if required.
    fixed_bandwidth = 1
    logger.info("Starting density map computation using Gaussian kernel with fixed bandwidth h=%d.", fixed_bandwidth)
    x_centers, y_centers, z_centers, density_map = density_calculator.calculate_density_map(
        kernel_func=KernelFunctions.gaussian, h=fixed_bandwidth
    )
    logger.info("Density map computation completed. Density map shape: %s", density_map.shape)
    
    # 6. Determine output file name dynamically based on the input file name.
    base_name = os.path.basename(cube_data_file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_file = f"density_{name_without_ext}.npy"
    logger.info("Saving density map to output file: %s", output_file)
    
    np.save(output_file, density_map)
    logger.info("Density map successfully saved to %s", output_file)
    print(f"Density map successfully saved to {output_file}.")
    
    # Record end time and report total execution time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info("End time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    logger.info("Total execution time: %.2f seconds", total_time)
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("End time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("Total execution time: {:.2f} seconds".format(total_time))

if __name__ == '__main__':
    # Run main() with profiling using cProfile; the profiling data is saved to a file.
    cProfile.run('main()', filename='cprofile_subcube_density.prof')

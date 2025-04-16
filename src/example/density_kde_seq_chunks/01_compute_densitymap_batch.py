#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_densitymap_batch_gpu.py

This script processes .npy particle data files in a specified directory (from configuration),
computes continuous density maps using particle-centered kernel smoothing with GPU acceleration,
and saves the computed density maps to dedicated output directories.

Modifications in this version:
  - For each file, the particle data is loaded only once.
  - Density maps are computed for different kernel functions and grid resolutions.
    Specifically, the kernel functions KernelFunctions.uniform and KernelFunctions.triangular
    are used in combination with grid resolutions 0.41, 0.164, and 0.082.
  - Parallel processing per chunk is retained for efficient GPU use.
  
Author: Mingyeong Yang (Modified for multiple kernel and resolution configurations)
Date: 2025-04-11
"""

import os
import logging
import numpy as np
import sys
import time
import cProfile
import pstats

# Add parent directory (src) to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Load paths from configuration
from config import OUTPUT_DATA_PATHS
npy_folder: str = OUTPUT_DATA_PATHS["TNG300_snapshot99"]

# Configure logging (WARNING level to reduce verbosity)
logger = logging.getLogger("BatchDensityLogger")
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("batch_density.log", mode="a")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

from tqdm import tqdm
from density import DensityCalculator
from kernel import KernelFunctions

# ----------------------------------------------------------------------------
def process_chunk(chunk: np.ndarray, grid_bounds: dict, grid_spacing: tuple, bandwidth: float, kernel) -> np.ndarray:
    """
    Processes a chunk of particle data to compute a partial density map using GPU acceleration.
    
    Parameters:
        chunk (np.ndarray): Array of particle coordinates with shape (N_chunk, 3).
        grid_bounds (dict): Dictionary defining the simulation volume, e.g., {'x': (xmin, xmax), ...}.
        grid_spacing (tuple): Tuple of grid spacings (dx, dy, dz).
        bandwidth (float): Kernel bandwidth.
        kernel (callable): Kernel function to be used for density estimation.
    
    Returns:
        np.ndarray: The partial density map computed from the particle chunk.
    """
    # Instantiate DensityCalculator with GPU acceleration enabled.
    density_calculator = DensityCalculator(chunk, grid_bounds, grid_spacing, use_gpu=True)
    _, _, _, partial_map = density_calculator.calculate_density_map(kernel_func=kernel, h=bandwidth)
    return partial_map

# ----------------------------------------------------------------------------
def compute_density_map(particles: np.ndarray, output_file: str,
                        cube_size: float = 205.0,
                        bandwidth: float = 1.0,
                        res: float = 0.82,
                        chunk_size: int = 1_000_000,
                        kernel=KernelFunctions.uniform) -> None:
    """
    Computes the density map from particle data and saves the result to a specified file.
    
    This function splits the particle data into chunks, computes partial density maps in parallel
    using GPU acceleration, and aggregates the results.
    
    Parameters:
        particles (np.ndarray): Array of particle coordinates.
        output_file (str): Path to the output .npy file where the density map will be saved.
        cube_size (float): The side length of the simulation cube (default: 205.0).
        bandwidth (float): Kernel bandwidth.
        res (float): Grid spacing.
        chunk_size (int): Number of particles per processing chunk.
        kernel (callable): Kernel function to be used (e.g., KernelFunctions.uniform or triangular).
    
    Returns:
        None.
    """
    grid_bounds = {
        'x': (0.0, cube_size),
        'y': (0.0, cube_size),
        'z': (0.0, cube_size)
    }
    grid_spacing = (res, res, res)
    n_total = particles.shape[0]
    
    # Split particle data into chunks to manage memory and leverage parallelism.
    chunks = [particles[i:i + chunk_size] for i in range(0, n_total, chunk_size)]

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(process_chunk,
                               chunks,
                               [grid_bounds] * len(chunks),
                               [grid_spacing] * len(chunks),
                               [bandwidth] * len(chunks),
                               [kernel] * len(chunks))
        # Retrieve the partial density maps.
        partial_maps = list(results)
    
    # Aggregate partial maps to construct the final density map.
    density_map = np.sum(np.array(partial_maps), axis=0)
    np.save(output_file, density_map)
    print(f"Density map saved to {output_file}")

# ----------------------------------------------------------------------------
def main() -> None:
    """
    Main routine to iterate over all .npy particle data files and compute
    density maps for multiple kernel functions and grid resolutions.
    
    Each density map is saved in a dedicated output directory named according to
    the kernel function and grid spacing.
    """
    # Specify desired grid resolutions and kernel functions.
    resolution_list = [0.41, 0.164, 0.082]
    kernel_list = [KernelFunctions.uniform, KernelFunctions.triangular]
    
    # Retrieve the sorted list of .npy files.
    all_files = sorted([f for f in os.listdir(npy_folder) if f.endswith(".npy")])
    start_time = time.time()
    
    # Process each file with a progress bar.
    for fname in tqdm(all_files, desc="Processing files"):
        full_path = os.path.join(npy_folder, fname)
        base_name = os.path.splitext(os.path.basename(full_path))[0]
        
        try:
            # Load particle data once per file.
            particles = np.load(full_path)
        except Exception as e:
            logger.error(f"Failed to load {full_path}: {e}")
            continue
        
        # Iterate over all kernel/resolution combinations.
        # kernel, res = 0,0 [~] 
        # kernel, res = 0,1 []
        # kernel, res = 0, 2 []
        # kernel, res = 1,0 []
        # kernel, res = 1,1 []
        # kernel, res = 1,2 []
        kernel = kernel_list[0]
        res = resolution_list[0]
        out_dir = f"densitymap_{kernel.__name__}_dx{res}"
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, f"density_{base_name}_{kernel.__name__}.npy")
        
        if os.path.exists(output_file):
            logger.warning(f"Skipping {full_path} for {kernel.__name__} with dx{res}: Output already exists.")
            continue
        
        # Compute and save the density map for the specific configuration.
        compute_density_map(particles, output_file, cube_size=205.0,
                            bandwidth=1.0, res=res, chunk_size=1_000_000, kernel=kernel)
    
    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    # Profile the main routine and save profiling information.
    cProfile.run('main()', filename='cprofile_batch_density.prof')
    with open('cprofile_output.txt', 'w') as f:
        stats = pstats.Stats('cprofile_batch_density.prof', stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats()
    print("Profiling results have been saved to cprofile_output.txt.")

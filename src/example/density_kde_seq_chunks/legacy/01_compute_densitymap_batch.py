#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_compute_densitymap_batch.py

This script processes .npy particle data files in a specified directory (from configuration),
computes continuous density maps using particle-centered kernel smoothing with GPU acceleration,
and saves the computed density maps to dedicated output directories.

Modifications in this version:
  - For each file, the particle data is loaded only once.
  - Density maps are computed for different kernel functions and grid resolutions.
    Specifically, the kernel functions KernelFunctions.uniform and KernelFunctions.triangular
    are used in combination with grid resolutions 0.41, 0.164, and 0.082.
  - Parallel processing per chunk is retained for efficient GPU use.
  - Calculation results are stored as temporary .npy files per chunk and are later aggregated using np.load with mmap_mode to reduce memory usage.
  - The npy folder path is loaded from configuration.

Author: Mingyeong Yang (Modified for multiple kernel and resolution configurations)
Date: 2025-04-11
"""

import os
import sys
import time
import cProfile
import pstats
import logging
import numpy as np

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
from concurrent.futures import ProcessPoolExecutor

# Import custom modules
from density import DensityCalculator
from kernel import KernelFunctions


def process_chunk_and_save(chunk_id: int, chunk: np.ndarray, grid_bounds: dict,
                           grid_spacing: tuple, bandwidth: float, kernel,
                           output_dir: str, base_name: str) -> str:
    """
    Computes a partial density map for a given chunk of particle data using GPU acceleration,
    and saves the result as an .npy file.

    Parameters:
        chunk (np.ndarray): Particle coordinate array for this chunk.
        grid_bounds (dict): Grid boundaries, e.g., {'x': (xmin, xmax), ...}.
        grid_spacing (tuple): Grid spacing (dx, dy, dz).
        bandwidth (float): Kernel bandwidth.
        kernel (callable): Kernel function to use (e.g., KernelFunctions.triangular).
        output_dir (str): Directory path for temporary storage.
        base_name (str): Base name for the output file.

    Returns:
        str: Path to the saved partial density map (.npy file).
    """
    # Compute partial density map using GPU acceleration
    density_calculator = DensityCalculator(chunk, grid_bounds, grid_spacing, use_gpu=True)
    _, _, _, partial_map = density_calculator.calculate_density_map(kernel_func=kernel, h=bandwidth)
    
    # Save the partial density map to a temporary npy file
    out_path = os.path.join(output_dir, f"{base_name}_chunk{chunk_id}.npy")
    np.save(out_path, partial_map)
    return out_path


def compute_density_map(particles: np.ndarray, output_file: str,
                        cube_size: float = 205.0,
                        bandwidth: float = 1.0,
                        res: float = 0.82,
                        chunk_size: int = 1_000_000,
                        kernel=KernelFunctions.triangular) -> None:
    """
    Computes a density map from particle data by dividing the data into chunks,
    computing partial density maps, and aggregating them using np.load with memory mapping.

    Parameters:
        particles (np.ndarray): Array of particle coordinates.
        output_file (str): Path where the final density map will be saved.
        cube_size (float): Side length of the simulation cube.
        bandwidth (float): Kernel bandwidth.
        res (float): Grid resolution.
        chunk_size (int): Number of particles per chunk.
        kernel (callable): Kernel function to use.

    Returns:
        None.
    """
    # Define grid boundaries and spacing
    grid_bounds = {
        'x': (0.0, cube_size),
        'y': (0.0, cube_size),
        'z': (0.0, cube_size)
    }
    grid_spacing = (res, res, res)
    n_total = particles.shape[0]
    
    # Split particle data into chunks
    chunks = [particles[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
    
    # Create a temporary directory to store partial density maps
    tmp_dir = output_file + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    
    # Parallel computation of partial density maps for each chunk
    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, chunk in enumerate(chunks):
            futures.append(
                executor.submit(
                    process_chunk_and_save,
                    i, chunk, grid_bounds, grid_spacing, bandwidth, kernel,
                    tmp_dir, base_name
                )
            )
    
    # Collect file paths of the computed partial density maps
    chunk_file_paths = [f.result() for f in futures]
    
    # Aggregate partial maps using memory mapping to reduce memory consumption
    density_map = None
    for path in chunk_file_paths:
        partial_map = np.load(path, mmap_mode="r")
        if density_map is None:
            density_map = np.zeros_like(partial_map)
        density_map += partial_map
        # Remove the temporary file after it has been used
        os.remove(path)
    os.rmdir(tmp_dir)
    
    # Save the final aggregated density map
    np.save(output_file, density_map)
    print(f"Density map saved to {output_file}")


def main() -> None:
    """
    Main function to process each npy particle data file, compute the density maps,
    and save the results to a specified output directory.
    """
    # Use the npy_folder path loaded from configuration
    all_files = sorted([f for f in os.listdir(npy_folder) if f.endswith(".npy")])
    start_time = time.time()
    
    # Set fixed kernel and resolution parameters
    kernel = KernelFunctions.triangular
    res = 0.41
    # Set the output directory for the final density maps
    out_dir = f"/caefs/data/IllustrisTNG/density_map_99/kde_densitymap_{kernel.__name__}_dx{res}"
    os.makedirs(out_dir, exist_ok=True)
    
    for fname in tqdm(all_files, desc="Processing files"):
        full_path = os.path.join(npy_folder, fname)
        base_name = os.path.splitext(os.path.basename(full_path))[0]
        try:
            particles = np.load(full_path)
        except Exception as e:
            logger.error(f"Failed to load {full_path}: {e}")
            continue
        
        output_file = os.path.join(out_dir, f"density_{base_name}_{kernel.__name__}_dx{res}.npy")
        if os.path.exists(output_file):
            logger.warning(f"Skipping {full_path}: Output already exists.")
            continue
        
        compute_density_map(particles, output_file, cube_size=205.0,
                            bandwidth=1.0, res=res, chunk_size=1_000_000, kernel=kernel)
    
    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()

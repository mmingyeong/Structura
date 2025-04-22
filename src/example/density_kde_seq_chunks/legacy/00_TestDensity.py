#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_compute_densitymap_test.py

This test script computes a single density map from either an .npy or .h5/.hdf5 file
in the configured TNG300_snapshot99 folder. Select the input format via the
file_format parameter and specify the HDF5 dataset path if needed.
Performance is measured via cProfile, and results are saved under the 'test' directory.

Author: Mingyeong Yang (Test Mode)
Date:   2025-04-21
"""

import os
import sys
import time
import logging
import cProfile
import pstats
import numpy as np
import h5py
import re

# Add parent directory (src) to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Load paths from configuration
from config import OUTPUT_DATA_PATHS, INPUT_DATA_PATHS
from density import DensityCalculator
from kernel import KernelFunctions

# ---- PARAMETERS (modify as needed) ----
file_index   = 7                             # Index of the file to test
file_format  = 'hdf5'                        # 'npy' or 'hdf5'
hdf5_dataset = 'PartType1/Coordinates'       # HDF5 dataset path inside file
kernel_func  = KernelFunctions.triangular     # Kernel: uniform or triangular
res          = 0.82                        # Grid resolution (cMpc/h)
bandwidth    = 2000                           # Kernel bandwidth (h)
use_lut = True
# ---------------------------------------

# Assign default kernel if not set
if kernel_func is None:
    kernel_func = KernelFunctions.triangular

# Determine input folder based on format
if file_format == 'npy':
    input_paths  = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    input_folder = input_paths[0] if isinstance(input_paths, (list, tuple)) else input_paths
elif file_format == 'hdf5':
    input_paths  = INPUT_DATA_PATHS["TNG300_snapshot99"]
    input_folder = input_paths[0] if isinstance(input_paths, (list, tuple)) else input_paths
else:
    logging.error("Unsupported file_format: %s", file_format)
    sys.exit(1)

# Prepare test output directory
test_dir = os.path.join(os.getcwd(), 'test')
os.makedirs(test_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger("TestDensityLogger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

# Set up file extensions based on format
if file_format == 'npy':
    exts = ('.npy',)
elif file_format == 'hdf5':
    exts = ('.h5', '.hdf5')

# Discover files in input_folder
total_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(exts))
if not total_files:
    logger.error(f"No {exts} files found in {input_folder}")
    sys.exit(1)

# Validate file_index
if file_index < 0 or file_index >= len(total_files):
    logger.error(f"file_index={file_index} out of range (0..{len(total_files)-1})")
    sys.exit(1)

input_file = os.path.join(input_folder, total_files[file_index])

# Extract index string from filename (e.g., snapshot-99.114.hdf5 → "114")
base_name = os.path.splitext(os.path.basename(input_file))[0]
match = re.search(r"(\d+)$", base_name)
index_str = match.group(1) if match else "000"

# Set output HDF5 file name
output_path = os.path.join(test_dir, f"density_snapshot99_{index_str}.hdf5")
profile_path = os.path.join(test_dir, 'profile.txt')

def save_density_to_hdf5(output_path, group_key, density_map,
                         x_centers, y_centers, z_centers, meta_dict,
                          compression=None):
    """
    Save the computed density map and grid info to an HDF5 file under a specific group key.
    """
    start = time.time()
    with h5py.File(output_path, 'a') as f:
        if group_key in f:
            del f[group_key]
        grp = f.create_group(group_key)
        grp.create_dataset("density_map", data=density_map, chunks=None, compression=compression)
        grp.create_dataset("x_centers", data=x_centers)
        grp.create_dataset("y_centers", data=y_centers)
        grp.create_dataset("z_centers", data=z_centers)
        for key, val in meta_dict.items():
            try:
                # 문자열로 저장 가능한 경우: 문자열
                if isinstance(val, str):
                    grp.attrs[key] = val
                # 숫자형 타입 (int, float)
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    grp.attrs[key] = val
                # 기타: 문자열로 변환해서 저장
                else:
                    grp.attrs[key] = str(val)
                    logger.warning(f"Converted {key}={val} to string for HDF5 attribute.")
            except Exception as e:
                logger.error(f"Failed to store attribute {key}: {val} — {e}")
    elapsed = time.time() - start
    logging.info(f"[HDF5 Write] Saved density_map → {output_path} ({group_key}), took {elapsed:.3f}s")

def load_particles_hdf5_direct(file_path, dataset_path):
    """
    Efficient loading using .read_direct()
    """
    with h5py.File(file_path, 'r') as hf:
        dset = hf[dataset_path]
        shape = dset.shape
        dtype = dset.dtype
        logger.info(f"Reading particles via read_direct: shape={shape}, dtype={dtype}")
        particles = np.empty(shape, dtype=dtype)
        dset.read_direct(particles)
    return particles

# ---- Main Execution ----
def main():
    # Load particles
    if file_format == 'npy':
        logger.info(f"Loading particles from NPY file {input_file}")
        particles = np.load(input_file, mmap_mode=None)
        cube_size = 205.0
        dx = res
    else:
        logger.info(f"Loading particles from HDF5 file {input_file}")
        with h5py.File(input_file, 'r') as hf:
            try:
                logger.info(f"Reading particles from HDF5: {input_file}")
                particles = load_particles_hdf5_direct(input_file, hdf5_dataset)
            except (KeyError, TypeError):
                available = list(hf.keys())
                logger.error(f"Dataset path '{hdf5_dataset}' not found in {input_file}. Top-level keys: {available}")
                sys.exit(1)
        cube_size = 205000.0  # ckpc/h
        dx = res * 1000.0     # convert from cMpc/h to ckpc/h

    logger.info(f"Number of particles: {particles.shape[0]:,}")

    # Grid setup
    grid_bounds = {'x': (0.0, cube_size), 'y': (0.0, cube_size), 'z': (0.0, cube_size)}
    grid_spacing = (dx, dx, dx)
    logger.info(f"Grid cube size = {cube_size:.1f} ({'cMpc/h' if file_format == 'npy' else 'ckpc/h'})")
    logger.info(f"Grid spacing dx = {dx:.1f} ({'cMpc/h' if file_format == 'npy' else 'ckpc/h'})")

    # Compute density
    logger.info("Starting density calculation...")
    start_time = time.time()
    calc = DensityCalculator(particles, grid_bounds, grid_spacing, use_gpu=False)
    x_centers, y_centers, z_centers, density_map = calc.calculate_density_map(kernel_func=kernel_func, h=bandwidth)
    elapsed = time.time() - start_time

    # Save
    group_key = f"{kernel_func.__name__}_dx{res:.2f}"
    meta_dict = {
        "kernel": kernel_func.__name__,
        "bandwidth": bandwidth,
        "resolution_dx": res * 1000.0,
        "particle_count": particles.shape[0]
    }
    save_density_to_hdf5(output_path, group_key, density_map,
                         x_centers, y_centers, z_centers, meta_dict)

    logger.info(f"Density map saved to {output_path}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")

if __name__ == '__main__':
    logger.info("Warming up Numba compilation (outside profiler)...")
    warm_particles = np.zeros((10, 3))
    warm_calc = DensityCalculator(warm_particles,
                                  {'x': (0.0, 205.0), 'y': (0.0, 205.0), 'z': (0.0, 205.0)},
                                  (res, res, res), use_gpu=False)
    warm_calc.calculate_density_map(kernel_func=kernel_func, h=bandwidth, use_lut=use_lut)

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    with open(profile_path, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumtime')
        stats.print_stats()

    logger.info(f"Profiling results saved to {profile_path}")
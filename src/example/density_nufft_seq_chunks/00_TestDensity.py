#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_compute_densitymap_nufft.py

This script computes a single density map using the NUFFTKDE class.
Input data can be either in .npy or HDF5 format (e.g., TNG300).
Output is saved to an HDF5 file with full grid metadata.

Author: Mingyeong Yang (NUFFT version)
Date: 2025-04-22
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import OUTPUT_DATA_PATHS, INPUT_DATA_PATHS
from kernel import KernelFunctions
from nufft_kde import NUFFTKDE

# ---- PARAMETERS ----
file_index = 7
file_format = 'hdf5'
hdf5_dataset = 'PartType1/Coordinates'
kernel_func = KernelFunctions.triangular  # uniform or triangular
res = 0.
bandwidth = 2000
compression = None
# ---------------------

logger = logging.getLogger("NUFFTKDELogger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

if file_format == 'npy':
    input_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    input_folder = input_folder[0] if isinstance(input_folder, (list, tuple)) else input_folder
    exts = ('.npy',)
    cube_size = 205.0
    dx = res
elif file_format == 'hdf5':
    input_folder = INPUT_DATA_PATHS["TNG300_snapshot99"]
    input_folder = input_folder[0] if isinstance(input_folder, (list, tuple)) else input_folder
    exts = ('.h5', '.hdf5')
    cube_size = 205000.0
    dx = res * 1000.0
else:
    logger.error("Unsupported file format.")
    sys.exit(1)

test_dir = os.path.join(os.getcwd(), 'test')
os.makedirs(test_dir, exist_ok=True)

files = sorted(f for f in os.listdir(input_folder) if f.endswith(exts))
input_file = os.path.join(input_folder, files[file_index])

match = re.search(r"(\d+)$", os.path.splitext(os.path.basename(input_file))[0])
index_str = match.group(1) if match else "000"
output_path = os.path.join(test_dir, f"density_snapshot99_{index_str}.hdf5")
profile_path = os.path.join(test_dir, 'profile.txt')

def load_particles():
    if file_format == 'npy':
        return np.load(input_file)
    else:
        with h5py.File(input_file, 'r') as hf:
            dset = hf[hdf5_dataset]
            particles = np.empty(dset.shape, dtype=dset.dtype)
            dset.read_direct(particles)
        return particles

def save_density(output_path, group_key, x_centers, y_centers, z_centers, density_map, meta_dict):
    with h5py.File(output_path, 'a') as f:
        if group_key in f:
            del f[group_key]
        grp = f.create_group(group_key)
        grp.create_dataset("x", data=x_centers, compression=compression)
        grp.create_dataset("y", data=y_centers, compression=compression)
        if z_centers is not None:
            grp.create_dataset("z", data=z_centers, compression=compression)
        grp.create_dataset("density_map", data=density_map, compression=compression)

        for k, v in meta_dict.items():
            grp.attrs[k] = v
    logger.info(f"Saved: {output_path} ({group_key})")

def main():
    logger.info(f"Loading particles from: {input_file}")
    particles = load_particles()
    logger.info(f"Loaded {particles.shape[0]:,} particles")

    grid_bounds = {'x': (0, cube_size), 'y': (0, cube_size), 'z': (0, cube_size)}
    grid_spacing = (dx, dx, dx)

    kde = NUFFTKDE(particles, grid_bounds, grid_spacing, kernel_func=kernel_func, h=bandwidth)

    start = time.time()
    x_centers, y_centers, z_centers, density_map = kde.compute_density()
    elapsed = time.time() - start

    group_key = f"{kernel_func.__name__}_dx{res:.2f}"
    meta = {
        "kernel": kernel_func.__name__,
        "bandwidth": bandwidth,
        "resolution_dx": dx,
        "particle_count": particles.shape[0]
    }
    save_density(output_path, group_key, x_centers, y_centers, z_centers, density_map, meta)
    logger.info(f"Elapsed time: {elapsed:.2f} s")

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    with open(profile_path, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumtime')
        stats.print_stats()
    logger.info(f"Profiling results saved to {profile_path}")
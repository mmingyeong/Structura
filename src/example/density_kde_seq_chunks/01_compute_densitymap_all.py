#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_compute_densitymap_all.py

SLURM Job Array + multiprocessing-compatible script.
Each SLURM task processes one HDF5 input file based on index (sys.argv[1]),
and computes 4 density maps (2 kernels × 2 resolutions).

Author: Mingyeong Yang (Updated for parallel job array processing)
Date: 2025-04-22
"""

import os
import sys
import time
import logging
import numpy as np
import h5py
import re
from multiprocessing import Pool

# Add parent directory (src) to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import OUTPUT_DATA_PATHS, INPUT_DATA_PATHS
from density import DensityCalculator
from kernel import KernelFunctions

# ----------- Configurable Parameters -----------
file_format = 'hdf5'
dataset_path = 'PartType1/Coordinates'
kernel_funcs = [KernelFunctions.triangular, KernelFunctions.uniform]
resolutions = [0.82, 0.41]  # in cMpc/h
bandwidth = 2000
use_lut = True
input_folder = INPUT_DATA_PATHS["TNG300_snapshot99"]
output_folder = "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5"
os.makedirs(output_folder, exist_ok=True)
# -----------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DensityBatchLogger")

# Helper functions
def extract_index(filename):
    match = re.search(r"(\d+)$", os.path.splitext(filename)[0])
    return match.group(1) if match else "000"

def load_particles(file_path, dataset_path):
    with h5py.File(file_path, 'r') as f:
        dset = f[dataset_path]
        particles = np.empty(dset.shape, dtype=dset.dtype)
        dset.read_direct(particles)
    return particles

def save_to_hdf5(output_path, group_key, density_map, x, y, z, meta, compression=None):
    with h5py.File(output_path, 'a') as f:
        if group_key in f:
            del f[group_key]
        grp = f.create_group(group_key)
        grp.create_dataset("density_map", data=density_map, compression=compression)
        grp.create_dataset("x_centers", data=x)
        grp.create_dataset("y_centers", data=y)
        grp.create_dataset("z_centers", data=z)
        for k, v in meta.items():
            grp.attrs[k] = v

def compute_density(args):
    kernel, res, particles, out_path, index_str = args
    dx = res * 1000.0
    spacing = (dx, dx, dx)
    grid_bounds = {'x': (0, 205000.0), 'y': (0, 205000.0), 'z': (0, 205000.0)}
    logger.info(f"  → Kernel: {kernel.__name__}, res: {res:.2f}")

    calc = DensityCalculator(particles, grid_bounds, spacing, use_gpu=False)
    x, y, z, rho = calc.calculate_density_map(kernel_func=kernel, h=bandwidth, use_lut=use_lut)

    group_key = f"{kernel.__name__}_dx{res:.2f}"
    meta = {
        "kernel": kernel.__name__,
        "bandwidth": bandwidth,
        "resolution_dx": dx,
        "particle_count": particles.shape[0]
    }
    save_to_hdf5(out_path, group_key, rho, x, y, z, meta)
    logger.info(f"    → Saved group {group_key} to {out_path}")

if __name__ == '__main__':
    files = sorted(f for f in os.listdir(input_folder) if f.endswith(('.h5', '.hdf5')))

    if len(sys.argv) < 2:
        logger.error("Missing SLURM_ARRAY_TASK_ID or file index argument")
        sys.exit(1)

    idx = int(sys.argv[1])
    if idx < 0 or idx >= len(files):
        logger.error(f"Index {idx} out of range (0 to {len(files) - 1})")
        sys.exit(1)

    fname = files[idx]
    index_str = extract_index(fname)
    in_path = os.path.join(input_folder, fname)
    out_path = os.path.join(output_folder, f"density_snapshot99_{index_str}.hdf5")

    logger.info(f"Processing file {fname} → {out_path}")
    particles = load_particles(in_path, dataset_path)

    tasks = [(k, r, particles, out_path, index_str) for k in kernel_funcs for r in resolutions]

    with Pool(processes=4) as pool:
        pool.map(compute_density, tasks)

    logger.info(f"All kernel-resolution combinations completed for {fname}.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_compute_densitymap_all.py

SLURM Job Array + multiprocessing-compatible script.
Each SLURM task processes one HDF5 input file based on index (sys.argv[1]),
and computes 4 density maps (2 kernels × 2 resolutions), storing each output separately.

Author: Mingyeong Yang
Date: 2025-04-22 (Updated: 2025-05-08)
"""

import os
import sys
import time
import logging
import numpy as np
import h5py
import re
import traceback
from multiprocessing import Pool

# Add parent directory (src) to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

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
input_folder = "/caefs/data/IllustrisTNG/pbs_download/"
base_output_folder = "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5"
fail_log_path = "logs/fail_chunks.txt"
# -----------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DensityBatchLogger")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Helper functions
def extract_index(filename):
    match = re.search(r"(\d+)$", os.path.splitext(filename)[0])
    return match.group(1) if match else "000"

def load_particles(file_path, dataset_path):
    try:
        with h5py.File(file_path, 'r') as f:
            dset = f[dataset_path]
            particles = np.empty(dset.shape, dtype=dset.dtype)
            dset.read_direct(particles)

        if particles.size == 0 or not np.any(particles):
            logger.warning(f"⚠️ Empty or invalid particle array in file: {file_path}")
            return None

        logger.info(f"    → particles.shape = {particles.shape}, min = {particles.min():.2f}, max = {particles.max():.2f}")
        return particles

    except Exception as e:
        logger.error(f"❌ Failed to load particles from {file_path}: {e}\n{traceback.format_exc()}")
        return None

def save_to_hdf5(output_path, density_map, x, y, z, meta, compression=None):
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("density_map", data=density_map, compression=compression)
        f.create_dataset("x_centers", data=x)
        f.create_dataset("y_centers", data=y)
        f.create_dataset("z_centers", data=z)
        for k, v in meta.items():
            f.attrs[k] = v

def log_fail(index_str, kernel_name, res):
    with open(fail_log_path, "a") as f:
        f.write(f"{index_str},{kernel_name},dx={res:.3f}\n")

def compute_density(args):
    kernel, res, particles, index_str = args
    try:
        dx = res * 1000.0
        spacing = (dx, dx, dx)
        grid_bounds = {'x': (0, 205000.0), 'y': (0, 205000.0), 'z': (0, 205000.0)}
        kernel_name = kernel.__name__
        group_key = f"{kernel_name}_dx{res:.2f}"
        logger.info(f"  → Kernel: {kernel_name}, res: {res:.2f}")

        output_dir = os.path.join(base_output_folder, group_key)
        if not os.path.exists(output_dir):
            logger.info(f"📁 Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"density_snapshot99_{index_str}.hdf5")

        if os.path.exists(out_path):
            logger.info(f"    → Output already exists. Skipping: {out_path}")
            return

        calc = DensityCalculator(particles, grid_bounds, spacing, use_gpu=False)
        x, y, z, rho = calc.calculate_density_map(kernel_func=kernel, h=bandwidth, use_lut=use_lut)

        # 유효성 검사
        if not np.any(rho > 0):
            logger.warning(f"⚠️ Density map is all zero for {kernel_name}, dx={res}, index={index_str}")
            log_fail(index_str, kernel_name, res)
            return

        if not np.all(np.isfinite(rho)):
            logger.error(f"❌ Non-finite values found in rho for {kernel_name}, dx={res}, index={index_str}")
            log_fail(index_str, kernel_name, res)
            return

        logger.info(f"    → rho stats: min={rho.min():.4e}, max={rho.max():.4e}, sum={rho.sum():.4e}")

        meta = {
            "kernel": kernel_name,
            "bandwidth": bandwidth,
            "resolution_dx": dx,
            "particle_count": particles.shape[0]
        }
        save_to_hdf5(out_path, rho, x, y, z, meta)
        logger.info(f"    → Saved: {out_path}")

    except Exception as e:
        logger.error(f"❌ Failed to compute/save {kernel.__name__} dx={res:.2f} for index {index_str}: {e}\n{traceback.format_exc()}")
        log_fail(index_str, kernel.__name__, res)

if __name__ == '__main__':
    files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(('.h5', '.hdf5'))],
        key=lambda x: int(extract_index(x))
    )

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

    logger.info(f"📄 Processing file {fname} (index {index_str})")
    particles = load_particles(in_path, dataset_path)
    if particles is None:
        logger.error("❌ No valid particles loaded. Exiting.")
        with open(fail_log_path, "a") as f:
            f.write(f"{index_str},load_failed,dx=NA\n")
        sys.exit(1)

    tasks = [(k, r, particles, index_str) for k in kernel_funcs for r in resolutions]

    with Pool(processes=4) as pool:
        pool.map(compute_density, tasks)

    logger.info(f"✅ All kernel-resolution outputs completed for file {fname}.")

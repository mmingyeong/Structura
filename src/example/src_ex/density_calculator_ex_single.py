#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
density_calculator_ex_single.py

Example script to compute multiple density maps from a single large HDF5 file,
for multiple kernel functions and resolutions. Output is saved to separate HDF5 files.

Author: Mingyeong Yang
Date: 2025-04-22
"""

import os
import time
import logging
import numpy as np
import h5py
from multiprocessing import Pool
from density import DensityCalculator
from kernel import KernelFunctions  # Assumes triangular, uniform kernels are defined here

# ========== 🛠️ 사용자 설정 ==========
INPUT_FILE = "/path/to/your/input_snapshot.hdf5"  # 약 400GB 단일 파일
DATASET_PATH = "PartType1/Coordinates"
OUTPUT_DIR = "/path/to/output_density_maps"
KERNEL_FUNCS = [KernelFunctions.triangular, KernelFunctions.uniform]
RESOLUTIONS = [0.82, 0.41]  # cMpc/h
BANDWIDTH = 2000  # smoothing bandwidth (e.g., in kpc)
USE_LUT = True
USE_GPU = False  # GPU 사용 여부
N_PROCESSES = 4
# =====================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DensitySingleLogger")


def load_particles(hdf5_path, dataset_path):
    logger.info(f"🔍 Loading particle data from: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        dset = f[dataset_path]
        particles = np.empty(dset.shape, dtype=dset.dtype)
        dset.read_direct(particles)
    logger.info(f"✅ Loaded particles: {particles.shape}")
    return particles


def save_density_to_hdf5(out_path, density_map, x, y, z, meta, compression=None):
    with h5py.File(out_path, 'w') as f:
        f.create_dataset("density_map", data=density_map, compression=compression)
        f.create_dataset("x_centers", data=x)
        f.create_dataset("y_centers", data=y)
        f.create_dataset("z_centers", data=z)
        for k, v in meta.items():
            f.attrs[k] = v


def compute_density(args):
    kernel_func, res, particles = args
    try:
        dx = res * 1000.0
        spacing = (dx, dx, dx)
        bounds = {'x': (0, 205000.0), 'y': (0, 205000.0), 'z': (0, 205000.0)}

        kernel_name = kernel_func.__name__
        tag = f"{kernel_name}_dx{res:.2f}"
        output_path = os.path.join(OUTPUT_DIR, f"density_{tag}.hdf5")

        logger.info(f"🧪 Calculating {tag}")
        calc = DensityCalculator(particles, bounds, spacing, use_gpu=USE_GPU)
        x, y, z, rho = calc.calculate_density_map(kernel_func=kernel_func, h=BANDWIDTH, use_lut=USE_LUT)

        meta = {
            "kernel": kernel_name,
            "resolution_dx": dx,
            "bandwidth": BANDWIDTH,
            "particle_count": particles.shape[0],
        }
        save_density_to_hdf5(output_path, rho, x, y, z, meta)
        logger.info(f"💾 Saved result to {output_path}")

    except Exception as e:
        logger.error(f"❌ Error in {kernel_func.__name__}, dx={res:.2f}: {e}")


if __name__ == "__main__":
    start = time.time()

    # Load particle data once
    particles = load_particles(INPUT_FILE, DATASET_PATH)

    # Prepare tasks
    tasks = [(k, r, particles) for k in KERNEL_FUNCS for r in RESOLUTIONS]

    with Pool(processes=N_PROCESSES) as pool:
        pool.map(compute_density, tasks)

    elapsed = time.time() - start
    logger.info(f"🏁 All calculations completed in {elapsed:.2f} seconds.")

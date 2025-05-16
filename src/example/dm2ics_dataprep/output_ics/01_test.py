#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import logging
import sys

# Add parent directory (src) to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from density import DensityCalculator
from kernel import KernelFunctions

import glob

input_folder = "/caefs/data/IllustrisTNG/snapshot-0-ics"
ics_files = sorted(glob.glob(os.path.join(input_folder, "ics_chunk_*.hdf5")))
if not ics_files:
    raise FileNotFoundError("No ics_chunk_*.hdf5 files found in the directory!")

input_path = ics_files[0]  # 예: /caefs/data/IllustrisTNG/snapshot-0-ics/ics_chunk_0008.hdf5

# -------------- Configurable Test Parameters --------------
#input_path = "/caefs/data/IllustrisTNG/snapshot-0-ics/snapshot_0.8.hdf5"
dataset_path = "PartType1/Coordinates"
kernel_func = KernelFunctions.uniform  # Change if needed
resolution = 0.41  # in cMpc/h
bandwidth = 2000  # in ckpc/h
use_lut = True
test_output_path = "test/density_snapshot_0_test.hdf5"
# ---------------------------------------------------------

os.makedirs("test", exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DensityTestLogger")

def load_particles(file_path, dataset_path):
    with h5py.File(file_path, 'r') as f:
        dset = f[dataset_path]
        particles = np.empty(dset.shape, dtype=np.float64)
        dset.read_direct(particles)
    if "snapshot-0-ics" in file_path:
        particles *= 1000.0
    return particles

def save_to_hdf5(output_path, density_map, x, y, z, meta):
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("density_map", data=density_map)
        f.create_dataset("x_centers", data=x)
        f.create_dataset("y_centers", data=y)
        f.create_dataset("z_centers", data=z)
        for k, v in meta.items():
            f.attrs[k] = v

def summarize_density(density):
    print("\n📊 Density Map Statistics")
    print(f" - shape : {density.shape}")
    print(f" - dtype : {density.dtype}")
    print(f" - min   : {np.min(density):.4e}")
    print(f" - max   : {np.max(density):.4e}")
    print(f" - mean  : {np.mean(density):.4e}")
    print(f" - sum   : {np.sum(density):.4e}")

if __name__ == '__main__':
    logger.info(f"▶️ Loading particles from {input_path}")
    particles = load_particles(input_path, dataset_path)
    logger.info(f"  → Particle shape: {particles.shape}, min: {particles.min()}, max: {particles.max()}")

    dx = resolution * 1000.0  # convert cMpc/h → ckpc/h
    spacing = (dx, dx, dx)
    grid_bounds = {'x': (0, 205000.0), 'y': (0, 205000.0), 'z': (0, 205000.0)}

    logger.info(f"▶️ Starting density computation with {kernel_func.__name__}, dx = {dx} ckpc/h")
    calc = DensityCalculator(particles, grid_bounds, spacing, use_gpu=False)
    x, y, z, rho = calc.calculate_density_map(kernel_func=kernel_func, h=bandwidth, use_lut=use_lut)

    summarize_density(rho)

    meta = {
        "kernel": kernel_func.__name__,
        "bandwidth": bandwidth,
        "resolution_dx": dx,
        "particle_count": particles.shape[0],
        "grid_spacing": spacing[0]
    }

    save_to_hdf5(test_output_path, rho, x, y, z, meta)
    logger.info(f"✅ Test density map saved to {test_output_path}")

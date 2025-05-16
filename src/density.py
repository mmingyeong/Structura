#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
@Date:    2025-04-21
@Filename: density.py

3D density estimation with performance optimizations:
- Outer loops parallelized using Numba prange (with caching)
- Grid center coordinates precomputed and reused
- Separate optimized routines for uniform (top-hat) and triangular kernels
- Memory-mapped loading of particle arrays from .npy files
- Silverman's rule for automatic bandwidth selection
- GPU fallback available
- Post-normalization to match total particle count
"""

import numpy as np
from logger import logger

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Numba for CPU acceleration
USE_NUMBA = False
try:
    import numba
    from numba import prange
    USE_NUMBA = True
except ImportError:
    logger.info("Numba not available; running without JIT acceleration.")

# Kernel functions
from kernel import KernelFunctions

# =============================================================================
# Utility: load particles via memory mapping
# =============================================================================
def load_particles(path_or_array):
    """
    Load particle data from a .npy file using memory mapping,
    or accept an existing ndarray.
    """
    if isinstance(path_or_array, str):
        logger.info(f"Loading particles from {path_or_array} with mmap_mode='r'")
        return np.load(path_or_array, mmap_mode='r')
    return path_or_array

# ----------------------------------------------------------------------
# Optional fast sqrt approximation via LUT
# ----------------------------------------------------------------------
@numba.njit(cache=True)
def fast_sqrt_lut(x, table_x, table_y):
    idx = np.searchsorted(table_x, x) - 1
    if idx < 0:
        return table_y[0]
    elif idx >= len(table_x) - 1:
        return table_y[-1]
    x0, x1 = table_x[idx], table_x[idx + 1]
    y0, y1 = table_y[idx], table_y[idx + 1]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


# =============================================================================
# Numba-accelerated uniform (top-hat) kernel density
# =============================================================================
if USE_NUMBA:
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def particle_centered_density_uniform(particles, density_map,
                                          x_centers, y_centers, z_centers,
                                          r_idx,
                                          box_size_x, box_size_y, box_size_z,
                                          h2, inv_volume):
        n_particles = particles.shape[0]
        nx = x_centers.shape[0]
        ny = y_centers.shape[0]
        nz = z_centers.shape[0]
        half_x = box_size_x * 0.5
        half_y = box_size_y * 0.5
        half_z = box_size_z * 0.5
        for i in prange(n_particles):
            px, py, pz = particles[i]
            fx = (px - x_centers[0] + 0.5*(x_centers[1]-x_centers[0])) / (x_centers[1]-x_centers[0])
            fy = (py - y_centers[0] + 0.5*(y_centers[1]-y_centers[0])) / (y_centers[1]-y_centers[0])
            fz = (pz - z_centers[0] + 0.5*(z_centers[1]-z_centers[0])) / (z_centers[1]-z_centers[0])
            cx = int(fx)
            cy = int(fy)
            cz = int(fz)
            ix_min = max(0, cx - r_idx)
            ix_max = min(nx-1, cx + r_idx)
            iy_min = max(0, cy - r_idx)
            iy_max = min(ny-1, cy + r_idx)
            iz_min = max(0, cz - r_idx)
            iz_max = min(nz-1, cz + r_idx)
            for ix in range(ix_min, ix_max+1):
                dx = x_centers[ix] - px
                if dx > half_x:
                    dx -= box_size_x
                elif dx < -half_x:
                    dx += box_size_x
                dx2 = dx*dx
                for iy in range(iy_min, iy_max+1):
                    dy = y_centers[iy] - py
                    if dy > half_y:
                        dy -= box_size_y
                    elif dy < -half_y:
                        dy += box_size_y
                    dy2 = dy*dy
                    for iz in range(iz_min, iz_max+1):
                        dz = z_centers[iz] - pz
                        if dz > half_z:
                            dz -= box_size_z
                        elif dz < -half_z:
                            dz += box_size_z
                        if dx2 + dy2 + dz*dz <= h2:
                            density_map[ix, iy, iz] += inv_volume
else:
    def particle_centered_density_uniform(particles, density_map,
                                          x_centers, y_centers, z_centers,
                                          r_idx,
                                          box_size_x, box_size_y, box_size_z,
                                          h2, inv_volume):
        n_particles = particles.shape[0]
        nx = x_centers.shape[0]
        ny = y_centers.shape[0]
        nz = z_centers.shape[0]
        half_x = box_size_x * 0.5
        half_y = box_size_y * 0.5
        half_z = box_size_z * 0.5
        for i in range(n_particles):
            px, py, pz = particles[i]
            cx = int(round((px - x_centers[0]) / (x_centers[1]-x_centers[0])))
            cy = int(round((py - y_centers[0]) / (y_centers[1]-y_centers[0])))
            cz = int(round((pz - z_centers[0]) / (z_centers[1]-z_centers[0])))
            ix_min = max(0, cx - r_idx)
            ix_max = min(nx-1, cx + r_idx)
            iy_min = max(0, cy - r_idx)
            iy_max = min(ny-1, cy + r_idx)
            iz_min = max(0, cz - r_idx)
            iz_max = min(nz-1, cz + r_idx)
            for ix in range(ix_min, ix_max+1):
                dx = x_centers[ix] - px
                if dx > half_x:
                    dx -= box_size_x
                elif dx < -half_x:
                    dx += box_size_x
                dx2 = dx*dx
                for iy in range(iy_min, iy_max+1):
                    dy = y_centers[iy] - py
                    if dy > half_y:
                        dy -= box_size_y
                    elif dy < -half_y:
                        dy += box_size_y
                    dy2 = dy*dy
                    for iz in range(iz_min, iz_max+1):
                        dz = z_centers[iz] - pz
                        if dz > half_z:
                            dz -= box_size_z
                        elif dz < -half_z:
                            dz += box_size_z
                        if dx2 + dy2 + dz*dz <= h2:
                            density_map[ix, iy, iz] += inv_volume

# =============================================================================
# Numba-accelerated triangular kernel density
# =============================================================================
if USE_NUMBA:
# ----------------------------------------------------------------------
# Triangular kernel (Numba JIT)
# ----------------------------------------------------------------------
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def particle_centered_density_triangular(particles, density_map,
                                            x_centers, y_centers, z_centers,
                                            r_idx,
                                            box_size_x, box_size_y, box_size_z,
                                            h, h2, constant,
                                            use_lut, table_x, table_y):
        n_particles = particles.shape[0]
        nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
        half_x, half_y, half_z = box_size_x * 0.5, box_size_y * 0.5, box_size_z * 0.5

        for i in prange(n_particles):
            px, py, pz = particles[i]
            cx = int((px - x_centers[0]) / (x_centers[1] - x_centers[0]) + 0.5)
            cy = int((py - y_centers[0]) / (y_centers[1] - y_centers[0]) + 0.5)
            cz = int((pz - z_centers[0]) / (z_centers[1] - z_centers[0]) + 0.5)

            for ix in range(max(0, cx - r_idx), min(nx, cx + r_idx + 1)):
                dx = x_centers[ix] - px
                if dx > half_x: dx -= box_size_x
                elif dx < -half_x: dx += box_size_x
                dx2 = dx * dx

                for iy in range(max(0, cy - r_idx), min(ny, cy + r_idx + 1)):
                    dy = y_centers[iy] - py
                    if dy > half_y: dy -= box_size_y
                    elif dy < -half_y: dy += box_size_y
                    dy2 = dy * dy

                    for iz in range(max(0, cz - r_idx), min(nz, cz + r_idx + 1)):
                        dz = z_centers[iz] - pz
                        if dz > half_z: dz -= box_size_z
                        elif dz < -half_z: dz += box_size_z
                        sq = dx2 + dy2 + dz * dz
                        if sq <= h2:
                            dist = fast_sqrt_lut(sq, table_x, table_y) if use_lut else np.sqrt(sq)
                            w = constant * (1.0 - dist / h)
                            density_map[ix, iy, iz] += w
else:
    def particle_centered_density_triangular(particles, density_map,
                                             x_centers, y_centers, z_centers,
                                             r_idx,
                                             box_size_x, box_size_y, box_size_z,
                                             h, h2, constant):
        n_particles = particles.shape[0]
        nx = x_centers.shape[0]
        ny = y_centers.shape[0]
        nz = z_centers.shape[0]
        half_x = box_size_x * 0.5
        half_y = box_size_y * 0.5
        half_z = box_size_z * 0.5
        for i in range(n_particles):
            px, py, pz = particles[i]
            cx = int(round((px - x_centers[0]) / (x_centers[1]-x_centers[0])))
            cy = int(round((py - y_centers[0]) / (y_centers[1]-y_centers[0])))
            cz = int(round((pz - z_centers[0]) / (z_centers[1]-z_centers[0])))
            ix_min = max(0, cx - r_idx)
            ix_max = min(nx-1, cx + r_idx)
            iy_min = max(0, cy - r_idx)
            iy_max = min(ny-1, cy + r_idx)
            iz_min = max(0, cz - r_idx)
            iz_max = min(nz-1, cz + r_idx)
            for ix in range(ix_min, ix_max+1):
                dx = x_centers[ix] - px
                if dx > half_x:
                    dx -= box_size_x
                elif dx < -half_x:
                    dx += box_size_x
                dx2 = dx*dx
                for iy in range(iy_min, iy_max+1):
                    dy = y_centers[iy] - py
                    if dy > half_y:
                        dy -= box_size_y
                    elif dy < -half_h:
                        dy += box_size_y
                    dy2 = dy*dy
                    for iz in range(iz_min, iz_max+1):
                        dz = z_centers[iz] - pz
                        if dz > half_z:
                            dz -= box_size_z
                        elif dz < -half_z:
                            dz += box_size_z
                        sq = dx2 + dy2 + dz*dz
                        if sq <= h2:
                            dist = np.sqrt(sq)
                            density_map[ix, iy, iz] += constant * (1.0 - dist/h)

# ----------------------------------------------------------------------
# DensityCalculator class
# ----------------------------------------------------------------------
class DensityCalculator:
    def __init__(self, particles, grid_bounds, grid_spacing, use_gpu=False):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.use_gpu = use_gpu

    def calculate_density_map(self, kernel_func, h=None, use_lut=False):
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        zmin, zmax = self.grid_bounds['z']
        dx, dy, dz = self.grid_spacing

        x_centers = np.arange(xmin + 0.5 * dx, xmax, dx)
        y_centers = np.arange(ymin + 0.5 * dy, ymax, dy)
        z_centers = np.arange(zmin + 0.5 * dz, zmax, dz)
        nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
        density_map = np.zeros((nx, ny, nz), dtype=np.float64)

        total_particles = self.particles.shape[0]
        box_x, box_y, box_z = xmax - xmin, ymax - ymin, zmax - zmin

        # Automatically estimate optimal h using Silverman's rule if not provided
        if h is None:
            std_x = np.std(self.particles[:, 0])
            std_y = np.std(self.particles[:, 1])
            std_z = np.std(self.particles[:, 2])
            std_avg = (std_x + std_y + std_z) / 3.0
            h = 1.06 * std_avg * (total_particles ** (-1.0 / 5.0))
            logger.info(f"Estimated optimal bandwidth h = {h:.4f} using Silverman's rule")

        r_idx = int(np.ceil(h / dx))
        h2 = h * h

        if r_idx == 0:
            logger.warning("Kernel support radius is zero (r_idx == 0) — skipping density computation.")
            return x_centers, y_centers, z_centers, density_map

        if kernel_func.__name__ == "triangular":
            constant = 3.0 / (np.pi * h ** 3)
            table_x, table_y = (np.linspace(0.0, h2, 1024), np.sqrt(np.linspace(0.0, h2, 1024))) if use_lut else (np.zeros(1), np.zeros(1))

            particle_centered_density_triangular(self.particles, density_map,
                                                 x_centers, y_centers, z_centers,
                                                 r_idx,
                                                 box_x, box_y, box_z,
                                                 h, h2, constant,
                                                 use_lut, table_x, table_y)

        elif kernel_func.__name__ == "uniform":
            constant = 1.0 / ((4.0 / 3.0) * np.pi * h ** 3)

            particle_centered_density_uniform(self.particles, density_map,
                                              x_centers, y_centers, z_centers,
                                              r_idx,
                                              box_x, box_y, box_z,
                                              h2, constant)
        else:
            raise NotImplementedError(f"Unsupported kernel: {kernel_func.__name__}")

        # ✅ Post-normalization to match total particle count
        sum_before = np.sum(density_map)
        if sum_before > 0:
            correction_factor = total_particles / sum_before
            density_map *= correction_factor
            logger.info(f"Density map normalized: ∑ρ = {sum_before:.6f} → {total_particles} (factor = {correction_factor:.6f})")
        else:
            logger.warning("Density map has zero sum. Normalization skipped.")

        return x_centers, y_centers, z_centers, density_map
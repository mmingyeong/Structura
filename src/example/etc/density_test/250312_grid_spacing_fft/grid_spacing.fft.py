#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import time

# GPU FFT 가속을 위해 cupy 임포트 시도
try:
    import cupy as cp
    use_gpu_fft = True
except ImportError:
    use_gpu_fft = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Parameters
# --------------------------
DATASET_SIZE = 50000    # Number of 2D data points
L = 100.0               # Domain: [0, L]^2
h = 1.0                 # Kernel bandwidth
RESULTS_DIR = "mod_res" # Folder to save result images

NONUNIFORM_FILE = "particles_nonuniform.npy"
UNIFORM_FILE = "particles_uniform.npy"

def generate_sine_dataset_2d(n_points, L):
    """
    Generates a non-uniform dataset based on a sine function.
    """
    x_vals = np.linspace(0, L, 10000, dtype=np.float64)
    F_vals = x_vals / L + (1 - np.cos(2 * np.pi * x_vals / L)) / (2 * np.pi)

    def sample_axis(n):
        u = np.random.rand(n).astype(np.float64)
        return np.interp(u, F_vals, x_vals)

    x = sample_axis(n_points)
    y = sample_axis(n_points)
    return np.column_stack((x, y)).astype(np.float64)

def generate_uniform_dataset_2d(n_points, L):
    """
    Generates a uniformly distributed dataset.
    """
    x = np.random.uniform(0, L, n_points).astype(np.float64)
    y = np.random.uniform(0, L, n_points).astype(np.float64)
    return np.column_stack((x, y)).astype(np.float64)

def gaussian_kernel(r, h):
    """
    2D Gaussian kernel (normalized)
    K(r;h) = (1/(2πh²)) * exp( - r²/(2h²) )
    """
    norm = 1.0 / (2 * np.pi * h**2)
    return norm * np.exp(-0.5 * (r / h)**2)

class FFTKDE2D:
    def __init__(self, particles, grid_bounds, grid_spacing, kernel_func, h):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.kernel_func = kernel_func
        self.h = h

    def compute_density(self):
        """
        FFT-based density computation (Periodic):
        1. Create a histogram on the grid.
        2. Normalize the histogram by the cell area to obtain a density field
           (units: particles per unit area).
        3. Compute FFT of the density field, multiply with the analytical Fourier transform
           of the Gaussian kernel, and perform inverse FFT.
        4. The resulting density field (in units: particles per unit area) integrates to the total mass.
        GPU acceleration is applied if cupy is available.
        """
        from numpy.fft import fft2, ifft2

        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing

        # Generate histogram on the grid
        x_edges = np.arange(xmin, xmax + dx, dx, dtype=np.float64)
        y_edges = np.arange(ymin, ymax + dy, dy, dtype=np.float64)
        H, xed, yed = np.histogram2d(self.particles[:, 0], self.particles[:, 1],
                                     bins=[x_edges, y_edges])
        nx, ny = H.shape
        x_centers = (xed[:-1] + xed[1:]) / 2
        y_centers = (yed[:-1] + yed[1:]) / 2

        # Normalize the histogram by the cell area to obtain density (particles per unit area)
        cell_area = dx * dy
        density_hist = H / cell_area

        # Fourier transform of the Gaussian kernel for periodic domain
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        kernel_fft = np.exp(-2 * (np.pi**2) * (self.h**2) * (KX**2 + KY**2))

        if use_gpu_fft:
            density_gpu = cp.asarray(density_hist)
            density_fft_gpu = cp.fft.fft2(density_gpu)
            conv_result_gpu = density_fft_gpu * cp.asarray(kernel_fft)
            density_fft = cp.asnumpy(cp.fft.ifft2(conv_result_gpu).real)
        else:
            density_fft = np.abs(ifft2(fft2(density_hist) * kernel_fft).real)

        return x_centers, y_centers, density_fft.astype(np.float64)

def compare_fft_density_maps_by_grid_spacing(particles, dataset_type, grid_spacings):
    """
    Computes FFT-based density maps for a set of grid spacings and saves a side-by-side comparison image.
    오직 FFT 밀도 지도 결과만을 저장합니다.
    """
    results = []
    for spacing in grid_spacings:
        logger.info("Computing FFT density map for grid spacing: %s", spacing)
        fft_calc = FFTKDE2D(
            particles,
            grid_bounds={'x': (0, L), 'y': (0, L)},
            grid_spacing=spacing,
            kernel_func=gaussian_kernel,
            h=h
        )
        start_time = time.time()
        x_centers, y_centers, density_fft = fft_calc.compute_density()
        elapsed_time = time.time() - start_time
        # Integrated mass is now computed as: sum(density) * cell_area = total particles
        cell_area = spacing[0] * spacing[1]
        total_mass = np.sum(density_fft) * cell_area
        logger.info("Grid spacing %s: FFT time: %.4f s, density shape: %s, Total mass: %.4f", 
                    spacing, elapsed_time, density_fft.shape, total_mass)
        results.append((spacing, x_centers, y_centers, density_fft, elapsed_time, total_mass))
    
    # Plot all FFT density maps side-by-side for visual comparison
    n_plots = len(grid_spacings)
    plt.figure(figsize=(5 * n_plots, 4), dpi=200)
    for i, (spacing, x_centers, y_centers, density_fft, elapsed_time, total_mass) in enumerate(results):
        plt.subplot(1, n_plots, i+1)
        plt.imshow(density_fft, cmap='viridis', interpolation='bicubic')
        plt.colorbar()
        plt.title(f"Grid: {spacing}\nTime: {elapsed_time:.2f}s\nMass: {total_mass:.2f}")
    plt.tight_layout()
    filename = os.path.join(RESULTS_DIR, f"fft_density_maps_{dataset_type}.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    
    return results

def load_or_generate_dataset(filename, generator_func, dataset_type):
    """
    Loads dataset from file if it exists; otherwise, generates a new dataset.
    """
    if os.path.exists(filename):
        logger.info("Loading %s particles from file: %s", dataset_type, filename)
        particles = np.load(filename, mmap_mode='r')
    else:
        logger.info("No %s data found. Generating new %s dataset.", dataset_type, dataset_type)
        particles = generator_func(DATASET_SIZE, L)
        np.save(filename, particles)
        logger.info("Saved %s particles to file: %s", dataset_type, filename)
    return particles

def main():
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Define grid spacings to compare
    grid_spacings = [(1, 1), (0.2, 0.2), (0.02, 0.02)]
    
    # Process non-uniform dataset
    particles_nonuniform = load_or_generate_dataset(NONUNIFORM_FILE, generate_sine_dataset_2d, "nonuniform")
    logger.info("Non-uniform particles shape: %s", particles_nonuniform.shape)
    compare_fft_density_maps_by_grid_spacing(particles_nonuniform, "nonuniform", grid_spacings)
    
    # Process uniform dataset
    particles_uniform = load_or_generate_dataset(UNIFORM_FILE, generate_uniform_dataset_2d, "uniform")
    logger.info("Uniform particles shape: %s", particles_uniform.shape)
    compare_fft_density_maps_by_grid_spacing(particles_uniform, "uniform", grid_spacings)

if __name__ == "__main__":
    main()

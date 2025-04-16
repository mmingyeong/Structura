#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fft_kde.py

The FFTKDE class performs 3D kernel density estimation using an FFT-based convolution 
approach, entirely on the CPU.

Particle data are first binned using numpy.histogramdd, and then the density map is computed 
by convolving with an FFT-transformed kernel using NumPy’s FFT functions.
This version uses Dask for out-of-core processing so that large arrays are handled 
in blocks while keeping the original data types (float64 / complex128).

Usage:
    fftkde = FFTKDE(particles, grid_bounds, grid_spacing, kernel_func=KernelFunctions.uniform, h=1.0)
    x_centers, y_centers, z_centers, density_conv = fftkde.compute_density()
"""

import os
import numpy as np
from logger import logger
from kernel import KernelFunctions
from numpy.lib.format import open_memmap
import dask.array as da
import dask.array.fft as da_fft

class FFTKDE:
    """
    The FFTKDE class performs 3D kernel density estimation via FFT-based convolution.
    This version uses Dask for out-of-core processing.
    """
    def __init__(self, particles, grid_bounds, grid_spacing, kernel_func=KernelFunctions.gaussian, h=1.0):
        """
        Initialize the FFTKDE object.
        
        Parameters
        ----------
        particles : np.ndarray
            (N, 3) array of particle coordinates.
        grid_bounds : dict
            Grid bounds: {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}.
        grid_spacing : tuple
            (dx, dy, dz) specifying grid cell sizes.
        kernel_func : callable, optional
            Kernel function to use (default: KernelFunctions.gaussian).
        h : float, optional
            Kernel bandwidth (default: 1.0).
        """
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.kernel_func = kernel_func
        self.h = h

        # Temporary directory for memory-mapped result storage.
        self.temp_dir = "tmp_fftkde"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        logger.debug("FFTKDE object created: grid_bounds=%s, grid_spacing=%s, h=%.3f",
                     grid_bounds, grid_spacing, h)

    def _compute_histogram(self):
        """
        Compute a 3D particle density histogram using numpy.histogramdd.
        
        Returns
        -------
        H : np.ndarray
            The 3D density histogram (dtype=int64).
        edges : list of np.ndarray
            Bin edge arrays for each axis.
        """
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        zmin, zmax = self.grid_bounds['z']
        dx, dy, dz = self.grid_spacing

        # Create bin edges ensuring the upper boundary is included.
        x_edges = np.arange(xmin, xmax + dx, dx)
        y_edges = np.arange(ymin, ymax + dy, dy)
        z_edges = np.arange(zmin, zmax + dz, dz)
        bins = [x_edges, y_edges, z_edges]

        H, edges = np.histogramdd(self.particles, bins=bins)
        return H, edges

    def compute_density(self):
        """
        Compute the density map using FFT-based convolution on the CPU with Dask for out-of-core processing.
        The final density map is stored via memory mapping.
        
        Returns
        -------
        x_centers, y_centers, z_centers : np.ndarray
            The center coordinates of each grid cell.
        density_conv : np.ndarray
            The computed 3D density map (loaded from a memory-mapped file, dtype=float64).
        """
        # 1. Compute the particle histogram.
        H, edges = self._compute_histogram()
        x_edges, y_edges, z_edges = edges
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        nx, ny, nz = H.shape

        dx, dy, dz = self.grid_spacing

        # 2. Convert H to a Dask array and rechunk so that each axis is a single chunk.
        # Dask FFT functions require each axis to be a single chunk.
        H_dask = da.from_array(H, chunks=H.shape)

        # 3. Compute FFT frequencies using full-chunk Dask arrays.
        x_freq = np.fft.fftfreq(nx, d=dx) * nx * dx
        y_freq = np.fft.fftfreq(ny, d=dy) * ny * dy
        z_freq = np.fft.fftfreq(nz, d=dz) * nz * dz
        x_kernel = da.from_array(x_freq, chunks=(nx,))
        y_kernel = da.from_array(y_freq, chunks=(ny,))
        z_kernel = da.from_array(z_freq, chunks=(nz,))
        X, Y, Z = da.meshgrid(x_kernel, y_kernel, z_kernel, indexing='ij')
        R = da.sqrt(X**2 + Y**2 + Z**2)

        # 4. Compute the kernel grid using the provided kernel function.
        # Assume kernel_func is elementwise and supports Dask arrays.
        kernel_grid = self.kernel_func(R, self.h)
        kernel_grid = kernel_grid.rechunk(kernel_grid.shape)

        # 5. Compute FFTs using Dask.
        H_fft = da_fft.fftn(H_dask)
        kernel_fft = da_fft.fftn(kernel_grid)
        density_fft = H_fft * kernel_fft

        # 6. Compute the inverse FFT and take the real part.
        density_conv_dask = da_fft.ifftn(density_fft).real

        # 7. Evaluate the Dask graph using the threads scheduler.
        # This converts the lazy HighLevelGraph into a concrete NumPy array.
        density_conv = density_conv_dask.compute(scheduler='threads')

        # 8. Store the computed density map into a memory-mapped file.
        temp_file = os.path.join(self.temp_dir, "density_conv_tmp.npy")
        target = open_memmap(temp_file, mode='w+', dtype=np.float64, shape=(nx, ny, nz))
        target[:] = density_conv[:]

        # 9. Load the memory-mapped result.
        density_mm = np.lib.format.open_memmap(temp_file, mode='r', dtype=np.float64, shape=(nx, ny, nz))

        logger.debug("FFT-based kernel density estimation complete (histogram shape: %s)", H.shape)
        return x_centers, y_centers, z_centers, density_mm

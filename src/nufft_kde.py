#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nufft_kde.py (optimized)

Optimized NUFFTKDE class for high-performance 2D/3D kernel density estimation using NUFFT.
Includes safe kernel FT computation, and efficient logging.
"""

import os
import numpy as np
import finufft
from logger import logger
from kernel import KernelFunctions

#from numpy.fft import fftfreq, ifftn, ifftshift
from scipy.fft import fftfreq, ifftn, ifftshift


class NUFFTKDE:
    _kernel_ft_cache = {}  # global cache for (kernel_func, h, grid_shape)
    _sinc_lut_x = np.linspace(-20 * np.pi, 20 * np.pi, 10000)
    _sinc_lut_y = np.sinc(_sinc_lut_x / np.pi)

    def __init__(self, particles, grid_bounds, grid_spacing, kernel_func=KernelFunctions.triangular, h=1.0):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.kernel_func = kernel_func
        self.h = h

        self.dim = particles.shape[1]
        if self.dim not in (2, 3):
            raise ValueError("Particles must be a 2D (N,2) or 3D (N,3) array.")

        if self.dim == 2:
            self.nx = int((grid_bounds['x'][1] - grid_bounds['x'][0]) / grid_spacing[0])
            self.ny = int((grid_bounds['y'][1] - grid_bounds['y'][0]) / grid_spacing[1])
            self.Lx, self.Ly = grid_bounds['x'][1] - grid_bounds['x'][0], grid_bounds['y'][1] - grid_bounds['y'][0]
            self.grid_shape = (self.nx, self.ny)
            self.Ls = (self.Lx, self.Ly)
        else:
            self.nx = int((grid_bounds['x'][1] - grid_bounds['x'][0]) / grid_spacing[0])
            self.ny = int((grid_bounds['y'][1] - grid_bounds['y'][0]) / grid_spacing[1])
            self.nz = int((grid_bounds['z'][1] - grid_bounds['z'][0]) / grid_spacing[2])
            self.Lx, self.Ly, self.Lz = (grid_bounds['x'][1] - grid_bounds['x'][0],
                                         grid_bounds['y'][1] - grid_bounds['y'][0],
                                         grid_bounds['z'][1] - grid_bounds['z'][0])
            self.grid_shape = (self.nx, self.ny, self.nz)
            self.Ls = (self.Lx, self.Ly, self.Lz)

    def _scale_particles(self):
        scaled_coords = []
        axes = ['x', 'y'] if self.dim == 2 else ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            center = sum(self.grid_bounds[axis]) / 2.0
            L_axis = self.grid_bounds[axis][1] - self.grid_bounds[axis][0]
            scaled = (self.particles[:, i] - center) * (2 * np.pi / L_axis)
            scaled_coords.append(scaled)
        return tuple(scaled_coords)

    def _sinc_lut(self, x):
        return np.interp(x, self._sinc_lut_x, self._sinc_lut_y, left=0.0, right=0.0)

    def _compute_kernel_fourier(self, kx, ky, kz=None):
        cache_key = (self.kernel_func.__name__, self.h, self.grid_shape)
        if cache_key in self._kernel_ft_cache:
            return self._kernel_ft_cache[cache_key]

        h_half = self.h / 2
        if self.kernel_func == KernelFunctions.triangular:
            def ft_term(k): return self._sinc_lut(k * h_half)**2
        elif self.kernel_func == KernelFunctions.uniform:
            def ft_term(k): return self._sinc_lut(k * h_half)
        else:
            raise ValueError("Unknown kernel function for NUFFTKDE.")

        if self.dim == 2:
            result = ft_term(kx) * ft_term(ky)
        else:
            result = ft_term(kx) * ft_term(ky) * ft_term(kz)

        self._kernel_ft_cache[cache_key] = result
        return result

    def compute_density(self):
        scaled_pts = self._scale_particles()

        plan = finufft.Plan(1, self.grid_shape, isign=-1, eps=1e-6)
        plan.setpts(*scaled_pts)

        c = np.ones(self.particles.shape[0], dtype=np.complex128) / self.particles.shape[0]
        F_k = plan.execute(c).reshape(self.grid_shape)
        F_k = ifftshift(F_k)

        freqs = [fftfreq(n, d=1.0) * n for n in self.grid_shape]
        if self.dim == 2:
            kx, ky = np.meshgrid(freqs[0], freqs[1], indexing='ij')
            kernel_ft = self._compute_kernel_fourier((2 * np.pi / self.Lx) * kx,
                                                     (2 * np.pi / self.Ly) * ky)
            density_complex = ifftn(F_k * kernel_ft)
        else:
            kx, ky, kz = np.meshgrid(freqs[0], freqs[1], freqs[2], indexing='ij')
            kernel_ft = self._compute_kernel_fourier((2 * np.pi / self.Lx) * kx,
                                                     (2 * np.pi / self.Ly) * ky,
                                                     (2 * np.pi / self.Lz) * kz)
            density_complex = ifftn(F_k * kernel_ft)

        density_map = np.real(density_complex)

        # Remove negative artifacts
        density_map = np.clip(density_map, 0, None)

        # Normalize to match particle count
        density_map *= (self.particles.shape[0] / np.sum(density_map))



        axes = ['x', 'y'] if self.dim == 2 else ['x', 'y', 'z']
        centers = [np.linspace(self.grid_bounds[axis][0] + self.grid_spacing[i] / 2,
                               self.grid_bounds[axis][1] - self.grid_spacing[i] / 2,
                               self.grid_shape[i]) for i, axis in enumerate(axes)]

        return (*centers, density_map)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# 파라미터
# --------------------------
DATASET_SIZE = 20000  # 2D 데이터 개수
L = 100.0             # 도메인: [0, L]^2
grid_spacing = (1.0, 1.0)  # 격자 해상도 -> 100×100 격자
h = 1.0               # 커널 밴드위스

def generate_sine_dataset_2d(n_points, L):
    x_vals = np.linspace(0, L, 10000, dtype=np.float32)
    F_vals = x_vals / L + (1 - np.cos(2*np.pi*x_vals / L)) / (2*np.pi)
    def sample_axis(n):
        u = np.random.rand(n).astype(np.float32)
        return np.interp(u, F_vals, x_vals)
    x = sample_axis(n_points)
    y = sample_axis(n_points)
    return np.column_stack((x, y)).astype(np.float32)

def gaussian_kernel(r, h):
    return np.exp(-0.5*(r/h)**2)

class DensityCalculator2D:
    def __init__(self, particles, grid_bounds, grid_spacing):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing

    def calculate_density_map(self, kernel_func, h):
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing
        x_centers = np.arange(xmin+dx/2, xmax, dx)
        y_centers = np.arange(ymin+dy/2, ymax, dy)
        nx, ny = len(x_centers), len(y_centers)
        density_map = np.zeros((nx, ny), dtype=np.float32)

        # 모든 격자점에 대해 (간단히 직렬 계산)
        for i, xc in enumerate(x_centers):
            for j, yc in enumerate(y_centers):
                diff_x = xc - self.particles[:, 0]
                diff_y = yc - self.particles[:, 1]
                dist = np.sqrt(diff_x**2 + diff_y**2)
                vals = kernel_func(dist, h)
                density_map[i, j] = vals.sum()
        return x_centers, y_centers, density_map

class FFTKDE2D:
    def __init__(self, particles, grid_bounds, grid_spacing, kernel_func, h):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.kernel_func = kernel_func
        self.h = h

    def compute_density(self):
        from numpy.fft import fft2, ifft2
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing

        x_edges = np.arange(xmin, xmax+dx, dx)
        y_edges = np.arange(ymin, ymax+dy, dy)
        H, xed, yed = np.histogram2d(self.particles[:,0], self.particles[:,1],
                                     bins=[x_edges, y_edges])
        nx, ny = H.shape
        x_centers = (xed[:-1] + xed[1:]) / 2
        y_centers = (yed[:-1] + yed[1:]) / 2

        kx = np.fft.fftfreq(nx, d=dx)*nx*dx
        ky = np.fft.fftfreq(ny, d=dy)*ny*dy
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        R = np.sqrt(KX**2 + KY**2)
        kernel_grid = self.kernel_func(R, self.h)

        H_fft = fft2(H)
        kernel_fft = fft2(kernel_grid)
        density_fft = H_fft * kernel_fft
        density_map = ifft2(density_fft).real.astype(np.float32)
        return x_centers, y_centers, density_map

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def compute_mae(a, b):
    return np.mean(np.abs(a - b))

def compute_pearson(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]

def plot_density_maps(direct_map, fft_map, diff_map, filename="density_maps_2d.png"):
    vmin = min(direct_map.min(), fft_map.min())
    vmax = max(direct_map.max(), fft_map.max())
    diff_min, diff_max = diff_map.min(), diff_map.max()
    max_abs = max(abs(diff_min), abs(diff_max))

    plt.figure(figsize=(14, 6), dpi=200)  # figure 크기와 dpi를 높여서 세밀하게
    plt.subplot(1,3,1)
    plt.title("DensityCalculator2D (Direct)")
    plt.imshow(direct_map, cmap='viridis', interpolation='bicubic',
               vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title("FFTKDE2D (FFT-based)")
    plt.imshow(fft_map, cmap='viridis', interpolation='bicubic',
               vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title("Difference (Direct - FFT)")
    plt.imshow(diff_map, cmap='bwr', interpolation='bicubic',
               vmin=-max_abs, vmax=max_abs)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---------------------------
# 산점도(Scatter Plot) 추가
# ---------------------------
def plot_scatter_comparison_values(direct_map, fft_map, filename="scatter_density_values.png"):
    """
    Direct와 FFT로 계산된 2D 맵을 평탄화(flatten)하여,
    각 셀의 밀도값을 산점도로 비교합니다.
    """
    direct_flat = direct_map.flatten()
    fft_flat = fft_map.flatten()

    plt.figure(figsize=(6,6), dpi=200)
    plt.scatter(direct_flat, fft_flat, s=2, alpha=0.5)
    plt.title("Scatter of Density Values: Direct vs. FFT")
    plt.xlabel("Direct Density")
    plt.ylabel("FFT Density")

    # y=x 기준선
    lim_min = min(direct_flat.min(), fft_flat.min())
    lim_max = max(direct_flat.max(), fft_flat.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    logger.info("2D Sine dataset: n=%d, domain=[0,%.1f]^2", DATASET_SIZE, L)
    particles = generate_sine_dataset_2d(DATASET_SIZE, L)
    logger.info("Particles shape: %s", particles.shape)

    # Direct
    calc2d = DensityCalculator2D(particles, {'x':(0,L), 'y':(0,L)}, grid_spacing)
    x_centers_direct, y_centers_direct, density_direct = calc2d.calculate_density_map(
        kernel_func=gaussian_kernel, h=h
    )

    # FFT-based
    fft2d = FFTKDE2D(particles, {'x':(0,L), 'y':(0,L)}, grid_spacing,
                     kernel_func=gaussian_kernel, h=h)
    x_centers_fft, y_centers_fft, density_fft = fft2d.compute_density()

    # Align shapes
    nx_d, ny_d = density_direct.shape
    nx_f, ny_f = density_fft.shape
    nx_c = min(nx_d, nx_f)
    ny_c = min(ny_d, ny_f)
    dmap_direct = density_direct[:nx_c, :ny_c]
    dmap_fft = density_fft[:nx_c, :ny_c]

    # 비교 지표
    rmse_val = compute_rmse(dmap_direct, dmap_fft)
    mae_val = compute_mae(dmap_direct, dmap_fft)
    corr_val = compute_pearson(dmap_direct, dmap_fft)
    logger.info("RMSE=%.4f, MAE=%.4f, Pearson=%.4f", rmse_val, mae_val, corr_val)

    diff_map = dmap_direct - dmap_fft
    plot_density_maps(dmap_direct, dmap_fft, diff_map, filename="density_maps_2d.png")

    # 새로 추가한 산점도
    plot_scatter_comparison_values(dmap_direct, dmap_fft, filename="scatter_density_values.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import time  # 실행시간 측정을 위한 모듈 추가
from tqdm import tqdm  # added progress bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# 파라미터
# --------------------------
DATASET_SIZE = 50000  # 2D 데이터 개수
L = 100.0             # 도메인: [0, L]^2
grid_spacing = (1, 1)  # 격자 해상도 -> 100×100 격자
h = 1.0               # 커널 밴드위스
PARTICLES_FILE = "particles.npy"  # 데이터 저장 파일 이름
RESULTS_DIR = "res"   # 결과 이미지 저장 폴더

# 파일 이름들: 각 데이터셋에 대해 별도의 파일로 저장
NONUNIFORM_FILE = "particles_nonuniform.npy"
UNIFORM_FILE = "particles_uniform.npy"

def generate_sine_dataset_2d(n_points, L):
    """
    비균일 분포 데이터 생성기 (sine 기반)
    """
    x_vals = np.linspace(0, L, 10000, dtype=np.float32)
    F_vals = x_vals / L + (1 - np.cos(2*np.pi*x_vals / L)) / (2*np.pi)
    def sample_axis(n):
        u = np.random.rand(n).astype(np.float32)
        return np.interp(u, F_vals, x_vals)
    x = sample_axis(n_points)
    y = sample_axis(n_points)
    return np.column_stack((x, y)).astype(np.float32)

def generate_uniform_dataset_2d(n_points, L):
    """
    균일 분포 데이터 생성기
    """
    x = np.random.uniform(0, L, n_points).astype(np.float32)
    y = np.random.uniform(0, L, n_points).astype(np.float32)
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
        for i, xc in enumerate(tqdm(x_centers, desc="Direct Density Calculation")):
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

    plt.figure(figsize=(14, 6), dpi=200)
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
    # Save the figure in the specified directory
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

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
    # Save the figure in the specified directory
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def process_dataset(particles, dataset_type):
    """
    각 데이터셋에 대해 direct 방식과 FFT 방식의 밀도 맵을 계산 및 비교하고,
    결과 이미지와 오차 지표를 저장 및 출력합니다.
    또한, 두 방법의 실행시간을 측정하여 로깅합니다.
    """
    logger.info("Processing %s dataset...", dataset_type)
    
    # Direct method 실행시간 측정
    start_direct = time.time()
    calc2d = DensityCalculator2D(particles, {'x': (0, L), 'y': (0, L)}, grid_spacing)
    x_centers_direct, y_centers_direct, density_direct = calc2d.calculate_density_map(
        kernel_func=gaussian_kernel, h=h
    )
    end_direct = time.time()
    direct_elapsed = end_direct - start_direct
    logger.info("Direct method elapsed time: %.4f seconds", direct_elapsed)
    
    # FFT-based method 실행시간 측정
    start_fft = time.time()
    fft2d = FFTKDE2D(particles, {'x': (0, L), 'y': (0, L)}, grid_spacing,
                      kernel_func=gaussian_kernel, h=h)
    x_centers_fft, y_centers_fft, density_fft = fft2d.compute_density()
    end_fft = time.time()
    fft_elapsed = end_fft - start_fft
    logger.info("FFT-based method elapsed time: %.4f seconds", fft_elapsed)

    # 두 결과의 격자 크기 맞춤 (가능한 공통 부분만 사용)
    nx_d, ny_d = density_direct.shape
    nx_f, ny_f = density_fft.shape
    nx_c = min(nx_d, nx_f)
    ny_c = min(ny_d, ny_f)
    dmap_direct = density_direct[:nx_c, :ny_c]
    dmap_fft = density_fft[:nx_c, :ny_c]

    # 비교 지표 계산
    rmse_val = compute_rmse(dmap_direct, dmap_fft)
    mae_val = compute_mae(dmap_direct, dmap_fft)
    corr_val = compute_pearson(dmap_direct, dmap_fft)
    logger.info("%s dataset: RMSE=%.4f, MAE=%.4f, Pearson=%.4f",
                dataset_type, rmse_val, mae_val, corr_val)

    diff_map = dmap_direct - dmap_fft
    density_filename = f"density_maps_2d_{dataset_type}.png"
    scatter_filename = f"scatter_density_values_{dataset_type}.png"
    # Save density plots
    plot_density_maps(dmap_direct, dmap_fft, diff_map, filename=density_filename)
    plot_scatter_comparison_values(dmap_direct, dmap_fft, filename=scatter_filename)
    
    # Save raw density maps as .npy files
    np.save(os.path.join(RESULTS_DIR, f"density_direct_{dataset_type}.npy"), dmap_direct)
    np.save(os.path.join(RESULTS_DIR, f"density_fft_{dataset_type}.npy"), dmap_fft)

def main():
    # 결과 저장 폴더 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 비균일 데이터 처리
    if os.path.exists(NONUNIFORM_FILE):
        logger.info("Loading non-uniform particles from file: %s", NONUNIFORM_FILE)
        particles_nonuniform = np.load(NONUNIFORM_FILE)
    else:
        logger.info("No non-uniform data found. Generating new non-uniform dataset.")
        particles_nonuniform = generate_sine_dataset_2d(DATASET_SIZE, L)
        np.save(NONUNIFORM_FILE, particles_nonuniform)
        logger.info("Saved non-uniform particles to file: %s", NONUNIFORM_FILE)
    logger.info("Non-uniform particles shape: %s", particles_nonuniform.shape)
    process_dataset(particles_nonuniform, "nonuniform")

    # 균일 데이터 처리
    if os.path.exists(UNIFORM_FILE):
        logger.info("Loading uniform particles from file: %s", UNIFORM_FILE)
        particles_uniform = np.load(UNIFORM_FILE)
    else:
        logger.info("No uniform data found. Generating new uniform dataset.")
        particles_uniform = generate_uniform_dataset_2d(DATASET_SIZE, L)
        np.save(UNIFORM_FILE, particles_uniform)
        logger.info("Saved uniform particles to file: %s", UNIFORM_FILE)
    logger.info("Uniform particles shape: %s", particles_uniform.shape)
    process_dataset(particles_uniform, "uniform")

if __name__ == "__main__":
    main()

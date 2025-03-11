#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script generates a synthetic dataset based on a sine-modulated density function,
computes density maps using two methods (DensityCalculator with optional GPU/Numba
and FFT-based estimation via FFTKDE), then compares the results both numerically and visually.
Comparison metrics (RMSE, MAE, Pearson correlation) are logged, and representative slices
and histograms are saved as PNG files.

Additionally, several plotting methods are provided including pcolormesh-based visualization.
"""

import sys
import os
# 상위 디렉토리 (src)를 모듈 검색 경로에 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import LogNorm

from density import DensityCalculator
from fft_kde import FFTKDE
from kernel import KernelFunctions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------
# Dataset Size Parameter (Easy to Change)
# -------------------------------------
DATASET_SIZE = 5000  # 여기에서 테스트할 데이터셋 개수를 쉽게 수정할 수 있습니다.

# ----------------------------
# Metrics
# ----------------------------
def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def compute_mae(a, b):
    return np.mean(np.abs(a - b))

def compute_pearson(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]


# ----------------------------
# Original Plotting Functions
# ----------------------------
def plot_density_maps(slice_direct, slice_fft, diff_slice, filename="density_maps.png"):
    """
    DensityCalculator(Direct) 결과와 FFTKDE 결과를 같은 컬러 스케일로 표시하고,
    차이 맵(Difference)은 대칭 범위로 표시합니다.
    """

    # 두 결과에서 공통으로 사용할 컬러 범위를 계산
    vmin = min(slice_direct.min(), slice_fft.min())
    vmax = max(slice_direct.max(), slice_fft.max())

    # 차이 맵의 경우 ±max_abs로 대칭 범위 설정
    diff_min, diff_max = diff_slice.min(), diff_slice.max()
    max_abs = max(abs(diff_min), abs(diff_max))

    plt.figure(figsize=(12, 5))

    # 1) DensityCalculator (Direct)
    plt.subplot(1, 3, 1)
    plt.title("DensityCalculator (Direct)")
    plt.imshow(slice_direct, cmap='viridis', interpolation='bicubic',
               vmin=vmin, vmax=vmax)  # 공통 범위 적용
    plt.colorbar()

    # 2) FFTKDE
    plt.subplot(1, 3, 2)
    plt.title("FFTKDE (Overlapped)")
    plt.imshow(slice_fft, cmap='viridis', interpolation='bicubic',
               vmin=vmin, vmax=vmax)  # 공통 범위 적용
    plt.colorbar()

    # 3) Difference (Direct - FFTKDE)
    plt.subplot(1, 3, 3)
    plt.title("Difference (Direct - FFTKDE)")
    plt.imshow(diff_slice, cmap='bwr', interpolation='bicubic',
               vmin=-max_abs, vmax=max_abs)  # 대칭 범위
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



def plot_histograms(direct, fft, filename="histogram_comparison.png"):
    plt.figure(figsize=(10, 5))
    plt.hist(direct.flatten(), bins=50, alpha=0.5, label="Direct")
    plt.hist(fft.flatten(), bins=50, alpha=0.5, label="FFTKDE")
    plt.xlabel("Density Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram Comparison")
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_scatter_comparison(direct, fft, filename="scatter_comparison.png"):
    plt.figure(figsize=(8, 8))
    plt.scatter(direct.flatten(), fft.flatten(), alpha=0.5, s=1)
    plt.xlabel("Direct Density")
    plt.ylabel("FFT Density")
    plt.title("Scatter Plot: Direct vs FFT Density")
    lims = [min(direct.min(), fft.min()), max(direct.max(), fft.max())]
    plt.plot(lims, lims, 'r--', label="y=x")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_difference_histogram(direct, fft, filename="difference_histogram.png"):
    diff = direct - fft
    plt.figure(figsize=(8, 5))
    plt.hist(diff.flatten(), bins=50, alpha=0.7, color='gray')
    plt.xlabel("Difference (Direct - FFT)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Density Differences")
    plt.savefig(filename, dpi=300)
    plt.close()


# ----------------------------
# Additional Plotting Methods: pcolormesh
# ----------------------------
def plot_density_maps_pcolormesh(slice_direct, slice_fft, diff_slice, 
                                 x_centers, y_centers, filename="density_maps_pcolormesh.png",
                                 use_log=False):
    """
    pcolormesh를 사용해 2D 슬라이스를 같은 컬러 스케일로 표시합니다.
    Difference 맵은 ± 대칭 범위로 설정.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(x_centers) > 1:
        dx = x_centers[1] - x_centers[0]
    else:
        dx = 1.0
    if len(y_centers) > 1:
        dy = y_centers[1] - y_centers[0]
    else:
        dy = 1.0

    x_edges = np.concatenate(([x_centers[0] - dx/2], x_centers + dx/2))
    y_edges = np.concatenate(([y_centers[0] - dy/2], y_centers + dy/2))

    # 두 결과에서 공통으로 사용할 컬러 범위를 계산
    vmin_common = min(slice_direct.min(), slice_fft.min())
    vmax_common = max(slice_direct.max(), slice_fft.max())

    # 차이 맵의 경우 ± 대칭 범위
    diff_min, diff_max = diff_slice.min(), diff_slice.max()
    max_abs = max(abs(diff_min), abs(diff_max))

    plt.figure(figsize=(12, 5))

    # 1) Direct
    plt.subplot(1, 3, 1)
    plt.title("Direct (pcolormesh)")
    if use_log:
        eps = 1e-8
        slice_direct_log = np.maximum(slice_direct, eps)
        plt.pcolormesh(x_edges, y_edges, slice_direct_log, cmap='viridis', shading='auto',
                       norm=LogNorm(vmin=vmin_common, vmax=vmax_common))
    else:
        plt.pcolormesh(x_edges, y_edges, slice_direct, cmap='viridis', shading='auto',
                       vmin=vmin_common, vmax=vmax_common)
    plt.colorbar()

    # 2) FFT
    plt.subplot(1, 3, 2)
    plt.title("FFTKDE (pcolormesh)")
    if use_log:
        eps = 1e-8
        slice_fft_log = np.maximum(slice_fft, eps)
        plt.pcolormesh(x_edges, y_edges, slice_fft_log, cmap='viridis', shading='auto',
                       norm=LogNorm(vmin=vmin_common, vmax=vmax_common))
    else:
        plt.pcolormesh(x_edges, y_edges, slice_fft, cmap='viridis', shading='auto',
                       vmin=vmin_common, vmax=vmax_common)
    plt.colorbar()

    # 3) Difference
    plt.subplot(1, 3, 3)
    plt.title("Difference (pcolormesh)")
    plt.pcolormesh(x_edges, y_edges, diff_slice, cmap='bwr', shading='auto',
                   vmin=-max_abs, vmax=max_abs)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ----------------------------
# Synthetic Data Generator (Inverse Transform Sampling)
# ----------------------------
def generate_sine_dataset(n_points, L):
    """
    [0, L]^3 내에서 sine 함수 모양의 분포를 따르는 synthetic dataset을 생성합니다.
    각 축은 독립적으로 아래 분포를 따릅니다:
        p(x) = (sin(2*pi*x/L) + 1) / L,
    누적분포함수는:
        F(x) = x/L + (1 - cos(2*pi*x/L))/(2*pi)
    이를 inverse transform sampling (보간법)으로 샘플링합니다.
    """
    # x 값을 충분히 세분화하여 lookup table 생성 (예: 10000 포인트)
    x_vals = np.linspace(0, L, 10000, dtype=np.float32)
    F_vals = x_vals / L + (1 - np.cos(2 * np.pi * x_vals / L)) / (2 * np.pi)
    
    def sample_axis(n):
        u = np.random.uniform(0, 1, n).astype(np.float32)
        return np.interp(u, F_vals, x_vals)
    
    x = sample_axis(n_points)
    y = sample_axis(n_points)
    z = sample_axis(n_points)
    
    return np.stack((x, y, z), axis=1).astype(np.float32)


# ----------------------------
# Main Test Code
# ----------------------------
def main():
    # 도메인 및 파라미터 설정
    L = 205.0  # 전체 도메인 크기
    n_particles = DATASET_SIZE  # DATASET_SIZE 상수를 사용하여 데이터셋 크기를 조절
    grid_bounds = {'x': (0, L), 'y': (0, L), 'z': (0, L)}
    grid_spacing = (4, 4, 4)  # coarser grid for faster testing
    h = 1.0

    logger.info("합성 sine-modulated dataset 생성: 입자 수 = %d, 도메인 = [0, %.1f]^3", n_particles, L)
    particles = generate_sine_dataset(n_particles, L)
    logger.info("생성된 데이터셋 shape: %s", particles.shape)

    # DensityCalculator를 이용한 직접 계산 방식
    logger.info("DensityCalculator를 이용하여 밀도 맵 계산 (직접 계산 방식)")
    density_calc = DensityCalculator(
        particles,
        grid_bounds,
        grid_spacing,
        use_gpu=True,     # GPU 사용 (GPU 환경이면 float32 사용)
        batch_size=2000   # CPU 모드 시 배치 크기
    )
    x_centers_direct, y_centers_direct, z_centers_direct, density_direct = density_calc.calculate_density_map(
        kernel_func=KernelFunctions.gaussian,
        h=h
    )

    # FFTKDE를 이용한 FFT 기반 방식
    logger.info("FFTKDE를 이용하여 밀도 맵 계산 (FFT 기반 방식)")
    fft_kde = FFTKDE(particles, grid_bounds, grid_spacing, kernel_func=KernelFunctions.gaussian, h=h)
    x_centers_fft, y_centers_fft, z_centers_fft, density_fft = fft_kde.compute_density()

    # 두 결과 격자 크기 맞추기 (예: DensityCalculator: (nx,ny,nz) vs. FFTKDE: (nx,ny,nz))
    target_shape = density_direct.shape
    density_fft_aligned = density_fft[:target_shape[0], :target_shape[1], :target_shape[2]]
    logger.info("DensityCalculator 결과 shape: %s, FFTKDE 결과 shape: %s -> aligned FFTKDE shape: %s",
                density_direct.shape, density_fft.shape, density_fft_aligned.shape)

    # 비교 지표 계산
    rmse_val = compute_rmse(density_direct, density_fft_aligned)
    mae_val = compute_mae(density_direct, density_fft_aligned)
    pearson_val = compute_pearson(density_direct, density_fft_aligned)
    logger.info("비교 결과:")
    logger.info("  RMSE: %.6f", rmse_val)
    logger.info("  MAE: %.6f", mae_val)
    logger.info("  Pearson Correlation: %.6f", pearson_val)

    # 중간 z 슬라이스
    mid_index = density_direct.shape[2] // 2
    slice_direct = density_direct[:, :, mid_index]
    slice_fft = density_fft_aligned[:, :, mid_index]
    diff_slice = slice_direct - slice_fft

    # 기존 플롯 (선형 스케일)
    plot_density_maps(slice_direct, slice_fft, diff_slice, filename="density_maps.png")

    # 추가 시각화: pcolormesh 기반 시각화
    plot_density_maps_pcolormesh(slice_direct, slice_fft, diff_slice, 
                                  x_centers_direct, y_centers_direct,
                                  filename="density_maps_pcolormesh.png",
                                  use_log=False)

    # 히스토그램, 산점도, 차이 히스토그램
    plot_histograms(density_direct, density_fft_aligned, filename="histogram_comparison.png")
    plot_scatter_comparison(density_direct, density_fft_aligned, filename="scatter_comparison.png")
    plot_difference_histogram(density_direct, density_fft_aligned, filename="difference_histogram.png")


if __name__ == '__main__':
    main()

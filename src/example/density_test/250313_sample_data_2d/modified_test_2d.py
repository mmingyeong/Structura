#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import cProfile
import pstats
from tqdm import tqdm  # progress bar 추가

# GPU FFT 및 GPU 직접 합산을 위해 cupy 임포트 시도
try:
    import cupy as cp
    from cupy.cuda import runtime
    use_gpu = True
except ImportError:
    use_gpu = False

from numba import njit, prange


def select_best_gpu():
    num_devices = runtime.getDeviceCount()
    best_device = 0
    best_score = 0
    for device in range(num_devices):
        props = runtime.getDeviceProperties(device)
        # 단순 평가: 멀티프로세서 수와 클럭 속도의 곱
        score = props['multiProcessorCount'] * props['clockRate']
        if score > best_score:
            best_score = score
            best_device = device
    return best_device

# 최적의 GPU 선택 및 할당
GPU_DEVICE = select_best_gpu()
print(f"Selected GPU: {GPU_DEVICE}")

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("application.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Parameters
# --------------------------
DATASET_SIZE = 50000    # 2D 데이터 포인트 개수
L = 100.0               # 도메인: [0, L]^2
grid_spacing = (0.002, 0.002)   # 그리드 해상도
h = 1.0                 # 커널 대역폭
RESULTS_DIR = "mod_res" # 결과 이미지 저장 폴더

NONUNIFORM_FILE = "particles_nonuniform.npy"
UNIFORM_FILE = "particles_uniform.npy"

# --------------------------
# Dataset Generators & Kernel
# --------------------------
def generate_sine_dataset_2d(n_points, L):
    x_vals = np.linspace(0, L, 10000, dtype=np.float64)
    F_vals = x_vals / L + (1 - np.cos(2 * np.pi * x_vals / L)) / (2 * np.pi)
    def sample_axis(n):
        u = np.random.rand(n).astype(np.float64)
        return np.interp(u, F_vals, x_vals)
    x = sample_axis(n_points)
    y = sample_axis(n_points)
    # 메모리 연속성 확보
    return np.ascontiguousarray(np.column_stack((x, y)).astype(np.float64))

def generate_uniform_dataset_2d(n_points, L):
    x = np.random.uniform(0, L, n_points).astype(np.float64)
    y = np.random.uniform(0, L, n_points).astype(np.float64)
    return np.ascontiguousarray(np.column_stack((x, y)).astype(np.float64))

def gaussian_kernel(r, h):
    norm = 1.0 / (2 * np.pi * h**2)
    return norm * np.exp(-0.5 * (r / h)**2)

# --------------------------
# Direct Method: Numba Version with Cutoff (CPU)
# --------------------------
@njit(parallel=True, fastmath=True)
def compute_density_direct_numba(particles, x_centers, y_centers, Lx, Ly, h, cutoff):
    nx = x_centers.shape[0]
    ny = y_centers.shape[0]
    density = np.zeros((nx, ny), dtype=np.float64)
    norm = 1.0 / (2.0 * np.pi * h * h)
    n_particles = particles.shape[0]
    cutoff2 = cutoff * cutoff
    inv_h2 = -0.5 / (h * h)  # 미리 계산하여 반복문 내부에서 사용
    for i in prange(nx):
        xc = x_centers[i]
        for j in range(ny):
            yc = y_centers[j]
            s = 0.0
            for k in range(n_particles):
                dx = xc - particles[k, 0]
                dy = yc - particles[k, 1]
                # 최소 이미지 방식 (주기적 경계조건)
                dx = (dx + Lx/2) % Lx - Lx/2
                dy = (dy + Ly/2) % Ly - Ly/2
                r2 = dx*dx + dy*dy
                if r2 <= cutoff2:
                    s += norm * np.exp(r2 * inv_h2)
            density[i, j] = s
    return density

# --------------------------
# Direct Method: GPU Vectorized Version with Cutoff (x-center 배치 처리)
# --------------------------
def compute_density_direct_gpu_batched(particles, x_centers, y_centers, Lx, Ly, h, cutoff,
                                         batch_size_x=3, batch_size_y=3):
    """
    GPU에서 cupy를 이용하여 x 및 y축을 모두 블록 단위로 처리.
    이중 배치 방식을 사용하여 큰 외적 계산 대신 메모리 부담을 줄입니다.
    """
    with cp.cuda.Device(GPU_DEVICE):
        cp.get_default_memory_pool().free_all_blocks()
        # 입력 배열의 메모리 연속성 확보 및 GPU로 옮김
        particles_gpu = cp.asarray(np.ascontiguousarray(particles))
        x_centers_gpu = cp.asarray(np.ascontiguousarray(x_centers))
        y_centers_gpu = cp.asarray(np.ascontiguousarray(y_centers))
        Lx_gpu = cp.float64(Lx)
        Ly_gpu = cp.float64(Ly)
        h2 = h * h
        nx = x_centers.shape[0]
        ny = y_centers.shape[0]
        norm = 1.0 / (2.0 * np.pi * h2)
        cutoff2 = cutoff * cutoff

        # 결과 저장용 GPU 배열 할당
        density_gpu = cp.empty((nx, ny), dtype=cp.float64)

        # x축과 y축을 각각 소규모 블록으로 나눠 처리
        for start_x in tqdm(range(0, nx, batch_size_x), desc="Processing GPU batches (x)", unit="batch"):
            end_x = min(start_x + batch_size_x, nx)
            block_x = x_centers_gpu[start_x:end_x]  # shape: (Bx,)
            # 각 x 블록에 대해 모든 입자와의 dx 계산
            dx_block = block_x[:, None] - particles_gpu[:, 0]  # shape: (Bx, n_particles)
            dx_block = (dx_block + Lx_gpu/2) % Lx_gpu - Lx_gpu/2

            # y축도 블록 처리
            for start_y in range(0, ny, batch_size_y):
                end_y = min(start_y + batch_size_y, ny)
                block_y = y_centers_gpu[start_y:end_y]  # shape: (By,)
                dy_block = block_y[:, None] - particles_gpu[:, 1]  # shape: (By, n_particles)
                dy_block = (dy_block + Ly_gpu/2) % Ly_gpu - Ly_gpu/2

                # dx_block: (Bx, n_particles) -> 확장: (Bx, 1, n_particles)
                # dy_block: (By, n_particles) -> 확장: (1, By, n_particles)
                # 두 배열의 제곱 합으로 (Bx, By, n_particles) 모양의 r2 계산
                r2 = dx_block[:, None, :]**2 + dy_block[None, :, :]**2

                # cutoff 조건 적용 및 기여도 계산
                mask = r2 <= cutoff2
                contributions = cp.where(mask, norm * cp.exp(-0.5 * r2 / h2), 0)
                density_block = cp.sum(contributions, axis=2)  # shape: (Bx, By)
                density_gpu[start_x:end_x, start_y:end_y] = density_block

                # 해당 y 배치 관련 임시 변수 해제 및 메모리 정리
                del block_y, dy_block, r2, mask, contributions, density_block
                cp.get_default_memory_pool().free_all_blocks()

            # x 배치 관련 임시 변수 해제
            del block_x, dx_block
            cp.get_default_memory_pool().free_all_blocks()

        # 최종 결과를 CPU 배열로 변환
        result = cp.asnumpy(density_gpu)
        del particles_gpu, x_centers_gpu, y_centers_gpu, density_gpu
        cp.get_default_memory_pool().free_all_blocks()
    return result


# --------------------------
# Density Calculator Classes
# --------------------------
class DensityCalculator2D:
    def __init__(self, particles, grid_bounds, grid_spacing):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing

    def calculate_density_map(self, kernel_func, h, cutoff):
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing
        x_centers = np.ascontiguousarray(np.arange(xmin + dx/2, xmax, dx, dtype=np.float64))
        y_centers = np.ascontiguousarray(np.arange(ymin + dy/2, ymax, dy, dtype=np.float64))
        Lx = xmax - xmin
        Ly = ymax - ymin
        density_map = compute_density_direct_numba(self.particles, x_centers, y_centers, Lx, Ly, h, cutoff)
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
        x_edges = np.ascontiguousarray(np.arange(xmin, xmax+dx, dx, dtype=np.float64))
        y_edges = np.ascontiguousarray(np.arange(ymin, ymax+dy, dy, dtype=np.float64))
        H, xed, yed = np.histogram2d(self.particles[:, 0], self.particles[:, 1],
                                     bins=[x_edges, y_edges])
        nx, ny = H.shape
        x_centers = (xed[:-1] + xed[1:]) / 2
        y_centers = (yed[:-1] + yed[1:]) / 2
        cell_area = dx * dy
        density_hist = H / cell_area
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        kernel_fft = np.exp(-2*(np.pi**2)*(self.h**2)*(KX**2 + KY**2))
        if use_gpu:
            with cp.cuda.Device(GPU_DEVICE):
                density_gpu = cp.asarray(density_hist)
                density_fft_gpu = cp.fft.fft2(density_gpu)
                conv_result_gpu = density_fft_gpu * cp.asarray(kernel_fft)
                density_fft = cp.asnumpy(cp.fft.ifft2(conv_result_gpu).real)
                del density_gpu, density_fft_gpu, conv_result_gpu
                cp.get_default_memory_pool().free_all_blocks()
        else:
            density_fft = np.abs(ifft2(fft2(density_hist) * kernel_fft).real)
        return x_centers, y_centers, density_fft.astype(np.float64)

# --------------------------
# Error Metrics & Plotting
# --------------------------
def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def compute_mae(a, b):
    return np.mean(np.abs(a - b))

def compute_pearson(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]

def plot_density_maps(direct_map, fft_map, diff_map, filename="density_maps_2d.png"):
    vmin = min(direct_map.min(), fft_map.min())
    vmax = max(direct_map.max(), fft_map.max())
    max_abs = max(abs(diff_map.min()), abs(diff_map.max()))
    plt.figure(figsize=(14, 6), dpi=200)
    plt.subplot(1, 3, 1)
    plt.title("DensityCalculator2D (Direct)")
    plt.imshow(direct_map, cmap='viridis', interpolation='bicubic', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("FFTKDE2D (FFT-based)")
    plt.imshow(fft_map, cmap='viridis', interpolation='bicubic', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Difference (Direct - FFT)")
    plt.imshow(diff_map, cmap='bwr', interpolation='bicubic', vmin=-max_abs, vmax=max_abs)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

def plot_scatter_comparison_values(direct_map, fft_map, filename="scatter_density_values.png"):
    direct_flat = direct_map.flatten()
    fft_flat = fft_map.flatten()
    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(direct_flat, fft_flat, s=2, alpha=0.5)
    plt.title("Scatter of Density Values: Direct vs. FFT")
    plt.xlabel("Direct Density")
    plt.ylabel("FFT Density")
    lim_min = min(direct_flat.min(), fft_flat.min())
    lim_max = max(direct_flat.max(), fft_flat.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# --------------------------
# Dataset Loader
# --------------------------
def load_or_generate_dataset(filename, generator_func, dataset_type):
    if os.path.exists(filename):
        logger.info("Loading %s particles from file: %s", dataset_type, filename)
        particles = np.load(filename, mmap_mode='r')
        particles = np.ascontiguousarray(particles)
    else:
        logger.info("No %s data found. Generating new %s dataset.", dataset_type, dataset_type)
        particles = generator_func(DATASET_SIZE, L)
        np.save(filename, particles)
        logger.info("Saved %s particles to file: %s", dataset_type, filename)
    return particles

# --------------------------
# Process Dataset (Direct & FFT Comparison)
# --------------------------
def process_dataset(particles, dataset_type):
    logger.info("Starting processing for %s dataset...", dataset_type)
    cutoff = 4 * h

    logger.info("Beginning Direct Method computation for %s dataset.", dataset_type)
    start_direct = time.time()
    xmin, xmax = 0.0, L
    ymin, ymax = 0.0, L
    dx, dy = grid_spacing
    x_centers = np.ascontiguousarray(np.arange(xmin + dx/2, xmax, dx, dtype=np.float64))
    y_centers = np.ascontiguousarray(np.arange(ymin + dy/2, ymax, dy, dtype=np.float64))
    Lx = xmax - xmin
    Ly = ymax - ymin

    if use_gpu:
        density_direct = compute_density_direct_gpu_batched(particles, x_centers, y_centers, Lx, Ly, h, cutoff)
    else:
        calc2d = DensityCalculator2D(particles, {'x': (0, L), 'y': (0, L)}, grid_spacing)
        _, _, density_direct = calc2d.calculate_density_map(gaussian_kernel, h, cutoff)
    direct_elapsed = time.time() - start_direct
    logger.info("Direct Method computation completed in %.4f seconds.", direct_elapsed)

    logger.info("Beginning FFT Method computation for %s dataset.", dataset_type)
    start_fft = time.time()
    fft2d = FFTKDE2D(particles, {'x': (0, L), 'y': (0, L)}, grid_spacing, gaussian_kernel, h)
    x_centers_fft, y_centers_fft, density_fft = fft2d.compute_density()
    fft_elapsed = time.time() - start_fft
    logger.info("FFT Method computation completed in %.4f seconds.", fft_elapsed)

    nx_d, ny_d = density_direct.shape
    nx_f, ny_f = density_fft.shape
    nx_c = min(nx_d, nx_f)
    ny_c = min(ny_d, ny_f)
    dmap_direct = density_direct[:nx_c, :ny_c]
    dmap_fft = density_fft[:nx_c, :ny_c]
    logger.info("Adjusted grid shape: Direct (%d, %d), FFT (%d, %d), common (%d, %d).",
                nx_d, ny_d, nx_f, ny_f, nx_c, ny_c)

    rmse_val = compute_rmse(dmap_direct, dmap_fft)
    mae_val = compute_mae(dmap_direct, dmap_fft)
    corr_val = compute_pearson(dmap_direct, dmap_fft)
    logger.info("%s dataset error metrics: RMSE = %.6f, MAE = %.6f, Pearson = %.6f.",
                dataset_type, rmse_val, mae_val, corr_val)

    diff_map = dmap_direct - dmap_fft
    density_filename = f"density_maps_2d_{dataset_type}.png"
    scatter_filename = f"scatter_density_values_{dataset_type}.png"
    logger.info("Plotting and saving density maps and scatter comparison for %s dataset.", dataset_type)
    plot_density_maps(dmap_direct, dmap_fft, diff_map, filename=density_filename)
    plot_scatter_comparison_values(dmap_direct, dmap_fft, filename=scatter_filename)

    np.save(os.path.join(RESULTS_DIR, f"density_direct_{dataset_type}.npy"), dmap_direct)
    np.save(os.path.join(RESULTS_DIR, f"density_fft_{dataset_type}.npy"), dmap_fft)
    logger.info("Saved density maps for %s dataset.", dataset_type)

# --------------------------
# Main
# --------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    datasets = [("non-uniform", NONUNIFORM_FILE, generate_sine_dataset_2d),
                ("uniform", UNIFORM_FILE, generate_uniform_dataset_2d)]
    for dtype, filename, gen_func in datasets:
        logger.info("Processing %s dataset.", dtype)
        particles = load_or_generate_dataset(filename, gen_func, dtype)
        logger.info("Dataset %s loaded with shape %s.", dtype, particles.shape)
        process_dataset(particles, dtype)
    logger.info("All datasets processed successfully.")

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    with open("profile_results.txt", "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats("cumtime")
        ps.print_stats()

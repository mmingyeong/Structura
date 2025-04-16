#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: density.py
#
# 최적화를 적용한 3차원 밀도 계산 코드 (입자 중심 방식 추가)
# - 벡터화/브로드캐스팅: CPU (Numpy) 및 GPU (CuPy) 모두 지원하도록 구현.
# - 병렬 처리: CPU 계산 시 배치 처리 및 cutoff 최적화를 적용하여 불필요한 연산을 줄입니다.
# - (추가) 입자 중심 접근법: Top-hat 커널 및 삼각형 커널에 대해 각 입자가 영향을 미치는 영역만 업데이트.
# - (추가) Numba JIT를 통해 CPU 연산을 가속합니다.
# - (추가) GPU 가속 시, 최적 GPU 선택, 비동기 스트림 및 에러/메모리 관리를 적용합니다.

import numpy as np
from logger import logger
from tqdm import tqdm

# GPU 관련 라이브러리
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from concurrent.futures import ProcessPoolExecutor

# 외부 모듈에서 커널 함수들을 import (예: kernel.py의 KernelFunctions)
from kernel import KernelFunctions

# Numba 사용 여부 (CPU 가속용)
USE_NUMBA = False
try:
    import numba
    USE_NUMBA = True
except ImportError:
    logger.info("ImportError for numba")
    pass

# =============================================================================
# GPU 관련 유틸리티 함수
# -----------------------------------------------------------------------------
def select_best_gpu():
    """
    사용 가능한 GPU 중 총 메모리(TOTAL_GLOBAL_MEM)가 가장 큰 GPU의 device id를 반환합니다.
    cp.cuda.Device(i).attributes를 사용해 GPU 속성을 확인합니다.
    """
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        logger.error("CUDA 초기화 오류: %s", e)
        return None

    best_device = 0
    best_mem = 0
    for i in range(device_count):
        # cp.cuda.Device(i).attributes: dict with keys such as 'TOTAL_GLOBAL_MEM'
        props = cp.cuda.Device(i).attributes
        total_mem = props.get('TOTAL_GLOBAL_MEM', 0)
        logger.info("GPU %d: Total Memory = %.2f GB", i, total_mem / (1024**3))
        if total_mem > best_mem:
            best_mem = total_mem
            best_device = i
    logger.info("최적 GPU 선택: GPU %d", best_device)
    return best_device

def log_gpu_memory_info(stage=""):
    """
    현재 선택된 GPU의 메모리 정보를 로깅합니다.
    """
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        logger.info("[%s] GPU Memory: Free = %.2f GB, Total = %.2f GB",
                    stage, free / (1024**3), total / (1024**3))
    except Exception as e:
        logger.error("GPU 메모리 정보 로깅 실패: %s", e)

# =============================================================================
# 입자 중심 방식 밀도 계산 함수 (Top-hat 커널)
# -----------------------------------------------------------------------------
if USE_NUMBA:
    @numba.njit(fastmath=True)
    def particle_centered_density(particles, density_map,
                                  x_min, x_max, dx, nx,
                                  y_min, y_max, dy, ny,
                                  z_min, zmax, dz, nz,
                                  h, inv_volume):
        """
        Numba 가속: Top-hat (uniform) 커널을 적용하여 입자 중심 밀도 계산.
        최적화를 위해 squared distance를 사용함.
        """
        box_size_x = x_max - x_min
        box_size_y = y_max - y_min
        box_size_z = zmax - z_min
        half_box_x = box_size_x / 2.0
        half_box_y = box_size_y / 2.0
        half_box_z = box_size_z / 2.0
        h2 = h * h  # squared bandwidth
        n_particles = particles.shape[0]
        for i in range(n_particles):
            px = particles[i, 0]
            py = particles[i, 1]
            pz = particles[i, 2]
            fx = (px - x_min) / dx
            fy = (py - y_min) / dy
            fz = (pz - z_min) / dz
            r_idx = int(np.ceil(h / dx))
            ix_min = max(0, int(np.floor(fx)) - r_idx)
            ix_max = min(nx - 1, int(np.floor(fx)) + r_idx)
            iy_min = max(0, int(np.floor(fy)) - r_idx)
            iy_max = min(ny - 1, int(np.floor(fy)) + r_idx)
            iz_min = max(0, int(np.floor(fz)) - r_idx)
            iz_max = min(nz - 1, int(np.floor(fz)) + r_idx)
            for ix in range(ix_min, ix_max + 1):
                x_center = x_min + (ix + 0.5) * dx
                diff_x = x_center - px
                if diff_x > half_box_x:
                    diff_x -= box_size_x
                elif diff_x < -half_box_x:
                    diff_x += box_size_x
                for iy in range(iy_min, iy_max + 1):
                    y_center = y_min + (iy + 0.5) * dy
                    diff_y = y_center - py
                    if diff_y > half_box_y:
                        diff_y -= box_size_y
                    elif diff_y < -half_box_y:
                        diff_y += box_size_y
                    for iz in range(iz_min, iz_max + 1):
                        z_center = z_min + (iz + 0.5) * dz
                        diff_z = z_center - pz
                        if diff_z > half_box_z:
                            diff_z -= box_size_z
                        elif diff_z < -half_box_z:
                            diff_z += box_size_z
                        # Compute squared distance to avoid costly sqrt
                        sq_dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                        if sq_dist <= h2:
                            density_map[ix, iy, iz] += inv_volume
else:
    def particle_centered_density(particles, density_map,
                                  x_min, x_max, dx, nx,
                                  y_min, y_max, dy, ny,
                                  z_min, zmax, dz, nz,
                                  h, inv_volume):
        """
        파이썬 루프 기반 Top-hat 커널의 입자 중심 밀도 계산.
        최적화를 위해 squared distance를 사용합니다.
        """
        box_size_x = x_max - x_min
        box_size_y = y_max - y_min
        box_size_z = zmax - z_min
        half_box_x = box_size_x / 2.0
        half_box_y = box_size_y / 2.0
        half_box_z = box_size_z / 2.0
        h2 = h * h
        n_particles = particles.shape[0]
        for i in range(n_particles):
            px = particles[i, 0]
            py = particles[i, 1]
            pz = particles[i, 2]
            fx = (px - x_min) / dx
            fy = (py - y_min) / dy
            fz = (pz - z_min) / dz
            r_idx = int(np.ceil(h / dx))
            ix_min = max(0, int(np.floor(fx)) - r_idx)
            ix_max = min(nx - 1, int(np.floor(fx)) + r_idx)
            iy_min = max(0, int(np.floor(fy)) - r_idx)
            iy_max = min(ny - 1, int(np.floor(fy)) + r_idx)
            iz_min = max(0, int(np.floor(fz)) - r_idx)
            iz_max = min(nz - 1, int(np.floor(fz)) + r_idx)
            for ix in range(ix_min, ix_max + 1):
                x_center = x_min + (ix + 0.5) * dx
                diff_x = x_center - px
                if diff_x > half_box_x:
                    diff_x -= box_size_x
                elif diff_x < -half_box_x:
                    diff_x += box_size_x
                for iy in range(iy_min, iy_max + 1):
                    y_center = y_min + (iy + 0.5) * dy
                    diff_y = y_center - py
                    if diff_y > half_box_y:
                        diff_y -= box_size_y
                    elif diff_y < -half_box_y:
                        diff_y += box_size_y
                    for iz in range(iz_min, iz_max + 1):
                        z_center = z_min + (iz + 0.5) * dz
                        diff_z = z_center - pz
                        if diff_z > half_box_z:
                            diff_z -= box_size_z
                        elif diff_z < -half_box_z:
                            diff_z += box_size_z
                        sq_dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                        if sq_dist <= h2:
                            density_map[ix, iy, iz] += inv_volume

# =============================================================================
# 삼각형 커널 전용 입자 중심 방식 밀도 계산 함수
# -----------------------------------------------------------------------------
if USE_NUMBA:
    @numba.njit(fastmath=True)
    def particle_centered_density_triangle(particles, density_map,
                                           x_min, x_max, dx, nx,
                                           y_min, y_max, dy, ny,
                                           z_min, zmax, dz, nz,
                                           h):
        """
        Numba 가속: 삼각형 커널을 적용하여 입자 중심 밀도 계산.
        squared distance로 cutoff 판정을 수행한 후 필요한 경우에만 sqrt 연산.
        """
        box_size_x = x_max - x_min
        box_size_y = y_max - y_min
        box_size_z = zmax - z_min
        half_box_x = box_size_x / 2.0
        half_box_y = box_size_y / 2.0
        half_box_z = box_size_z / 2.0
        h2 = h * h
        n_particles = particles.shape[0]
        constant = 3.0 / (np.pi * h ** 3)
        for i in range(n_particles):
            px = particles[i, 0]
            py = particles[i, 1]
            pz = particles[i, 2]
            fx = (px - x_min) / dx
            fy = (py - y_min) / dy
            fz = (pz - z_min) / dz
            r_idx = int(np.ceil(h / dx))
            ix_min = max(0, int(np.floor(fx)) - r_idx)
            ix_max = min(nx - 1, int(np.floor(fx)) + r_idx)
            iy_min = max(0, int(np.floor(fy)) - r_idx)
            iy_max = min(ny - 1, int(np.floor(fy)) + r_idx)
            iz_min = max(0, int(np.floor(fz)) - r_idx)
            iz_max = min(nz - 1, int(np.floor(fz)) + r_idx)
            for ix in range(ix_min, ix_max + 1):
                x_center = x_min + (ix + 0.5) * dx
                diff_x = x_center - px
                if diff_x > half_box_x:
                    diff_x -= box_size_x
                elif diff_x < -half_box_x:
                    diff_x += box_size_x
                for iy in range(iy_min, iy_max + 1):
                    y_center = y_min + (iy + 0.5) * dy
                    diff_y = y_center - py
                    if diff_y > half_box_y:
                        diff_y -= box_size_y
                    elif diff_y < -half_box_y:
                        diff_y += box_size_y
                    for iz in range(iz_min, iz_max + 1):
                        z_center = z_min + (iz + 0.5) * dz
                        diff_z = z_center - pz
                        if diff_z > half_box_z:
                            diff_z -= box_size_z
                        elif diff_z < -half_box_z:
                            diff_z += box_size_z
                        sq_dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                        if sq_dist <= h2:
                            dist = np.sqrt(sq_dist)  # sqrt only when necessary
                            density_map[ix, iy, iz] += constant * (1 - (dist / h))
else:
    def particle_centered_density_triangle(particles, density_map,
                                           x_min, x_max, dx, nx,
                                           y_min, y_max, dy, ny,
                                           z_min, zmax, dz, nz,
                                           h):
        """
        파이썬 루프 기반 삼각형 커널의 입자 중심 밀도 계산.
        squared distance로 cutoff 판정을 수행하고, 필요한 경우에만 sqrt 연산을 수행합니다.
        """
        box_size_x = x_max - x_min
        box_size_y = y_max - y_min
        box_size_z = zmax - z_min
        half_box_x = box_size_x / 2.0
        half_box_y = box_size_y / 2.0
        half_box_z = box_size_z / 2.0
        h2 = h * h
        n_particles = particles.shape[0]
        constant = 3.0 / (np.pi * h ** 3)
        for i in range(n_particles):
            px = particles[i, 0]
            py = particles[i, 1]
            pz = particles[i, 2]
            fx = (px - x_min) / dx
            fy = (py - y_min) / dy
            fz = (pz - z_min) / dz
            r_idx = int(np.ceil(h / dx))
            ix_min = max(0, int(np.floor(fx)) - r_idx)
            ix_max = min(nx - 1, int(np.floor(fx)) + r_idx)
            iy_min = max(0, int(np.floor(fy)) - r_idx)
            iy_max = min(ny - 1, int(np.floor(fy)) + r_idx)
            iz_min = max(0, int(np.floor(fz)) - r_idx)
            iz_max = min(nz - 1, int(np.floor(fz)) + r_idx)
            for ix in range(ix_min, ix_max + 1):
                x_center = x_min + (ix + 0.5) * dx
                diff_x = x_center - px
                if diff_x > half_box_x:
                    diff_x -= box_size_x
                elif diff_x < -half_box_x:
                    diff_x += box_size_x
                for iy in range(iy_min, iy_max + 1):
                    y_center = y_min + (iy + 0.5) * dy
                    diff_y = y_center - py
                    if diff_y > half_box_y:
                        diff_y -= box_size_y
                    elif diff_y < -half_box_y:
                        diff_y += box_size_y
                    for iz in range(iz_min, iz_max + 1):
                        z_center = z_min + (iz + 0.5) * dz
                        diff_z = z_center - pz
                        if diff_z > half_box_z:
                            diff_z -= box_size_z
                        elif diff_z < -half_box_z:
                            diff_z += box_size_z
                        sq_dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                        if sq_dist <= h2:
                            dist = np.sqrt(sq_dist)
                            density_map[ix, iy, iz] += constant * (1 - (dist / h))
                            
# =============================================================================
# DensityCalculator 클래스 (GPU와 CPU 모두 지원)
# -----------------------------------------------------------------------------
class DensityCalculator:
    """
    DensityCalculator 클래스는 3차원 입자 데이터를 받아서,
    주어진 커널 함수를 사용하여 3D 밀도 지도를 계산합니다.
    GPU 가속을 적용하며, GPU 초기화, 최적 GPU 선택, 비동기 스트림, 에러 핸들링, 메모리 로깅 등을 지원합니다.
    """
    def __init__(self, particles, grid_bounds, grid_spacing, batch_size=2000, use_gpu=True):
        """
        Parameters:
            particles (np.ndarray): (N, 3) 크기의 입자 좌표 배열.
            grid_bounds (dict): {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
            grid_spacing (tuple): (dx, dy, dz)
            batch_size (int): GPU/CPU 계산 시 배치 크기.
            use_gpu (bool): GPU 가속 사용 여부.
        """
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.batch_size = batch_size
        logger.debug("DensityCalculator 생성: grid_bounds=%s, grid_spacing=%s, use_gpu=%s, batch_size=%d",
                     grid_bounds, grid_spacing, self.use_gpu, self.batch_size)

    def compute_optimal_bandwidth(self):
        """
        Silverman's rule (다변량 버전)을 사용하여 최적의 커널 밴드위스 h 값을 계산합니다.
        """
        d = 3  # 차원
        n = self.particles.shape[0]
        sigma = np.std(self.particles, axis=0)
        sigma_avg = np.mean(sigma)
        C = (4 / (d + 2)) ** (1 / (d + 4))
        h_opt = C * (n ** (-1 / (d + 4))) * sigma_avg
        logger.info("Silverman's rule 최적 밴드위스: h_opt=%.6f (n=%d, sigma_avg=%.6f, C=%.6f)",
                    h_opt, n, sigma_avg, C)
        return h_opt

    def calculate_density_map(self, kernel_func=KernelFunctions.uniform, h=None):
        """
        3차원 격자에서 커널 밀도 추정을 수행하여 3D 밀도 맵을 생성합니다.
        h 값이 주어지지 않으면 Silverman's rule로 최적의 h 값을 자동 계산합니다.
        
        Parameters:
            kernel_func (callable): 사용할 커널 함수 (예: KernelFunctions.gaussian, uniform, triangular 등)
            h (float, optional): 커널 밴드위스. None이면 자동 계산.
        
        Returns:
            tuple: (x_centers, y_centers, z_centers, density_map)
        """
        if h is None:
            logger.info("h 값 미제공: Silverman's rule로 최적 h 계산")
            h = self.compute_optimal_bandwidth()
            logger.info("자동 계산된 h 값: %.6f", h)

        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        zmin, zmax = self.grid_bounds['z']
        dx, dy, dz = self.grid_spacing

        # 만약 uniform 및 triangular 커널 사용 시, 입자 중심 방식 적용
        if kernel_func in (KernelFunctions.uniform, KernelFunctions.triangular):
            nx = int((xmax - xmin) // dx)
            ny = int((ymax - ymin) // dy)
            nz = int((zmax - zmin) // dz)
            density_map = np.zeros((nx, ny, nz), dtype=np.float64)
            logger.debug("입자 중심 방식 밀도 계산 시작 (입자 수=%d, grid: %dx%dx%d)",
                         self.particles.shape[0], nx, ny, nz)
            if kernel_func == KernelFunctions.uniform:
                inv_volume = 1.0 / (4.0/3.0 * np.pi * h**3)
                particle_centered_density(self.particles, density_map,
                                            xmin, xmax, dx, nx,
                                            ymin, ymax, dy, ny,
                                            zmin, zmax, dz, nz,
                                            h, inv_volume)
            elif kernel_func == KernelFunctions.triangular:
                particle_centered_density_triangle(self.particles, density_map,
                                                   xmin, xmax, dx, nx,
                                                   ymin, ymax, dy, ny,
                                                   zmin, zmax, dz, nz,
                                                   h)
            x_centers = xmin + (np.arange(nx) + 0.5) * dx
            y_centers = ymin + (np.arange(ny) + 0.5) * dy
            z_centers = zmin + (np.arange(nz) + 0.5) * dz
            logger.debug("입자 중심 방식 밀도 계산 완료.")
            return x_centers, y_centers, z_centers, density_map

        # GPU 기반 grid 중심 방식 계산
        if self.use_gpu:
            logger.debug("GPU 가속 grid 중심 방식 계산 시작.")
            try:
                # CUDA 런타임 초기화 및 최적 GPU 선택
                cp.cuda.runtime.getDeviceCount()  # 초기화 체크
                best_gpu = select_best_gpu()
                if best_gpu is None:
                    raise RuntimeError("GPU 초기화 실패: 최적 GPU 선택 불가")
                with cp.cuda.Device(best_gpu):
                    log_gpu_memory_info("Before GPU computation")
                    
                    # 격자 생성 (grid-centered)
                    x_centers = np.arange(xmin + dx/2, xmax, dx)
                    y_centers = np.arange(ymin + dy/2, ymax, dy)
                    z_centers = np.arange(zmin + dz/2, zmax, dz)
                    nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
                    grid_points = np.array(np.meshgrid(x_centers, y_centers, z_centers, indexing='ij'))
                    grid_points = grid_points.reshape(3, -1).T  # shape: (n_grid, 3)
                    cutoff = 3 * h

                    # Pre-transfer the grid to GPU memory to reduce repeated host-device transfers.
                    grid_points_gpu = cp.asarray(grid_points)

                    # GPU 연산: 비동기 스트림 사용 및 batch 처리
                    n_points = grid_points.shape[0]
                    density_result = cp.zeros(n_points, dtype=cp.float64)
                    particles_gpu = cp.asarray(self.particles)
                    N_particles = particles_gpu.shape[0]
                    particle_batch_size = 10000

                    stream = cp.cuda.Stream(non_blocking=True)
                    with stream:
                        for start in tqdm(range(0, n_points, self.batch_size),
                                          desc="GPU grid batch processing", unit="batch"):
                            end_idx = min(start + self.batch_size, n_points)
                            # Use the preallocated grid_points_gpu
                            batch_points_gpu = grid_points_gpu[start:end_idx]
                            density_values_batch = cp.zeros(batch_points_gpu.shape[0], dtype=cp.float64)
                            
                            # 내부 루프: 입자 배치 처리 (vectorized over particle batches)
                            for p_start in range(0, N_particles, particle_batch_size):
                                p_end = min(p_start + particle_batch_size, N_particles)
                                particles_gpu_batch = particles_gpu[p_start:p_end]
                                # Compute differences in a vectorized fashion
                                diff = batch_points_gpu[:, cp.newaxis, :] - particles_gpu_batch[cp.newaxis, :, :]
                                # Compute squared distances to filter based on cutoff without computing sqrt unnecessarily
                                sq_dist = cp.sum(diff ** 2, axis=-1)
                                mask = sq_dist <= cutoff * cutoff
                                # Compute sqrt only for valid entries
                                dist = cp.where(mask, cp.sqrt(sq_dist), 0.)
                                kernel_vals = kernel_func(dist, h)
                                kernel_vals = cp.where(mask, kernel_vals, 0.)
                                density_values_batch += cp.sum(kernel_vals, axis=1)
                            density_result[start:end_idx] = density_values_batch
                    stream.synchronize()
                    log_gpu_memory_info("After GPU computation")
                    density_map = cp.asnumpy(density_result).reshape((nx, ny, nz))
                    logger.debug("GPU grid 중심 방식 계산 완료.")
                    return x_centers, y_centers, z_centers, density_map
            except Exception as gpu_e:
                logger.error("GPU 가속 중 오류 발생: %s", gpu_e)
                logger.info("CPU 기반 계산으로 전환합니다.")
                # 오류 발생 시 CPU 기반 계산으로 fallback
                self.use_gpu = False

        # CPU 기반 grid 중심 방식 계산 (cKDTree 사용)
        logger.debug("CPU 기반 grid 중심 방식 계산 시작.")
        x_centers = np.arange(xmin + dx/2, xmax, dx)
        y_centers = np.arange(ymin + dy/2, ymax, dy)
        z_centers = np.arange(zmin + dz/2, zmax, dz)
        nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
        grid_points = np.array(np.meshgrid(x_centers, y_centers, z_centers, indexing='ij'))
        grid_points = grid_points.reshape(3, -1).T  # shape: (n_grid, 3)
        cutoff = 3 * h
        from scipy.spatial import cKDTree
        tree = cKDTree(self.particles)
        n_points = grid_points.shape[0]
        density_result = np.empty(n_points, dtype=np.float64)
        for i, gp in enumerate(tqdm(grid_points, desc="CPU grid processing", unit="grid")):
            indices = tree.query_ball_point(gp, r=cutoff)
            if indices:
                neighbors = self.particles[indices]
                diff = gp - neighbors
                dist = np.linalg.norm(diff, axis=1)
                kernel_vals = kernel_func(dist, h)
                density_result[i] = np.sum(kernel_vals)
            else:
                density_result[i] = 0.0
        density_map = density_result.reshape((nx, ny, nz))
        logger.debug("CPU grid 중심 방식 계산 완료.")
        return x_centers, y_centers, z_centers, density_map
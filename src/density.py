#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: density.py
#
# 최적화를 적용한 3차원 밀도 계산 코드
# - 벡터화/브로드캐스팅: GPU(CuPy) 또는 CPU(Numpy) 기반으로 전체 격자에 대해 한 번에 연산합니다.
# - 병렬 처리: CPU 계산 시 격자 배치를 배치 처리로 멀티프로세싱을 이용합니다.
# - 코드 및 로깅 최적화: 내부 반복 로깅을 최소화하고, 메모리 관리를 위해 불필요한 복사를 제거합니다.
""" LAST ERROR
2025-03-07 23:20:08,854 - ERROR - Unhandled exception occurred:
Traceback (most recent call last):
  File "/caefs/user/mmingyeong/structura/Structura/src/example/density_ex.py", line 150, in <module>
    main()
  File "/caefs/user/mmingyeong/structura/Structura/src/example/density_ex.py", line 106, in main
    x_centers, y_centers, z_centers, density_map, used_h = calculator.calculate_density_map(selected_kernel_func, h, return_used_h=True)
TypeError: DensityCalculator.calculate_density_map() got an unexpected keyword argument 'return_used_h'
2025-03-07 23:20:08,854 - ERROR - Unhandled exception occurred:
Traceback (most recent call last):
  File "/caefs/user/mmingyeong/structura/Structura/src/example/density_ex.py", line 150, in <module>
    main()
  File "/caefs/user/mmingyeong/structura/Structura/src/example/density_ex.py", line 106, in main
    x_centers, y_centers, z_centers, density_map, used_h = calculator.calculate_density_map(selected_kernel_func, h, return_used_h=True)
TypeError: DensityCalculator.calculate_density_map() got an unexpected keyword argument 'return_used_h'
"""

import numpy as np
from logger import logger
from tqdm import tqdm

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from concurrent.futures import ProcessPoolExecutor

# 외부 모듈에서 커널 함수들을 import (예: kernel.py의 KernelFunctions)
from kernel import KernelFunctions

def compute_density_batch(batch_points, particles, h, kernel_func):
    """
    CPU 계산용 배치 처리 함수.
    각 배치에 대해 모든 입자와의 유클리드 거리를 계산하고,
    벡터화된 커널 함수(kernel_func)를 적용하여 각 격자 셀의 밀도를 계산합니다.
    
    Parameters:
        batch_points (np.ndarray): (M, 3) 크기의 격자 셀 중심 좌표
        particles (np.ndarray): (N, 3) 크기의 입자 좌표 배열
        h (float): 커널 밴드위스
        kernel_func (callable): 벡터화된 커널 함수 (예: KernelFunctions.gaussian)
    
    Returns:
        np.ndarray: (M,) 크기의 밀도 값 배열
    """
    # (M, N, 3): 각 배치 격자와 모든 입자 간의 차이
    diff = batch_points[:, None, :] - particles[None, :, :]
    # (M, N): 각 격자 셀에 대한 입자와의 유클리드 거리
    dist = np.linalg.norm(diff, axis=-1)
    # 커널 함수 적용 (벡터화되어 있음)
    kernel_vals = kernel_func(dist, h)
    # 각 격자 셀별로 입자 커널값을 합산
    density = np.sum(kernel_vals, axis=1)
    return density

class DensityCalculator:
    """
    DensityCalculator 클래스는 3차원 입자 데이터를 받아서,
    주어진 커널 함수를 사용하여 3D 밀도 지도를 계산합니다.
    """

    def __init__(self, particles, grid_bounds, grid_spacing, use_gpu=True):
        """
        Parameters:
            particles (np.ndarray): (N, 3) 크기의 입자 좌표 배열.
            grid_bounds (dict): {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
            grid_spacing (tuple): (dx, dy, dz)
            use_gpu (bool): GPU 가용 시 GPU 연산 사용 여부
        """
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.use_gpu = use_gpu and GPU_AVAILABLE
        logger.info("DensityCalculator 객체 생성: grid_bounds=%s, grid_spacing=%s, use_gpu=%s",
                    grid_bounds, grid_spacing, self.use_gpu)

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
        logger.info("Silverman's rule에 따른 최적 밴드위스: h_opt=%.6f (n=%d, sigma_avg=%.6f, C=%.6f)",
                    h_opt, n, sigma_avg, C)
        return h_opt

    def calculate_density_map(self, kernel_func=KernelFunctions.gaussian, h=None):
        """
        3차원 격자에서 커널 밀도 추정을 수행하여 3D 밀도 맵을 생성합니다.
        만약 h 값이 주어지지 않으면 Silverman's rule을 이용하여 최적의 h 값을 자동으로 계산합니다.
        
        Parameters:
            kernel_func (callable): 사용할 벡터화된 커널 함수 (예: KernelFunctions.gaussian)
            h (float, optional): 커널 밴드위스. None이면 자동 계산.
        
        Returns:
            tuple: (x_centers, y_centers, z_centers, density_map)
                - x_centers, y_centers, z_centers: 각 축의 격자 셀 중심 좌표 (1D 배열)
                - density_map (np.ndarray): 3D 밀도 배열, shape = [n_x, n_y, n_z]
        """
        if h is None:
            logger.info("h 값이 제공되지 않아 Silverman's rule로 최적 h 계산을 시작합니다.")
            h = self.compute_optimal_bandwidth()
            logger.info("자동 계산된 h 값: %.6f", h)

        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        zmin, zmax = self.grid_bounds['z']
        dx, dy, dz = self.grid_spacing

        # 격자 셀 중심 좌표 생성
        x_centers = np.arange(xmin + dx / 2, xmax, dx)
        y_centers = np.arange(ymin + dy / 2, ymax, dy)
        z_centers = np.arange(zmin + dz / 2, zmax, dz)
        nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)
        logger.info("격자 크기: nx=%d, ny=%d, nz=%d", nx, ny, nz)

        # 전체 격자 셀 중심 좌표를 2차원 배열 (N, 3)로 생성
        grid_points = np.array(np.meshgrid(x_centers, y_centers, z_centers, indexing='ij'))
        grid_points = grid_points.reshape(3, -1).T  # shape: (n_grid, 3)

        logger.info("커널 밀도 지도 계산 시작.")

        # GPU 가용 및 사용 시: CuPy를 활용한 벡터화 연산
        if self.use_gpu:
            logger.info("GPU를 사용하여 계산을 진행합니다.")
            particles_gpu = cp.asarray(self.particles)
            grid_points_gpu = cp.asarray(grid_points)
            # (n_grid, n_particles, 3): 각 격자와 모든 입자 간 차이
            diff = grid_points_gpu[:, cp.newaxis, :] - particles_gpu[cp.newaxis, :, :]
            # (n_grid, n_particles): 유클리드 거리 계산
            dist = cp.linalg.norm(diff, axis=-1)
            # 커널 함수 적용: 전달받은 kernel_func가 벡터화되어 있어야 함
            kernel_vals = kernel_func(dist, h)
            density_values = cp.sum(kernel_vals, axis=1)
            density_map = density_values.get().reshape((nx, ny, nz))
        else:
            logger.info("CPU를 사용하여 계산을 진행합니다. (벡터화 및 병렬 처리 적용)")
            batch_size = 1000
            n_points = grid_points.shape[0]
            density_result = np.empty(n_points, dtype=np.float64)

            def process_batch(start_idx):
                end_idx = min(start_idx + batch_size, n_points)
                batch_points = grid_points[start_idx:end_idx]
                return compute_density_batch(batch_points, self.particles, h, kernel_func)

            futures = []
            with ProcessPoolExecutor() as executor:
                for start in range(0, n_points, batch_size):
                    futures.append(executor.submit(process_batch, start))
                idx = 0
                for future in tqdm(futures, desc="CPU 배치 처리", unit="batch"):
                    batch_density = future.result()
                    batch_len = batch_density.shape[0]
                    density_result[idx:idx+batch_len] = batch_density
                    idx += batch_len
            density_map = density_result.reshape((nx, ny, nz))

        logger.info("커널 밀도 지도 계산 완료.")
        return x_centers, y_centers, z_centers, density_map


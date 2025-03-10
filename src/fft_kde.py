#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-07
# @Filename: fft_kde.py
# structura/fft_kde.py

import numpy as np
from logger import logger
from kernel import KernelFunctions

# 플래그 확인: 모듈 최상위에서 한 번만 초기화하도록 함.
if not globals().get("_LOGGING_INITIALIZED", False):
    globals()["_LOGGING_INITIALIZED"] = True

# GPU 사용 가능 여부 확인 (CuPy)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    if not globals().get("_GPU_LOGGED", False):
        logger.info("GPU is available. CuPy will be used for FFT.")
        globals()["_GPU_LOGGED"] = True
except ImportError:
    GPU_AVAILABLE = False
    if not globals().get("_GPU_LOGGED", False):
        logger.info("GPU is not available. Falling back to NumPy FFT.")
        globals()["_GPU_LOGGED"] = True

# Dask 및 Numba 임포트 (대규모 데이터 처리를 위해)
import multiprocessing

def compute_optimal_chunksize(total_particles):
    """
    전체 입자 수(total_particles)와 시스템의 CPU 코어 수를 기반으로
    적절한 청크 크기를 계산합니다.
    
    예: 각 코어당 최소 4개의 청크가 있도록 설정하며, 최소값은 100,000으로 제한.
    """
    num_cores = multiprocessing.cpu_count()
    optimal = max(100000, total_particles // (num_cores * 4))
    logger.info("계산된 최적의 chunksize: %d (총 입자 수: %d, CPU 코어 수: %d)", optimal, total_particles, num_cores)
    return int(optimal)

class FFTKDE:
    """
    FFT 기반 합성곱 방식을 이용하여 3차원 커널 밀도 추정을 수행하는 클래스입니다.
    
    이 클래스는 대용량 입자 데이터를 numpy.histogramdd로 격자에 바인딩한 후,
    FFT를 이용하여 합성곱으로 3D 밀도 지도를 계산합니다.
    
    Parameters
    ----------
    particles : np.ndarray
        (N, 3) 크기의 입자 좌표 배열.
    grid_bounds : dict
        {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)} 형식의 그리드 범위.
    grid_spacing : tuple
        (dx, dy, dz) 각 축의 그리드 셀 크기.
    kernel_func : callable, optional
        사용할 커널 함수. 기본값은 KernelFunctions.gaussian.
    h : float, optional
        커널 밴드위스. 기본값은 1.0.
    """
    def __init__(self, particles, grid_bounds, grid_spacing, kernel_func=KernelFunctions.gaussian, h=1.0):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.kernel_func = kernel_func
        self.h = h
        logger.info("FFTKDE 객체 생성: grid_bounds=%s, grid_spacing=%s, h=%.3f",
                    grid_bounds, grid_spacing, h)

    def _compute_histogram(self):
        """
        numpy.histogramdd를 사용하여 3차원 입자 밀도 히스토그램을 계산합니다.
        
        Returns
        -------
        H : np.ndarray
            3차원 입자 밀도 히스토그램 배열.
        edges : list of np.ndarray
            각 축의 bin edge 배열.
        """
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        zmin, zmax = self.grid_bounds['z']
        dx, dy, dz = self.grid_spacing

        # 각 축의 bin edge 생성 (마지막 경계까지 포함)
        x_edges = np.arange(xmin, xmax + dx, dx)
        y_edges = np.arange(ymin, ymax + dy, dy)
        z_edges = np.arange(zmin, zmax + dz, dz)
        bins = [x_edges, y_edges, z_edges]

        H, edges = np.histogramdd(self.particles, bins=bins)
        return H, edges

    def compute_density(self):
        """
        FFT 기반 합성곱 방식을 이용하여 커널 밀도 추정을 수행합니다.
        
        Returns
        -------
        x_centers, y_centers, z_centers : np.ndarray
            각 축의 그리드 셀 중심 좌표.
        density_conv : np.ndarray
            FFT 기반 합성곱 결과로 얻은 3D 밀도 지도.
        """
        H, edges = self._compute_histogram()
        x_edges, y_edges, z_edges = edges
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        nx, ny, nz = H.shape

        dx, dy, dz = self.grid_spacing
        x_kernel = np.fft.fftfreq(nx, d=dx) * nx * dx
        y_kernel = np.fft.fftfreq(ny, d=dy) * ny * dy
        z_kernel = np.fft.fftfreq(nz, d=dz) * nz * dz
        X, Y, Z = np.meshgrid(x_kernel, y_kernel, z_kernel, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        kernel_grid = self.kernel_func(R, self.h)

        H_fft = np.fft.fftn(H)
        kernel_fft = np.fft.fftn(kernel_grid)
        density_fft = H_fft * kernel_fft
        density_conv = np.fft.ifftn(density_fft).real

        logger.info("FFT 기반 커널 밀도 추정 완료. (히스토그램 shape: %s)", H.shape)
        return x_centers, y_centers, z_centers, density_conv

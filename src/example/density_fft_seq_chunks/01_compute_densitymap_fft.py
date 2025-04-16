#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_densitymap_fft.py

This script reads .npy files from a configuration-specified directory,
performs FFT-based kernel density estimation using the CPU-only version 
of the FFTKDE class, and saves the resulting density maps to an output directory.
Parallel processing is implemented via ProcessPoolExecutor using CPU cores only,
and Dask is used for out-of-core processing in FFTKDE.
A progress bar is displayed to show the processing status.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Date: 2025-03-07 (revised with Dask distributed client, chunk processing, and memory mapping)
"""

import os
import sys
import time
import logging
import numpy as np
import psutil
import multiprocessing
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from dask.distributed import Client

# -------------------------------
# 경로 설정 및 모듈 임포트
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import OUTPUT_DATA_PATHS
from fft_kde import FFTKDE
from kernel import KernelFunctions

npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]

# -------------------------------
# 로깅 설정
# -------------------------------
logger = logging.getLogger("FFTKDELogger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler("fft_kde.log", mode="a")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# -------------------------------
# 메모리 사용량 로그 함수
# -------------------------------
def log_memory_usage(label=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    logger.info(f"💾 Memory usage {label}: {mem_mb:.2f} MB")

# -------------------------------
# 파일별 밀도 계산 함수 (FFTKDE.compute_density() 호출)
# -------------------------------
def compute_density_for_file(npy_path: str, output_dir: str,
                              cube_size: float = 205.0,
                              bandwidth: float = 1.0,
                              grid_resolution: float = 0.82,
                              kernel=KernelFunctions.uniform):
    try:
        # 메모리 매핑 방식으로 입력 파일 로드
        particles = np.load(npy_path, mmap_mode='r')
    except Exception as e:
        logger.error("❌ Failed to load file %s: %s", npy_path, str(e))
        return

    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    output_file = os.path.join(output_dir, f"density_fft_{base_name}_{kernel.__name__}.npy")
    if os.path.exists(output_file):
        logger.warning("⚠️ Output already exists: %s. Skipping.", output_file)
        return

    grid_bounds = {'x': (0.0, cube_size),
                   'y': (0.0, cube_size),
                   'z': (0.0, cube_size)}
    grid_spacing = (grid_resolution, grid_resolution, grid_resolution)

    logger.info("📊 Starting FFTKDE for [%s] (N=%d)", base_name, particles.shape[0])
    log_memory_usage("before FFTKDE compute")

    try:
        fft_kde = FFTKDE(particles, grid_bounds, grid_spacing,
                         kernel_func=kernel, h=bandwidth)
        _, _, _, density_map = fft_kde.compute_density()
    except Exception as e:
        logger.error("❌ Error during density computation [%s]: %s", base_name, str(e))
        return

    log_memory_usage("after FFTKDE compute")

    try:
        # 메모리 매핑을 활용해 최종 결과 저장
        from numpy.lib.format import open_memmap
        fp = open_memmap(output_file, mode='w+', dtype=density_map.dtype, shape=density_map.shape)
        fp[:] = density_map[:]
    except Exception as e:
        logger.error("❌ Error saving density map for file %s: %s", base_name, str(e))
        return

    logger.info("✅ Density map saved: %s", output_file)
    print(f"[{base_name}] Density map saved to {output_file}")

# -------------------------------
# 파일 단위 처리 함수 (각 파일에 대해 compute_density_for_file() 호출)
# -------------------------------
def process_file(fname, output_dir, cube_size, bandwidth, grid_resolution, kernel):
    full_path = os.path.join(npy_folder, fname)
    logger.info("🔹 Processing file: %s", fname)
    compute_density_for_file(full_path, output_dir,
                             cube_size=cube_size,
                             bandwidth=bandwidth,
                             grid_resolution=grid_resolution,
                             kernel=kernel)

# -------------------------------
# 메모리 기반 max_workers 계산 함수
# -------------------------------
def estimate_memory_usage(resolution: float, cube_size: float = 205.0) -> float:
    grid_points = int(cube_size / resolution)
    total_elements = grid_points ** 3
    bytes_needed = total_elements * np.dtype(np.complex128).itemsize
    return bytes_needed / 1024**3  # GiB

def determine_max_workers(resolution: float, buffer_gib: float = 30.0) -> int:
    available_ram_gib = psutil.virtual_memory().available / 1024**3 - buffer_gib
    mem_per_task = estimate_memory_usage(resolution)
    max_tasks = int(available_ram_gib // mem_per_task)
    return max(1, min(max_tasks, multiprocessing.cpu_count()))

# -------------------------------
# 메인 실행 함수
# -------------------------------
def main():
    # Dask Distributed Client 초기화 (각 프로세스는 개별적으로 Dask 작업을 수행)
    try:
        from dask.distributed import Client
        client = Client()  # 기본 클러스터를 시작합니다.
        logger.info("Dask distributed client started. Dashboard: %s", client.dashboard_link)
    except Exception as e:
        logger.error("Failed to start Dask distributed client: %s", str(e))

    # 사용자 설정값
    resolution = 0.41
    kernel = KernelFunctions.uniform
    cube_size = 205.0
    bandwidth = 1.0

    output_dir = f"fft_densitymap_{kernel.__name__}_dx{resolution}"
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(npy_folder) if f.endswith(".npy")])
    logger.info("📁 Found %d input files in: %s", len(all_files), npy_folder)

    max_workers = determine_max_workers(resolution)
    logger.info("🧠 Auto-selected max_workers: %d (resolution=%.2f)", max_workers, resolution)

    proc_file = partial(process_file,
                        output_dir=output_dir,
                        cube_size=cube_size,
                        bandwidth=bandwidth,
                        grid_resolution=resolution,
                        kernel=kernel)

    start_time = time.time()
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fname in all_files:
            futures.append(executor.submit(proc_file, fname))
        for f in tqdm(futures, desc="Processing files", total=len(futures)):
            try:
                f.result()
            except Exception as e:
                logger.error("❌ Exception occurred: %s", str(e))

    elapsed = time.time() - start_time
    logger.info("🏁 All files processed. Elapsed time: %.2f seconds", elapsed)
    print(f"✅ Total execution time: {elapsed:.2f} seconds")

# -------------------------------
# 실행 시작
# -------------------------------
if __name__ == "__main__":
    main()

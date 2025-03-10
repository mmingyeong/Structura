#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-10
# @Filename: data_load_ex.py
#
# This example tests only the data loading part using the DataLoader class
# and outputs basic statistics for the loaded data.

import sys
import os
import time
import cProfile
import pstats
import io
import numpy as np

# Append the parent directory (src) to the module search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import DataLoader
from config import OUTPUT_DATA_PATHS
from logger import logger


def print_statistics(data, label="Data"):
    """
    Compute and log basic statistics (mean, median, min, max) for the given data.
    
    Parameters
    ----------
    data : np.ndarray or cupy.ndarray
        Loaded data array.
    label : str, optional
        Label to prefix the statistical information.
    """
    # If using CuPy, convert to NumPy for statistics.
    if hasattr(data, "get"):
        data_np = data.get()
    else:
        data_np = data

    data_mean = np.mean(data_np, axis=0)
    data_median = np.median(data_np, axis=0)
    data_min = np.min(data_np, axis=0)
    data_max = np.max(data_np, axis=0)
    logger.info(f"{label} statistics:")
    logger.info(f"  Mean:   {data_mean}")
    logger.info(f"  Median: {data_median}")
    logger.info(f"  Min:    {data_min}")
    logger.info(f"  Max:    {data_max}")


def main():
    start_time = time.time()
    
    # ------------------------------------------------------------------
    # Configuration for testing the data loading process.
    # ------------------------------------------------------------------
    npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    logger.info("Input directory (npy): %s", npy_folder)

    # Verify the existence of .npy files.
    if os.path.exists(npy_folder) and os.path.isdir(npy_folder):
        npy_files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
        if not npy_files:
            logger.error("No .npy files found in %s.", npy_folder)
            return
        logger.debug("Found %d .npy file(s). Example file: %s", len(npy_files), npy_files[0])
    else:
        logger.error("Input directory does not exist: %s", npy_folder)
        return

    # Initialize the DataLoader.
    loader = DataLoader(npy_folder, use_gpu=False)  # 테스트 시 GPU 사용 여부는 필요에 따라 설정

    # ------------------------------------------------------------------
    # Test 1: Load slice data
    # ------------------------------------------------------------------
    """
    x_min = 100    # 예: 100 cMpc/h
    x_max = 120    # 예: 20 cMpc/h 두께 (필요에 따라 변경)
    sampling_rate = 1.0  # 전체 데이터 사용
    projection_axis = 0  # x축 기준 필터링 (예시)
    
    logger.info("Loading slice data (x-range: %s to %s)...", x_min, x_max)
    positions = loader.load_all_chunks(
        x_min=x_min, 
        x_max=x_max, 
        sampling_rate=sampling_rate, 
        projection_axis=projection_axis, 
        workers=4  # 테스트용 worker 수, 환경에 맞게 조정
    )
    logger.info("Loaded slice data shape: %s", positions.shape)
    print_statistics(positions, label="Slice data")
    """
    
    # ------------------------------------------------------------------
    # Test 2: Load cube data using periodic boundary conditions.
    # ------------------------------------------------------------------
    cube_origin = (0, 0, 0)  # cube의 하단 모서리 좌표 (예시)
    cube_size = 50                 # cube의 한 변의 길이 (예: 50 cMpc/h)
    sampling_rate = 1
    full_length = 205.0            # 전체 도메인 크기 (예: 205 Mpc/h)
    
    logger.info("Loading cube data (origin: %s, size: %s, full_length: %.1f)...", cube_origin, cube_size, full_length)
    cube_positions = loader.load_cube_data(
        cube_origin=cube_origin,
        cube_size=cube_size,
        full_length=full_length,
        sampling_rate=sampling_rate,
        workers=48  # 테스트용 worker 수, 환경에 맞게 조정
    )
    logger.info("Loaded cube data shape: %s", cube_positions.shape)
    print_statistics(cube_positions, label="Cube data")
    
    elapsed_time = time.time() - start_time
    logger.info("Data loading test completed in %.2f seconds.", elapsed_time)


if __name__ == '__main__':
    # Run profiling and save results to profile_results.txt
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    with open("data_load_profile_results.txt", "w") as f:
        f.write(s.getvalue())
    print("Profiling results have been saved to profile_results.txt.")

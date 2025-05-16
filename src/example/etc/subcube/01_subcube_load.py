#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-10
# @Filename: 01_subcube_load.py
#
# This example demonstrates loading subcubes from the full domain (0~205 Mpc/h)
# with overlap included, extracting the central region (excluding the overlap)
# from each subcube sequentially in batches, and saving the result files to the "subcube_npy" folder.
#
# Optimizations applied:
#  - Minimized redundant conversion of cube_origin to list.
#  - Combined conditional checks to reduce overhead.
#  - Streamlined batch processing loop.
#  - Added resume functionality: if a file already exists, that subcube is skipped.
#  - After saving each subcube file, computes its MD5 hash for integrity verification.
#  - Logs are output to both terminal and a log file ("subcube_load.log").
#  - cProfile integration: if "profile" is passed as a command-line argument,
#    the script profiles the execution and prints the top 20 functions sorted by cumulative time.

import os
import sys
import time
from itertools import product
from typing import List
import hashlib
import numpy as np
import logging
import dask
dask.config.set(scheduler='single-threaded')

# Configure logging to both terminal and a file.
logger = logging.getLogger("SubcubeLogger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler: 로그 파일 저장
fh = logging.FileHandler("subcube_load.log", mode="a")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Add parent directory (src) to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data_loader import DataLoader
from config import OUTPUT_DATA_PATHS

def compute_file_md5(filename: str) -> str:
    """Compute MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception as e:
        logger.error(f"Error computing MD5 for {filename}: {e}")
        return ""
    return hash_md5.hexdigest()

def main() -> None:
    start_time = time.time()

    # ------------------------------------------------------------------
    # Basic configuration: full domain, central subcube size, and overlap
    # ------------------------------------------------------------------
    full_length: float = 205.0   # Full domain in Mpc/h
    central_size: float = 50.0   # Nominal central subcube size in Mpc/h
    overlap: float = 3.0         # Overlap length in Mpc/h
    load_size: float = central_size + 2 * overlap  # Actual load size for each subcube

    logger.info(f"Full domain: {full_length:.2f} Mpc/h, Nominal subcube size: {central_size:.2f} Mpc/h, Overlap: {overlap:.2f} Mpc/h")

    # Compute subcube parameters using DataLoader's method.
    centers, load_windows = DataLoader.compute_subcube_parameters(full_length, central_size, overlap)
    # Use the start of each load window as cube_origin along one axis.
    cube_origins_axis: List[float] = [lb[0] for lb in load_windows]
    # Generate all 3D subcube origin combinations.
    subcube_origins = list(product(cube_origins_axis, repeat=3))
    total_subcubes = len(subcube_origins)
    logger.info(f"Total number of subcubes: {total_subcubes}")

    # Initialize DataLoader (using GPU enabled/disabled as needed)
    npy_folder: str = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    loader = DataLoader(npy_folder, use_gpu=True)

    # Create output folder "subcube_npy" if it does not exist.
    output_folder = "subcube_npy"
    os.makedirs(output_folder, exist_ok=True)

    # Process subcubes in batches sequentially (batch size = 10)
    batch_size = 10  
    for batch_start in range(0, total_subcubes, batch_size):
        batch = subcube_origins[batch_start: batch_start + batch_size]
        for idx, origin in enumerate(batch, start=batch_start):
            output_filename = os.path.join(output_folder, f"subcube_{idx:04d}.npy")
            # Resume functionality: skip subcube if file already exists.
            if os.path.exists(output_filename):
                logger.info(f"Subcube {idx} already exists at {output_filename}. Skipping.")
                continue

            origin_list = list(origin)  # Convert once and reuse
            logger.info(f"Subcube {idx}: Loading cube_origin = {origin_list}, load size = {load_size:.2f} Mpc/h")
            try:
                # Enforce sequential processing by setting workers=1.
                subcube_data = loader.load_cube_data(
                    cube_origin=origin_list,
                    cube_size=load_size,
                    full_length=full_length,
                    sampling_rate=1.0,
                    workers=1
                )
            except Exception as e:
                logger.error(f"Subcube {idx} data load failed with error: {e}")
                continue

            if subcube_data is None or subcube_data.shape[0] == 0:
                logger.error(f"Subcube {idx} data load failed or returned empty data.")
                continue

            logger.info(f"Subcube {idx} loaded, data shape: {subcube_data.shape}")

            # Extract the central region (excluding the overlap)
            central_data = loader.extract_central_region(subcube_data, origin_list, central_size, overlap, full_length)
            logger.info(f"Subcube {idx} central region data shape: {central_data.shape}")

            try:
                np.save(output_filename, central_data)
                logger.info(f"Subcube {idx} saved to {output_filename}")
                # Compute file MD5 checksum and log it.
                md5_hash = compute_file_md5(output_filename)
                if md5_hash:
                    logger.info(f"Subcube {idx} file MD5 checksum: {md5_hash}")
            except Exception as e:
                logger.error(f"Error saving subcube {idx}: {e}")
                continue

            # Optionally, log basic statistics.
            if central_data.size > 0:
                data_mean = np.mean(central_data, axis=0)
                data_min = np.min(central_data, axis=0)
                data_max = np.max(central_data, axis=0)
                logger.info(f"Subcube {idx} central region statistics: Mean: {data_mean}, Min: {data_min}, Max: {data_max}")
            else:
                logger.warning(f"Subcube {idx} central region data is empty.")
        logger.info(f"Completed batch from subcube {batch_start} to {batch_start + len(batch) - 1}.")

    elapsed_time = time.time() - start_time
    logger.info(f"Completed sequential subcube data loading, processing, and saving in {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    # If "profile" is passed as an argument, run with cProfile.
    if "profile" in sys.argv:
        import cProfile, pstats
        profile_filename = "subcube_profile.prof"
        cProfile.run("main()", profile_filename)
        p = pstats.Stats(profile_filename)
        p.sort_stats("cumtime").print_stats(20)
    else:
        main()

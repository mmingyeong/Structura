#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_compute_final_density_map_array.py

각 PBS array task가 하나의 폴더를 처리하도록 수정된 버전.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Date: 2025-04-25
"""

import os
import sys
import time
import logging
import h5py
import numpy as np
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
from numba import njit, prange

def setup_logging():
    logger = logging.getLogger("HDF5DensitySum")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/sum_density_hdf5_array.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logging()

@njit(parallel=False)
def numba_add_arrays(a, b):
    for i in prange(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                a[i, j, k] += b[i, j, k]
    return a

def load_hdf5_data(fp):
    with h5py.File(fp, 'r') as f:
        return f["density_map"][:]

def compute_blockwise_sum(file_list, sample_shape, block_idx, block_size, out_dir):
    block_files = file_list[block_idx:block_idx+block_size]
    arrays = [
        da.from_delayed(delayed(load_hdf5_data)(fp), shape=sample_shape, dtype=np.float64)
        for fp in block_files
    ]
    stacked = da.stack(arrays, axis=0)
    total = stacked.sum(axis=0).rechunk((128, 128, 128))
    try:
        with ProgressBar():
            result = total.compute(scheduler='threads')
        out_path = os.path.join(out_dir, f"block_sum_{block_idx:04d}.npy")
        np.save(out_path, result)
        logger.info("중간 결과 저장 완료: %s", out_path)
        return out_path
    except Exception as e:
        logger.error("Dask 블록 합산 중 예외 발생: %s", str(e))
        return None

def final_merge_and_save(block_paths, output_path):
    total = None
    for path in block_paths:
        try:
            data = np.load(path)
            if total is None:
                total = data.copy()
            else:
                total = numba_add_arrays(total, data)
        except Exception as e:
            logger.error("중간 결과 로드 실패: %s", str(e))
    if total is not None:
        with h5py.File(output_path, 'w') as h5f:
            h5f.create_dataset("density_map", data=total, compression=None)
        logger.info("최종 저장 완료: %s", output_path)

def main():
    print("[DEBUG] sys.argv:", sys.argv)   # 👈 이거 추가

    if len(sys.argv) < 2:
        print("Usage: python 02_compute_final_density_map_array.py <folder_index>")
        sys.exit(1)

    folder_index = int(sys.argv[1])
    base_dir = "/caefs/data/IllustrisTNG/densitymap-ics-hdf5/res_0.16/"

    subfolders = sorted([
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ])

    if folder_index >= len(subfolders):
        logger.error("Invalid folder index: %d", folder_index)
        return

    subfolder = subfolders[folder_index]
    folder_path = os.path.join(base_dir, subfolder)
    logger.info("📂 처리 시작: %s", subfolder)

    start = time.time()

    file_list = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if (f.endswith(".hdf5") or f.endswith(".h5"))
        and os.path.isfile(os.path.join(folder_path, f))
    ])

    if not file_list:
        logger.warning("파일 없음: %s", folder_path)
        return
    
    print("[DEBUG] First file:", file_list[0])
    print("[DEBUG] Exists?", os.path.exists(file_list[0]))


    with h5py.File(file_list[0], 'r') as f:
        sample_shape = f["density_map"].shape

    block_size = 10
    out_dir = os.path.join(folder_path, "_blocks")
    os.makedirs(out_dir, exist_ok=True)

    block_paths = []
    for i in range(0, len(file_list), block_size):
        block_out = compute_blockwise_sum(file_list, sample_shape, i, block_size, out_dir)
        if block_out:
            block_paths.append(block_out)

    output_path = os.path.join(base_dir, f"final_{subfolder}.hdf5")
    final_merge_and_save(block_paths, output_path)

    logger.info("✅ 완료: %s (%.2f초)", subfolder, time.time() - start)

if __name__ == "__main__":
    main()

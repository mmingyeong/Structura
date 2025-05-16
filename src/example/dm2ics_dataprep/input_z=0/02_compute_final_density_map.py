#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sum_density_maps_by_folder.py

HDF5 형식의 density map들을 폴더별로 Dask를 활용해 메모리 효율적으로 블록 단위 병렬 처리하여
중간 저장 및 최종 병합까지 수행하는 최적화된 스크립트입니다. Numba 가속도 일부 연산에 적용됩니다.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Date: 2025-05-09
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
from tqdm import tqdm
from numba import njit, prange
import shutil

# ----------------------------------------------------------------------
# 로깅 설정

def setup_logging():
    logger = logging.getLogger("HDF5DensitySum")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/sum_density_hdf5_blockwise.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logging()

# ----------------------------------------------------------------------
# Numba 병렬 합산 함수

@njit(parallel=True)
def numba_add_arrays(a, b):
    for i in prange(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                a[i, j, k] += b[i, j, k]  # in-place 연산
    return a

# ----------------------------------------------------------------------
# Dask 기반 병렬 합산 함수

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

def final_merge_and_save(block_paths, output_path, out_dir):
    total = None
    for path in block_paths:
        try:
            data = np.load(path)
            if total is None:
                total = data.copy()  # in-place 합산을 위한 복사
            else:
                total = numba_add_arrays(total, data)
        except Exception as e:
            logger.error("중간 결과 로드 실패: %s", str(e))
    if total is not None:
        with h5py.File(output_path, 'w') as h5f:
            h5f.create_dataset("density_map", data=total, compression=None)
        logger.info("최종 저장 완료: %s", output_path)

    # _blocks 폴더 삭제
    try:
        shutil.rmtree(out_dir)
        logger.info("중간 결과 폴더 삭제 완료: %s", out_dir)
    except Exception as e:
        logger.warning("중간 폴더 삭제 실패: %s", str(e))

# ----------------------------------------------------------------------
# 메인 처리 루틴

def main():
    base_dir = "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5"
    if len(sys.argv) != 2:
        logger.error("하위 폴더명을 인자로 제공하세요.")
        return

    subfolder = sys.argv[1]
    if subfolder == "res_0.16":
        logger.warning("res_0.16은 처리에서 제외됨.")
        return

    folder_path = os.path.join(base_dir, subfolder)
    logger.info("\n📂 처리 시작: %s", subfolder)
    start = time.time()

    file_list = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".hdf5") or f.endswith(".h5")
    ])
    if not file_list:
        logger.warning("파일 없음: %s", folder_path)
        return

    # sample_shape 추출 (유효 파일 기반)
    sample_shape = None
    for fp in file_list:
        try:
            with h5py.File(fp, 'r') as f:
                sample_shape = f["density_map"].shape
            break
        except Exception as e:
            logger.warning("유효하지 않은 파일 건너뜀: %s", fp)
    if sample_shape is None:
        logger.error("모든 파일이 손상됨: %s", folder_path)
        return

    block_size = 10
    out_dir = os.path.join(folder_path, "_blocks")
    os.makedirs(out_dir, exist_ok=True)

    block_paths = []
    for i in range(0, len(file_list), block_size):
        block_out = compute_blockwise_sum(file_list, sample_shape, i, block_size, out_dir)
        if block_out:
            block_paths.append(block_out)

    output_path = os.path.join(base_dir, f"final_{subfolder}.hdf5")
    final_merge_and_save(block_paths, output_path, out_dir)

    logger.info("완료: %s (%.2f초 소요)\n", subfolder, time.time() - start)

if __name__ == "__main__":
    main()

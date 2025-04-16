#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_final_density_sum.py

설정된 디렉토리 내의 모든 npy 형식의 FFT 또는 기타 방식으로 계산된 density map들을  
메모리 매핑과 Dask Array를 이용해 병렬로 불러와 합산한 후, 최종 density map을  
HDF5 파일로 저장합니다. 전체 실행 시간도 로그와 콘솔에 출력합니다.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Date: 2025-03-XX
"""

import os
import sys
import time
import logging
import numpy as np
import dask.array as da
import h5py
import multiprocessing
from tqdm import tqdm
from dask import delayed

# config 파일의 경로를 모듈 검색 경로에 추가 (프로젝트 구조에 맞게 수정)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from kernel import KernelFunctions  # 만약 kernel import가 필요한 경우

# ----------------------------------------------------------------------
# 로깅 설정 함수
def setup_logging(logger_name="DensitySumLogger"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # 파일 핸들러
    fh = logging.FileHandler("sum_density.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logging()

# ----------------------------------------------------------------------
# npy 파일 리스트 가져오기
def get_npy_file_paths(directory: str):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npy")])
    logger.info("디렉토리 [%s] 에서 %d개의 npy 파일을 찾았습니다.", directory, len(files))
    return files

# ----------------------------------------------------------------------
# 지연 평가를 이용해 npy 파일 로드 함수
@delayed
def load_npy(fp):
    # mmap_mode='r'를 사용해 메모리 매핑으로 로드
    return np.load(fp, mmap_mode='r')

# ----------------------------------------------------------------------
# Dask를 이용한 npy 파일들의 density map 합산 함수 (최적화 버전)
def dask_sum_density_maps(density_dir: str, output_filepath: str):
    file_paths = get_npy_file_paths(density_dir)
    if not file_paths:
        logger.error("디렉토리 %s 에 npy 파일이 없습니다.", density_dir)
        sys.exit(1)

    # density map의 shape을 미리 알고 있다고 가정 (예: (500, 500, 500))
    sample_shape = (500, 500, 500)
    # 적절한 청크 크기를 지정 (메모리 부담을 줄이기 위해 필요 시 더 작은 값으로 조정)
    chunks = (100, 100, 100)

    logger.info("각 npy 파일을 지연 평가(delayed)로 불러와 Dask Array로 변환합니다.")
    dask_arrays = []
    for fp in tqdm(file_paths, desc="Loading Files", unit="file"):
        # delayed 객체로 로드한 후, dask array로 변환 (메모리 매핑 방식 유지)
        d_arr = da.from_delayed(load_npy(fp), shape=sample_shape, dtype=np.float64)
        d_arr = d_arr.rechunk(chunks)
        dask_arrays.append(d_arr)

    logger.info("총 %d개의 density map을 합산합니다.", len(dask_arrays))
    # Dask의 stack()를 이용해 새로운 축을 추가한 후, 축 0 방향으로 합산
    # (모든 density map이 동일한 shape임을 전제로 함)
    stacked = da.stack(dask_arrays, axis=0)
    final_density = stacked.sum(axis=0)

    logger.info("Dask 계산 실행 중...")
    # out-of-core 계산 수행 (scheduler='threads'를 사용)
    result = final_density.compute(scheduler='threads')
    logger.info("합산 완료, 최종 density map의 shape: %s", result.shape)

    try:
        with h5py.File(output_filepath, 'w') as h5f:
            h5f.create_dataset("density_map", data=result, compression="gzip")
        logger.info("최종 density map을 HDF5 파일로 저장 완료: %s", output_filepath)
    except Exception as e:
        logger.error("HDF5 파일 저장 실패: %s", str(e))
        sys.exit(1)

# ----------------------------------------------------------------------
def main():
    res = 0.41
    kernel = KernelFunctions.uniform  # 필요한 경우 사용
    start_time = time.time()
    
    # 입력 및 출력 경로 설정 (이전 KDE 기반 계산 결과가 저장된 디렉토리)
    density_dir = f"densitymap_{kernel.__name__}_dx{res}"
    output_filepath = f"final_snapshot-99_kde_density_map_{kernel.__name__}_dx{res}.hdf5"
    
    logger.info("Density map 합산 처리 시작: 입력 디렉토리: %s", density_dir)
    dask_sum_density_maps(density_dir, output_filepath)
    
    elapsed = time.time() - start_time
    logger.info("전체 실행 시간: %.2f초", elapsed)
    print(f"전체 실행 시간: {elapsed:.2f}초")

if __name__ == "__main__":
    main()

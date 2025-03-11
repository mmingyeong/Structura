# 01_subcube_load.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-10
# @Filename: 01_subcube_load.py
#
# 이 예시는 전체 도메인(0~205 Mpc/h)을 subcube 단위로 (overlap 포함) 불러오고,
# 각 subcube에서 overlap 영역을 제외한 중앙 영역만 추출하는 과정을 보여줍니다.

import os
import sys
import math
import time
import numpy as np
from itertools import product

# 상위 디렉토리(src)를 모듈 검색 경로에 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import DataLoader
from logger import logger
from config import OUTPUT_DATA_PATHS

def compute_subcube_parameters(full_length, S, O):
    """
    전체 영역 full_length에서 nominal subcube 중심 영역 크기 S와 overlap O를 고려하여,
    한 축에 대해 각 subcube의 load 영역(즉, subcube를 계산할 때 불러올 영역의 시작점과 끝점)을 계산합니다.
    
    Parameters
    ----------
    full_length : float
        전체 도메인 크기 (예: 205 Mpc/h).
    S : float
        subcube의 중심 영역 크기 (예: 90 Mpc/h).
    O : float
        각 subcube 경계에서의 overlap 길이 (예: 3 Mpc/h).
    
    Returns
    -------
    centers : list of float
        각 subcube의 중심 좌표 (중심 영역의 중앙 값).
    load_windows : list of tuple
        각 subcube에 대해 실제 로드할 영역의 (start, end) 튜플.
        영역이 0 미만이거나 full_length를 초과하면 modulo 연산으로 wrap-around 처리를 합니다.
    """
    N = math.ceil(full_length / S)
    centers = []
    load_windows = []
    for i in range(N):
        center = i * S + S/2
        centers.append(center)
        # 중앙 영역의 하단 경계는 center - S/2이고, 여기에 양쪽 overlap을 추가하여 실제 로드할 영역은:
        load_start = (center - S/2 - O) % full_length
        load_end = (center + S/2 + O) % full_length
        load_windows.append((load_start, load_end))
    return centers, load_windows

def main():
    start_time = time.time()
    
    # ------------------------------------------------------------------
    # 기본 설정: 전체 도메인 크기, nominal subcube 중심 영역 크기, overlap 길이
    # ------------------------------------------------------------------
    full_length = 205.0   # 전체 도메인 (Mpc/h)
    S = 50.0              # 중심 영역 크기 (Mpc/h)
    O = 3.0               # overlap 길이 (Mpc/h); 예를 들어, Gaussian kernel h=1이면 주효과 범위 약 3h
    load_size = S + 2 * O  # 각 subcube에서 실제 로드할 영역의 크기
    
    logger.info("전체 도메인: %.2f Mpc/h, Nominal subcube 크기: %.2f Mpc/h, Overlap: %.2f Mpc/h", full_length, S, O)
    
    # 한 축에 대해 subcube load 영역 계산 (wrap-around 적용)
    centers, load_windows = compute_subcube_parameters(full_length, S, O)
    logger.info("한 축 subcube 중심: %s", centers)
    logger.info("한 축 subcube load window: %s", load_windows)
    
    # 여기서는 load_window의 시작점(즉, cube_origin)만 사용하여 subcube를 불러옵니다.
    # (각 subcube의 load 영역은 load_size 만큼이며, periodic 조건은 load_cube_worker_periodic()에서 처리됩니다.)
    cube_origins_axis = [w[0] for w in load_windows]
    logger.info("각 축에서 불러올 cube_origin: %s", cube_origins_axis)
    
    # 3D 전체 subcube origin 조합 생성
    subcube_origins = list(product(cube_origins_axis, repeat=3))
    logger.info("전체 subcube 개수: %d", len(subcube_origins))
    
    # DataLoader 초기화
    npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    loader = DataLoader(npy_folder, use_gpu=False)
    
    # 각 subcube에 대해 데이터 로드 및 중앙 영역 추출
    for idx, origin in enumerate(subcube_origins):
        logger.info("Subcube %d: 로드할 cube_origin = %s, 로드 크기 = %.2f Mpc/h", idx, origin, load_size)
        # periodic boundary 조건을 적용하여 subcube 데이터 로드
        subcube_data = loader.load_cube_data(
            cube_origin=origin,
            cube_size=load_size,
            full_length=full_length,
            sampling_rate=1.0,
            workers=48  # 필요에 따라 조정
        )
        if subcube_data is None:
            logger.error("Subcube %d 데이터 로드 실패.", idx)
            continue
        logger.info("Subcube %d 로드 완료, 데이터 shape: %s", idx, subcube_data.shape)
        
        # 중앙 영역 추출: 중심 영역은 load 영역에서 양쪽 overlap O를 제외한 영역.
        # 각 축에 대해 중앙 영역은 [origin + O, origin + O + S) (wrap-around 적용)
        central_mask = np.ones(len(subcube_data), dtype=bool)
        for dim in range(3):
            central_start = (origin[dim] + O) % full_length
            central_end = (central_start + S) % full_length
            if central_start < central_end:
                mask_dim = (subcube_data[:, dim] >= central_start) & (subcube_data[:, dim] < central_end)
            else:
                # wrap-around 발생: 값이 central_start 이상이거나 central_end 미만
                mask_dim = (subcube_data[:, dim] >= central_start) | (subcube_data[:, dim] < central_end)
            central_mask &= mask_dim
        
        central_data = subcube_data[central_mask]
        logger.info("Subcube %d 중앙 영역 데이터 shape: %s", idx, central_data.shape)
        # 여기서 각 subcube의 중앙 영역에 대한 density map 계산을 진행하거나,
        # 결과를 개별 파일(npy 또는 hdf5)로 저장하면 됩니다.
        # 예: np.save(f"subcube_{idx}_central.npy", central_data)
        
        # 예시로, 간단한 통계 출력:
        if len(central_data) > 0:
            data_mean = np.mean(central_data, axis=0)
            data_min = np.min(central_data, axis=0)
            data_max = np.max(central_data, axis=0)
            logger.info("Subcube %d 중앙 영역 통계: Mean: %s, Min: %s, Max: %s", idx, data_mean, data_min, data_max)
        else:
            logger.warning("Subcube %d 중앙 영역 데이터가 비어 있습니다.", idx)
    
    elapsed_time = time.time() - start_time
    logger.info("전체 subcube 데이터 로드 및 처리 완료 (총 소요시간: %.2f초)", elapsed_time)


if __name__ == '__main__':
    main()

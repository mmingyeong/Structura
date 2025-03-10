#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-07
# @Filename: density_ex.py
# structura/density_ex.py

import os
import sys
import time  # 실행 시간 측정을 위해 사용

# 현재 스크립트의 상위 디렉토리 (src)를 Python 모듈 경로에 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import logger
from density import DensityCalculator, save_density_map, save_parameters_info
from kernel import KernelFunctions
from data_loader import DataLoader
from config import OUTPUT_DATA_PATHS, RESULTS_DIR

# 평균 입자 간격 (baseline) [cMpc/h]
baseline = 0.082

# ---------------------------
# Coarser resolution settings:
# 예: 2배, 5배, 10배 → 각각 0.082 * 2, 0.082 * 5, 0.082 * 10
# ---------------------------
coarser_factors = [2, 5, 10]
coarser_spacing = [baseline * factor for factor in coarser_factors]

# ---------------------------
# Finer resolution settings:
# 예: 1/2, 1/5, 1/10 → 각각 0.082 / 2, 0.082 / 5, 0.082 / 10
# ---------------------------
finer_factors = [1/2, 1/5, 1/10]
finer_spacing = [baseline * factor for factor in finer_factors]

# 두 가지 해상도 설정을 딕셔너리 형태로 정리합니다.
grid_spacings = {
    "coarser": coarser_spacing,  # [0.164, 0.41, 0.82] cMpc/h
    "finer": finer_spacing       # [0.041, 0.0164, 0.0082] cMpc/h
}

print("Grid Spacings (cMpc/h):", grid_spacings)

# 각 커널 함수들을 (이름, 함수) 튜플 형태로 리스트에 저장합니다.
kernel_list = [
    ("gaussian", KernelFunctions.gaussian),
    ("uniform", KernelFunctions.uniform),
    ("epanechnikov", KernelFunctions.epanechnikov),
    ("triangular", KernelFunctions.triangular),
    ("quartic", KernelFunctions.quartic),
    ("triweight", KernelFunctions.triweight),
    ("cosine", KernelFunctions.cosine),
    ("logistic", KernelFunctions.logistic),
    ("sigmoid", KernelFunctions.sigmoid),
    ("laplacian", KernelFunctions.laplacian)
]

def main():
    """
    예시:
      1. Snapshot99 전체 3차원 데이터를 DataLoader를 이용해 로드합니다.
      2. DensityCalculator와 KernelFunctions (여기서는 가우시안 커널)를 사용하여 전체 3D 밀도 맵을 계산합니다.
      3. 계산된 3D 밀도 맵을 RESULTS_DIR에 저장하고, 사용된 파라미터 정보를 JSON 파일로 저장합니다.
      
      (플롯 관련 코드는 생략됩니다.)
    """
    start_time = time.time()  # 실행 시작 시각 기록

    # ------------------------------------------------------------------
    # Snapshot99 전체 데이터 로드 (필터링 없이)
    # ------------------------------------------------------------------
    npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    logger.info("Snapshot99 데이터 폴더: %s", npy_folder)
    logger.info("Output directory: %s", RESULTS_DIR)
    
    logger.info("전체 Snapshot99 데이터를 로드합니다...")
    loader = DataLoader(npy_folder)
    positions = loader.load_all_chunks(sampling_rate=0.0001)
    logger.info("데이터 로드 완료, positions shape: %s", positions.shape)

    # ------------------------------------------------------------------
    # Density Map 계산 설정 (전체 영역)
    # ------------------------------------------------------------------
    grid_bounds = {
        'x': (0, 30),
        'y': (0, 30),
        'z': (0, 30)
    }
    grid_spacing = (1, 1, 1)  # 각 축 1 cMpc/h 해상도

    # h 값을 None으로 지정하면, DensityCalculator가 Silverman's rule로 최적의 h를 자동 계산합니다.
    h = 1.0

    logger.info("Density 계산을 위한 grid_bounds: %s, grid_spacing: %s", grid_bounds, grid_spacing)

    # DensityCalculator 객체 생성
    calculator = DensityCalculator(positions, grid_bounds, grid_spacing)
    
    # 여기서는 kernel_list에서 "gaussian" 커널을 선택합니다.
    selected_kernel_name, selected_kernel_func = "gaussian", KernelFunctions.gaussian

    logger.info("3D 밀도 맵 계산 시작...")
    x_centers, y_centers, z_centers, density_map, used_h = calculator.calculate_density_map(selected_kernel_func, h, return_used_h=True)
    logger.info("3D 밀도 맵 계산 완료. (사용된 h: %.6f)", used_h)

    # ------------------------------------------------------------------
    # 밀도 맵 저장 (RESULTS_DIR에 저장)
    # ------------------------------------------------------------------
    save_density_map(
        density_map,
        filename=None,
        data_name="TNG300_snapshot99",
        kernel_name=selected_kernel_name,
        h=used_h,
        folder=RESULTS_DIR,
        file_format="npy"
    )
    logger.info("계산된 3D 밀도 맵이 저장되었습니다.")

    # ------------------------------------------------------------------
    # 밀도 계산에 사용된 파라미터 정보 저장 (JSON 파일)
    # ------------------------------------------------------------------
    parameters_info = {
        "Calculation Info": {
            "grid_bounds": grid_bounds,
            "grid_spacing": grid_spacing,
            "kernel": selected_kernel_name,
            "used_h": used_h
        },
        "Simulation Info": {
            "box_size": 205,
            "data_unit": "cMpc/h"
        },
        "Paths": {
            "input_folder": npy_folder,
            "results_dir": RESULTS_DIR
        }
    }
    save_parameters_info(parameters_info, filename=None, folder=RESULTS_DIR)
    logger.info("계산에 사용된 파라미터 정보가 저장되었습니다.")

    end_time = time.time()  # 실행 종료 시각 기록
    elapsed_time = end_time - start_time  # 총 실행 시간 계산
    logger.info("전체 실행 시간: %.2f초", elapsed_time)

if __name__ == "__main__":
    main()

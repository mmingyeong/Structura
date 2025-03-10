#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-07
# @Filename: fft_kde_ex.py
# structura/fft_kde_ex.py

import os
import sys
import time

# 현재 스크립트의 상위 디렉토리 (src)를 Python 모듈 경로에 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import logger
from data_loader import DataLoader
from config import OUTPUT_DATA_PATHS, RESULTS_DIR

def main():
    start_time = time.time()

    # ------------------------------------------------------------------
    # 데이터 로드: TNG300_snapshot99 데이터 폴더에서 npy 파일들을 불러옵니다.
    # ------------------------------------------------------------------
    npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    logger.info("Snapshot99 데이터 폴더: %s", npy_folder)
    logger.info("Output directory: %s", RESULTS_DIR)
    
    logger.info("전체 Snapshot99 데이터를 로드합니다...")
    loader = DataLoader(npy_folder)
    # 아주 작은 sampling_rate로 전체 데이터를 로드합니다.
    positions = loader.load_all_chunks()
    logger.info("데이터 로드 완료, positions shape: %s", positions.shape)
"""
    # ------------------------------------------------------------------
    # FFT 기반 밀도 계산 조건 설정
    # ------------------------------------------------------------------
    # 여기서는 전체 영역에 대해 (0,205)x(0,205)x(0,205) 범위로 계산합니다.
    grid_bounds = {'x': (0, 205), 'y': (0, 205), 'z': (0, 205)}
    grid_spacing = (1, 1, 1)  # 각 축 1 cMpc/h 해상도
    h = 1.0  # 커널 밴드위스 (필요에 따라 조정 가능)

    logger.info("FFT 기반 밀도 계산을 위한 grid_bounds: %s, grid_spacing: %s", grid_bounds, grid_spacing)

    # ------------------------------------------------------------------
    # FFTKDE 객체 생성 및 밀도 지도 계산
    # ------------------------------------------------------------------
    fft_kde = FFTKDE(positions, grid_bounds, grid_spacing,
                      kernel_func=KernelFunctions.gaussian, h=h)
    logger.info("FFT 기반 커널 밀도 추정 계산 시작...")
    x_centers, y_centers, z_centers, density_map = fft_kde.compute_density()
    logger.info("FFT 기반 커널 밀도 추정 완료.")

    # ------------------------------------------------------------------
    # 결과 저장: density map과 계산 파라미터 정보를 RESULTS_DIR에 저장
    # ------------------------------------------------------------------
    from save_density_map import save_density_map, save_parameters_info

    save_density_map(
        density_map,
        filename=None,
        data_name="TNG300_snapshot99_FFT",
        kernel_name="gaussian",
        h=h,
        folder=RESULTS_DIR,
        file_format="npy"
    )
    logger.info("계산된 3D 밀도 맵이 저장되었습니다.")

    parameters_info = {
        "Calculation Info": {
            "grid_bounds": grid_bounds,
            "grid_spacing": grid_spacing,
            "kernel": "gaussian",
            "used_h": h
        },
        "Simulation Info": {
            "box_size": 205,
            "data_unit": "cMpc/h"
        },
        "Paths": {
            "input_folder": npy_folder,
            "results_dir": RESULTS_DIR
        }
    }"
    "
    save_parameters_info(parameters_info, filename=None, folder=RESULTS_DIR)
    logger.info("계산에 사용된 파라미터 정보가 저장되었습니다.")
"""
    elapsed_time = time.time() - start_time
    logger.info("전체 실행 시간: %.2f초", elapsed_time)

if __name__ == '__main__':
    main()

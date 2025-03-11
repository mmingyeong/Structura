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
import cProfile
import pstats
import io

# 상위 디렉토리 (src)를 모듈 검색 경로에 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import logger
from data_loader import DataLoader
from config import OUTPUT_DATA_PATHS, MAP_RESULTS_DIR

# 아래 FFTKDE, KernelFunctions, save_density_map, save_parameters_info 모듈은
# 프로젝트 내에 정의되어 있다고 가정합니다.
from fft_kde import FFTKDE, KernelFunctions
from save_density_map import save_density_map, save_parameters_info

def main():
    start_time = time.time()

    # ------------------------------------------------------------------
    # 설정: Snapshot99 데이터 폴더 및 결과 저장 경로
    # ------------------------------------------------------------------
    npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    logger.info("Snapshot99 데이터 폴더: %s", npy_folder)
    logger.info("Output directory: %s", MAP_RESULTS_DIR)
    
    # ------------------------------------------------------------------
    # Test: periodic boundary condition을 적용한 subcube 데이터 로드
    # ------------------------------------------------------------------
    # 예시로, 전체 도메인이 205 Mpc/h인 경우, cube_origin이 (100,100,100)이고,
    # cube_size를 50 Mpc/h로 지정하면, 각 축에서 100+50=150 Mpc/h 범위가 선택됩니다.
    # (periodic 조건이 적용되어, 도메인 경계를 넘지 않는 경우)
    cube_origin = (100, 100, 100)  # subcube의 하단 모서리 좌표
    cube_size = 10                 # subcube 중심 영역 크기 (GPU가 다룰 수 있는 최대 영역)
    full_length = 205.0            # 전체 도메인 크기 (Mpc/h)
    sampling_rate = 1.0            # 전체 데이터 사용 (샘플링 비율 1.0)
    
    logger.info("Loading subcube data (origin: %s, size: %s, full_length: %.1f)...", cube_origin, cube_size, full_length)
    loader = DataLoader(npy_folder, use_gpu=False)
    subcube_positions = loader.load_cube_data(
        cube_origin=cube_origin,
        cube_size=cube_size,
        full_length=full_length,
        sampling_rate=sampling_rate,
        workers=48   # 테스트용 worker 수; 환경에 맞게 조정
    )
    logger.info("Loaded subcube data shape: %s", subcube_positions.shape)
    
    # ------------------------------------------------------------------
    # FFT 기반 밀도 계산 조건 설정 (subcube 영역에 맞게)
    # ------------------------------------------------------------------
    # periodic 조건을 고려하여, grid_bounds는 (cube_origin, cube_origin+cube_size)로 설정합니다.
    grid_bounds = {
        'x': (cube_origin[0], (cube_origin[0] + cube_size) % full_length),
        'y': (cube_origin[1], (cube_origin[1] + cube_size) % full_length),
        'z': (cube_origin[2], (cube_origin[2] + cube_size) % full_length)
    }
    grid_spacing = (1, 1, 1)  # 각 축 1 cMpc/h 해상도
    h = 1.0  # Gaussian 커널 밴드위스 (표준편차: 1 cMpc/h)
    
    logger.info("FFT 기반 밀도 계산을 위한 grid_bounds: %s, grid_spacing: %s", grid_bounds, grid_spacing)
    
    # FFTKDE 객체 생성 및 밀도 map 계산
    fft_kde = FFTKDE(subcube_positions, grid_bounds, grid_spacing,
                      kernel_func=KernelFunctions.gaussian, h=h)
    logger.info("FFT 기반 커널 밀도 추정 계산 시작...")
    x_centers, y_centers, z_centers, density_map = fft_kde.compute_density()
    logger.info("FFT 기반 커널 밀도 추정 완료.")
    
    # ------------------------------------------------------------------
    # 결과 저장: 밀도 map과 계산 파라미터 정보를 MAP_RESULTS_DIR에 저장
    # ------------------------------------------------------------------
    save_density_map(
        density_map,
        filename=None,
        data_name="TNG300_snapshot99_subcube_FFT",
        kernel_name="gaussian",
        h=h,
        folder=MAP_RESULTS_DIR,
        file_format="npy"
    )
    logger.info("계산된 subcube 3D 밀도 맵이 저장되었습니다.")
    
    parameters_info = {
        "Calculation Info": {
            "grid_bounds": grid_bounds,
            "grid_spacing": grid_spacing,
            "kernel": "gaussian",
            "used_h": h,
            "method": FFTKDE
        },
        "Simulation Info": {
            "box_size": full_length,
            "data_unit": "cMpc/h"
        },
        "Paths": {
            "input_folder": npy_folder,
            "results_dir": MAP_RESULTS_DIR
        }
    }
    save_parameters_info(parameters_info, filename=None, folder=MAP_RESULTS_DIR)
    logger.info("계산에 사용된 파라미터 정보가 저장되었습니다.")
    
    elapsed_time = time.time() - start_time
    logger.info("전체 실행 시간: %.2f초", elapsed_time)


if __name__ == '__main__':
    # cProfile을 이용한 성능 프로파일링
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    with open("fft_profile_results.txt", "w") as f:
        f.write(s.getvalue())
    print("Profiling results have been saved to fft_profile_results.txt.")

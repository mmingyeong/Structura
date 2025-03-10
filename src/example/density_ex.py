#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-07
# @Filename: density_calculator_ex.py
# structura/density_calculator_ex.py

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

# density.py 안에 정의된 DensityCalculator를 import
from density import DensityCalculator
# kernel.py 안에 정의된 KernelFunctions (예: gaussian 커널) 사용 가능
from kernel import KernelFunctions

# 결과 저장용 함수
from save_density_map import save_density_map, save_parameters_info

def main():
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. 설정: Snapshot99 데이터 폴더 및 결과 저장 경로
    # ------------------------------------------------------------------
    npy_folder = OUTPUT_DATA_PATHS["TNG300_snapshot99"]
    logger.info("Snapshot99 데이터 폴더: %s", npy_folder)
    logger.info("Output directory: %s", MAP_RESULTS_DIR)

    # ------------------------------------------------------------------
    # 2. Test: subcube 데이터 로드
    # ------------------------------------------------------------------
    # 예: 전체 도메인이 205 Mpc/h이고, (100,100,100)에서 10 Mpc/h 범위를 추출
    cube_origin = (100, 100, 100)
    cube_size = 10
    full_length = 205.0
    sampling_rate = 1.0  # 전체 입자 사용 (샘플링 비율)

    logger.info("Loading subcube data (origin: %s, size: %s, full_length: %.1f)...",
                cube_origin, cube_size, full_length)

    loader = DataLoader(npy_folder, use_gpu=False)
    subcube_positions = loader.load_cube_data(
        cube_origin=cube_origin,
        cube_size=cube_size,
        full_length=full_length,
        sampling_rate=sampling_rate,
        workers=48  # 환경에 맞춰 조정
    )
    logger.info("Loaded subcube data shape: %s", subcube_positions.shape)

    # ------------------------------------------------------------------
    # 3. 밀도 계산을 위한 파라미터 설정
    # ------------------------------------------------------------------
    # periodic 조건 고려: grid_bounds = (cube_origin, cube_origin + cube_size)
    grid_bounds = {
        'x': (cube_origin[0], (cube_origin[0] + cube_size) % full_length),
        'y': (cube_origin[1], (cube_origin[1] + cube_size) % full_length),
        'z': (cube_origin[2], (cube_origin[2] + cube_size) % full_length)
    }
    grid_spacing = (1, 1, 1)  # 각 축 1 cMpc/h 해상도
    kernel_func = KernelFunctions.gaussian  # 가우시안 커널 예시
    h = 1.0  # 밴드위스 (필요 시 None으로 설정하면 Silverman's rule로 자동 계산)

    logger.info("Density map 계산을 위한 grid_bounds: %s, grid_spacing: %s", grid_bounds, grid_spacing)

    # ------------------------------------------------------------------
    # 4. DensityCalculator를 이용한 밀도 맵 계산
    # ------------------------------------------------------------------
    # use_gpu=True로 설정하면, GPU가 설치되어 있고 cupy가 import 가능한 경우 GPU로 연산
    density_calc = DensityCalculator(
        particles=subcube_positions,
        grid_bounds=grid_bounds,
        grid_spacing=grid_spacing,
        use_gpu=False  # GPU 사용 여부; True로 변경 가능
    )
    logger.info("DensityCalculator 객체가 생성되었습니다.")

    # calculate_density_map 호출 시:
    #  - kernel_func: 사용할 커널 함수
    #  - h: 커널 밴드위스 (None이면 Silverman's rule로 자동 추정)
    x_centers, y_centers, z_centers, density_map = density_calc.calculate_density_map(
        kernel_func=kernel_func,
        h=h
    )
    logger.info("DensityCalculator를 이용한 밀도 계산이 완료되었습니다. density_map shape: %s", density_map.shape)

    # ------------------------------------------------------------------
    # 5. 결과 저장
    # ------------------------------------------------------------------
    # (1) 밀도 맵 파일 저장
    save_density_map(
        density_map,
        filename=None,
        data_name="TNG300_snapshot99_subcube_DensityCalc",
        grid_spacing=grid_spacing,
        kernel_name="gaussian",
        h=h,
        folder=MAP_RESULTS_DIR,
        file_format="npy"
    )
    logger.info("계산된 subcube 3D 밀도 맵이 저장되었습니다.")

    # (2) 파라미터 정보 JSON 저장
    parameters_info = {
        "Calculation Info": {
            "grid_bounds": grid_bounds,
            "grid_spacing": grid_spacing,
            "kernel": "gaussian",
            "used_h": h,
            "method": DensityCalculator
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
    with open("density_calculator_profile_results.txt", "w") as f:
        f.write(s.getvalue())
    print("Profiling results have been saved to density_calculator_profile_results.txt.")

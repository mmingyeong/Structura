#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-07
# @Filename: plot_density_map.py
# structura/plot_density_map.py

import os
import sys
import time  # 실행 시간 측정을 위해 사용
import cProfile
import pstats
import io

# 현재 스크립트의 상위 디렉토리 (src)를 Python 모듈 경로에 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import json
from logger import logger
from config import MAP_RESULTS_DIR

def load_parameters_info(folder, filename="parameters_info.json"):
    """
    지정된 폴더에서 parameters_info JSON 파일을 불러옵니다.
    
    Parameters:
        folder (str): 파라미터 정보가 저장된 폴더 경로.
        filename (str): 불러올 JSON 파일 이름. 기본값은 "parameters_info.json".
    
    Returns:
        dict: 파라미터 정보 딕셔너리. 파일이 없거나 에러 발생시 None.
    """
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        logger.error("파라미터 정보 JSON 파일이 존재하지 않습니다: %s", file_path)
        return None
    try:
        with open(file_path, "r") as f:
            info = json.load(f)
        logger.info("파라미터 정보 JSON 파일 로드 완료: %s", file_path)
        return info
    except Exception as e:
        logger.error("JSON 파일 로드 중 오류 발생: %s", e)
        return None

def main():
    start_time = time.time()  # 실행 시작 시각 기록

    try:
        # 불러올 density map npy 파일 경로 (실제 파일 이름에 맞게 수정)
        density_map_file = os.path.join(MAP_RESULTS_DIR, "density_map_TNG300_snapshot99_subcube_FFT_None_gaussian_h1.0000_20250310_193731.npy")
        if not os.path.exists(density_map_file):
            logger.error("밀도 맵 파일이 존재하지 않습니다: %s", density_map_file)
            return

        # 3D 밀도 맵 불러오기
        density_map = np.load(density_map_file)
        logger.info("밀도 맵 파일 로드 완료, shape: %s", density_map.shape)

        # parameters_info JSON 파일 불러오기
        params = load_parameters_info(MAP_RESULTS_DIR, filename="parameters_info_20250310_193731.json")
        if params is None:
            # JSON 파일이 없으면 밀도 맵의 shape에 기반하여 grid parameters 설정
            logger.warning("파라미터 정보 JSON 파일이 없으므로 density_map shape에 따른 grid parameters를 사용합니다.")
            grid_bounds = {
                'x': (0, density_map.shape[0]),
                'y': (0, density_map.shape[1]),
                'z': (0, density_map.shape[2])
            }
            grid_spacing = (1, 1, 1)
        else:
            grid_bounds = params.get("Calculation Info", {}).get("grid_bounds", {'x': (0,205), 'y': (0,205), 'z': (0,205)})
            grid_spacing = params.get("Calculation Info", {}).get("grid_spacing", (1,1,1))
            # JSON 저장 시 튜플은 리스트로 저장될 수 있으므로 변환
            grid_bounds = {k: tuple(v) for k, v in grid_bounds.items()}
            grid_spacing = tuple(grid_spacing)

        logger.info("사용된 grid_bounds: %s, grid_spacing: %s", grid_bounds, grid_spacing)

        # 격자 셀 중심 좌표 재구성
        x_centers = np.arange(grid_bounds['x'][0] + grid_spacing[0]/2, grid_bounds['x'][1], grid_spacing[0])
        y_centers = np.arange(grid_bounds['y'][0] + grid_spacing[1]/2, grid_bounds['y'][1], grid_spacing[1])
        z_centers = np.arange(grid_bounds['z'][0] + grid_spacing[2]/2, grid_bounds['z'][1], grid_spacing[2])
        
        # x축에서 100~110 cMpc/h 범위에 해당하는 셀 선택
        x_mask = (x_centers >= 0) & (x_centers <= 10)
        if np.sum(x_mask) == 0:
            logger.error("x축 100~110 범위에 해당하는 셀이 없습니다.")
            return
        logger.info("선택된 x축 셀 개수: %d", np.sum(x_mask))
        
        # 선택된 x 범위에 대해 x축 합산하여 YZ 평면 프로젝션 계산
        projection_yz = np.sum(density_map[x_mask, :, :], axis=0)
        
        # YZ 평면 프로젝션 플롯 생성 (pcolormesh 사용)
        plt.figure(figsize=(8, 6))
        # x축은 y_centers, y축은 z_centers, density는 전치(transpose)하여 올바른 방향으로 매핑
        plt.pcolormesh(y_centers, z_centers, projection_yz.T, shading='auto', cmap='viridis')
        plt.xlabel("Y (cMpc/h)")
        plt.ylabel("Z (cMpc/h)")
        plt.title("Density Map YZ Projection (x=100-110 cMpc/h)")
        plt.colorbar(label="Density (summed)")
        plt.tight_layout()
        
        # 플롯 저장
        output_plot = os.path.join(MAP_RESULTS_DIR, "density_map_yz_projection_x100_110.png")
        plt.savefig(output_plot)
        plt.close()
        logger.info("x축 100~110 범위의 밀도 맵 YZ 프로젝션 플롯이 저장되었습니다: %s", output_plot)
    except Exception as e:
        logger.error("오류 발생: %s", e)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("전체 실행 시간: %.2f초", elapsed_time)

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    with open("plot_profile_results.txt", "w") as f:
        f.write(s.getvalue())
    print("Profiling results have been saved to plot_profile_results.txt.")

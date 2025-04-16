#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import cProfile
import pstats
import logging
import numpy as np
import matplotlib.pyplot as plt

# (1) GPU FFT 사용 가능 여부 확인 (선택 사항)
try:
    import cupy as cp
    from cupy.cuda import runtime

    def select_best_gpu():
        num_devices = runtime.getDeviceCount()
        best_device = 0
        best_score = 0
        for device in range(num_devices):
            props = runtime.getDeviceProperties(device)
            score = props['multiProcessorCount'] * props['clockRate']
            if score > best_score:
                best_score = score
                best_device = device
        return best_device

    GPU_DEVICE = select_best_gpu()
    use_gpu = True
except ImportError:
    use_gpu = False
    GPU_DEVICE = None

# (2) 이미 만들어둔 FFTKDE2D (density 클래스)와 2D 커널 함수들을 import
from density import FFTKDE2D  # 예: 다른 파일에서 정의된 FFTKDE2D 클래스
from kernel_2d import KernelFunctions2D  # 2D 가우시안, 에파네치니코프 등

##############################################################################
# Logging Setup
##############################################################################
def setup_logging(log_file="fft_kde_2d.log"):
    """
    Configure logging to both a file and the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

##############################################################################
# Main Execution
##############################################################################
def main():
    logger = setup_logging()
    pr = cProfile.Profile()
    total_start = time.time()

    pr.enable()
    logger.info("Beginning 2D FFT-KDE density calculation...")

    #---------------------------------------------------------------------
    # 1. 파티클(2D 점들) 데이터 로드
    #---------------------------------------------------------------------
    particle_file = "particles_nonuniform.npy"
    if not os.path.exists(particle_file):
        logger.error(f"Particle file {particle_file} not found.")
        return
    particles = np.load(particle_file)
    logger.info(f"Loaded particle data of shape: {particles.shape}")

    #---------------------------------------------------------------------
    # 2. 그리드/도메인/밴드폭 설정
    #---------------------------------------------------------------------
    grid_bounds = {'x': (0.0, 100.0), 'y': (0.0, 100.0)}
    grid_spacing = (0.5, 0.5)
    h = 3.0  # 커널 밴드폭

    #---------------------------------------------------------------------
    # 3. 원하는 2D 커널 함수 선택
    #    (여기서는 가우시안 예시, 다른 커널로 바꾸려면 epanechnikov 등 사용)
    #---------------------------------------------------------------------
    kernel_func = KernelFunctions2D.gaussian

    #---------------------------------------------------------------------
    # 4. FFTKDE2D 인스턴스 생성 & 밀도 계산
    #    (FFTKDE2D 클래스는 이미 다른 모듈에서 정의되어 있다고 가정)
    #---------------------------------------------------------------------
    kde = FFTKDE2D(
        particles=particles,
        grid_bounds=grid_bounds,
        grid_spacing=grid_spacing,
        kernel_func=kernel_func,
        h=h
    )
    x_centers, y_centers, density_map = kde.compute_density()
    logger.info("Density map computed successfully.")

    #---------------------------------------------------------------------
    # 5. 결과 시각화 및 저장
    #---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = (x_centers[0], x_centers[-1], y_centers[0], y_centers[-1])
    im = ax.imshow(
        density_map.T, origin='lower', extent=extent, aspect='auto', cmap='viridis'
    )
    fig.colorbar(im, ax=ax, label="Density")
    ax.set_title("2D Density Map (FFT-based KDE, Custom Kernel)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    output_png = "fft_kde_result.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Density map saved as '{output_png}'.")

    #---------------------------------------------------------------------
    # 6. cProfile 결과 출력/저장
    #---------------------------------------------------------------------
    pr.disable()
    profile_file = "profile_results.txt"
    with open(profile_file, "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats("cumtime")
        ps.print_stats()
    logger.info(f"Profile results saved to '{profile_file}'.")

    total_elapsed = time.time() - total_start
    logger.info(f"Total execution time: {total_elapsed:.4f} seconds.")

if __name__ == "__main__":
    main()

import os
import time
import numpy as np
import logging
import cProfile
import pstats
import matplotlib.pyplot as plt
from tqdm import tqdm

# 아래 import 경로를 사용자 환경에 맞춰 수정:
# DensityCalculator2D와 auto_tune_config 등이 들어있는 파이썬 모듈(예: density_calc.py)에서 import
# 여기서는 동일 파일 내 정의라고 가정.
from joblib import Parallel, delayed  # 병렬 처리
from density import DensityCalculator2D, use_gpu  # 예시: 'density.py'라는 모듈

GRID_SPACING = (0.5, 0.5)

############################################
# GPU용 가우시안 커널 함수 예시 (수정)
############################################
try:
    import cupy as cp
except ImportError:
    pass

def gaussian_kernel_gpu(r2, h2):
    """
    GPU용 가우시안 커널:
    r2: cupy.ndarray(거리 제곱)
    h2: float (bandwidth^2)
    반환: 각 원소별 exp(-0.5*r2/h2) / (2*pi*h2)
    """
    norm = 1.0 / (2.0 * cp.pi * h2)
    return norm * cp.exp(-0.5 * r2 / h2)

def test_density_calculation(
    npy_path="particles.npy",
    output_npy="density_map.npy",
    output_png="density_map.png",
    auto_tune=True,
):
    """
    1. NPY 파일에서 파티클 로드
    2. DensityCalculator2D 인스턴스 생성
    3. calculate_density_map_kdtree_gpu 실행
    4. 결과 저장 (npy, png)
    5. 전체 실행 시간 출력
    6. 주요 단계별 progress bar 표시
    """

    #############################
    # 로깅 설정
    #############################
    logging.basicConfig(
        filename="density_calculation.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # 전체 실행 시간 측정
    start_time = time.time()

    # 단계별 진행상황 표시를 위해 총 4단계로 가정
    with tqdm(total=4, desc="Density Calculation") as pbar:
        #############################
        # 1) 파티클 데이터 로드
        #############################
        if not os.path.exists(npy_path):
            logger.error(f"File not found: {npy_path}")
            return
        particles = np.load(npy_path)
        logger.info(f"Loaded particles from {npy_path}, shape={particles.shape}")
        pbar.update(1)

        #############################
        # 2) DensityCalculator2D 생성
        #    grid_bounds, grid_spacing은 상황에 맞게 수정
        #############################
        grid_bounds = {
            'x': (0.0, 100.0),
            'y': (0.0, 100.0)
        }
        grid_spacing = GRID_SPACING

        calc = DensityCalculator2D(
            particles,
            grid_bounds,
            grid_spacing
        )
        pbar.update(1)

        #############################
        # 3) cProfile 프로파일링 + GPU 계산
        #############################
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            x_centers, y_centers, density_map = calc.calculate_density_map_kdtree_gpu(
                kernel_func_gpu=gaussian_kernel_gpu,
                h=None,         # auto_tune=True 면 내부에서 추정 가능
                cutoff=None,    # 마찬가지
                neighbor_batch_size=None,
                leafsize=16,
                batch_size_x=16,
                batch_size_y=16,
                n_jobs=4,       # CPU 병렬 처리 스레드 수
                auto_tune=auto_tune
            )
        except RuntimeError as e:
            logger.exception("GPU not available or error occurred.")
            return

        profiler.disable()
        pbar.update(1)

        # 프로파일 결과 저장
        with open("density_profile.txt", "w") as f:
            ps = pstats.Stats(profiler, stream=f).sort_stats("cumtime")
            ps.print_stats()
        logger.info("Profile results saved to density_profile.txt")

        #############################
        # 4) 결과 저장 (NPY, PNG)
        #############################
        # npy 저장
        np.save(output_npy, density_map)
        logger.info(f"Density map saved to {output_npy}, shape={density_map.shape}")

        # PNG 저장
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(
            density_map.T,  # X→열, Y→행이므로 Transpose
            origin="lower",
            extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
            cmap="viridis"
        )
        ax.set_title("Density Map (KD-Tree + GPU)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(cax, ax=ax, label="Density")
        plt.savefig(output_png, dpi=150)
        plt.close(fig)

        logger.info(f"Density plot saved to {output_png}")
        logger.info("Density calculation test completed successfully.")
        pbar.update(1)

    # 전체 실행 시간 출력
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Total Execution Time: {elapsed:.2f} seconds.")
    print(f"Total Execution Time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    # 예시 실행
    test_density_calculation(
        npy_path="particles_nonuniform.npy",       # 실제 사용자 NPY 파일 경로
        output_npy="nonuniform_density_map.npy",
        output_png="nonuniform_density_map.png",
        auto_tune=True
    )

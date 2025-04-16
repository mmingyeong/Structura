"""
--------------------------------------------------------------------------------
Recommended Data Specifications (for ~10-minute runtime):

- Particle Count (N): up to around 50k–500k in 2D (float64).
  - Above 1e6 particles may push total runtime beyond 10 minutes, depending on
    hardware (CPU cores, GPU model, memory bandwidth, etc.).
- Grid Size: Nx x Ny up to ~2e4–2.5e5 total grid cells.
  - Example: 200x200 = 40k cells, 500x500 = 250k cells.
- Kernel Bandwidth (h):
  - Typically ~1/50 to 1/10 of the domain size, so that cutoff = 4*h doesn't
    inflate neighbor searches excessively.
- Memory Constraints:
  - With 16–32 GB of RAM and a mid-range GPU, these guidelines are typically
    feasible without exceeding ~10 minutes of compute time.

Example:
  - Domain = 100 x 100
  - ~300k particles
  - Grid spacing (dx=dy=0.5) -> 200x200 grid cells = 40k
  - Bandwidth h ~ 2.0 (cutoff = 8.0)
  - This setup often finishes in under 10 minutes on an 8–16 core CPU with a
    decent GPU (e.g. RTX 3060+).

--------------------------------------------------------------------------------
"""


import numpy as np
import psutil  # 시스템 메모리 조회
from scipy.spatial import cKDTree

try:
    import cupy as cp
    use_gpu = True
except ImportError:
    use_gpu = False

from joblib import Parallel, delayed  # Python 병렬 처리

############################################
# 자동 파라미터 튜닝 알고리즘 (Heuristic)
############################################
def auto_tune_config(
    particles,
    grid_bounds,
    grid_spacing,
    h=None,
    fraction_for_cutoff=4.0,
    max_memory_fraction=0.5,
    max_jobs=8,
):
    """
    데이터셋(파티클)과 격자 정보를 기반으로,
    neighbor_batch_size, leafsize, batch_size_x, batch_size_y, n_jobs, cutoff 등을
    대략적으로 추정하는 간단한 휴리스틱 알고리즘.

    :param particles: (N, 2) 파티클 좌표
    :param grid_bounds: {'x': (xmin, xmax), 'y': (ymin, ymax)}
    :param grid_spacing: (dx, dy)
    :param h: 대역폭 (미리 알고 있으면 전달)
    :param fraction_for_cutoff: cutoff를 대역폭의 몇 배로 할지 (기본 4)
    :param max_memory_fraction: 시스템 메모리 중 사용할 최대 비율 (기본 0.5 => 50%)
    :param max_jobs: CPU 병렬화 시 최대 코어/스레드 수
    :return: dict 형태로 추정된 파라미터 {
        'neighbor_batch_size', 'leafsize',
        'batch_size_x', 'batch_size_y', 'n_jobs', 'cutoff'
    }
    """
    n_particles = len(particles)
    (xmin, xmax) = grid_bounds['x']
    (ymin, ymax) = grid_bounds['y']
    dx, dy = grid_spacing

    # 1) cutoff 추정
    #    만약 h가 None이면, 간단히 전체 영역과 파티클 밀도를 기준으로 추정.
    if h is None:
        # 간단한 방식: 영역 대각선 / 50 정도로 가정.
        # (더 고급 방법: 샘플 파티클간 평균 거리 계산)
        box_diag = ((xmax - xmin)**2 + (ymax - ymin)**2)**0.5
        # 대략 1% 정도?
        h_est = box_diag / 50.0
        h = max(1e-5, h_est)

    cutoff = fraction_for_cutoff * h

    # 2) leafsize 추정
    #    KD-Tree leafsize는 대체로 16~64 범위를 많이 사용.
    #    파티클이 매우 많다면 leafsize도 크게.
    #    예) sqrt(n_particles)/10, 범위 제한
    ls_est = int(max(4, min(64, (n_particles**0.5) / 10)))
    leafsize = ls_est

    # 3) batch_size_x, batch_size_y 추정
    #    격자 크기에 따라 너무 큰 배치를 쓰면 메모리 부담, 너무 작으면 반복문 오버헤드.
    nx = int((xmax - xmin) / dx + 0.5)  # 대략 격자점 수
    ny = int((ymax - ymin) / dy + 0.5)
    # 대략 격자를 32~64 개 블록으로 나누도록.
    # 예) sqrt(nx) ~ batch_size_x, 단 최소 16 이상, 최대 128 정도.
    import math
    bx_est = max(8, min(128, int(math.sqrt(nx) + 0.5)))
    by_est = max(8, min(128, int(math.sqrt(ny) + 0.5)))

    # 4) neighbor_batch_size 추정(메모리 기반)
    #    시스템 메모리 중 일정 비율(max_memory_fraction) 사용 가능하다고 가정.
    total_mem = psutil.virtual_memory().total
    usable_mem = total_mem * max_memory_fraction
    # 대략 한 파티클당 32~64바이트(오버헤드 포함)로 추정.
    # 안전하게 잡으려면 한 배치에 64MB~256MB 수준으로 제한.
    # n_batch = usable_mem / 128_000_000(128MB)

    per_particle_bytes = 64.0
    approximate_batch_particles = int(usable_mem / per_particle_bytes / 8)  # 1/8 여유
    neighbor_batch_size = max(10000, min(n_particles, approximate_batch_particles))

    # 5) n_jobs 추정
    #    물리 코어 수 (혹은 논리 코어 수)를 보고 결정. joblib는 GIL 우회 가능.
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    n_jobs = min(max_jobs, cpu_count)

    return {
        'neighbor_batch_size': neighbor_batch_size,
        'leafsize': leafsize,
        'batch_size_x': bx_est,
        'batch_size_y': by_est,
        'n_jobs': n_jobs,
        'cutoff': cutoff,
        'h': h,
    }

class DensityCalculator2D:
    def __init__(
        self,
        particles,
        grid_bounds,
        grid_spacing,
    ):
        """
        :param particles: (N, 2) 형태의 파티클 좌표 (float)
        :param grid_bounds: {'x': (xmin, xmax), 'y': (ymin, ymax)}
        :param grid_spacing: (dx, dy)
        """
        self.particles = np.ascontiguousarray(particles)
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing

    def calculate_density_map_kdtree_gpu(
        self,
        kernel_func_gpu,
        h=None,
        cutoff=None,
        neighbor_batch_size=None,
        leafsize=16,
        batch_size_x=16,
        batch_size_y=16,
        n_jobs=1,
        auto_tune=False,
    ):
        """
        KD-Tree로 cutoff 이내 파티클을 찾은 후,
        CPU에서 이웃 인덱스를 검색하고,
        GPU(cupy)에서 batch 단위로 커널을 합산.

        auto_tune=True 인 경우, 데이터를 기반으로
        neighbor_batch_size, leafsize, batch_size_x, batch_size_y, n_jobs, cutoff 등을
        동적으로 추정.
        """
        if not use_gpu:
            raise RuntimeError("GPU (cupy) is unavailable. Install cupy or check your CUDA environment.")

        (xmin, xmax) = self.grid_bounds['x']
        (ymin, ymax) = self.grid_bounds['y']

        if auto_tune:
            # 자동 튜닝
            params = auto_tune_config(
                self.particles,
                self.grid_bounds,
                self.grid_spacing,
                h=h
            )
            if h is None:  # auto_tune_config에서 h가 추정됨
                h = params['h']
            if cutoff is None:
                cutoff = params['cutoff']
            if neighbor_batch_size is None:
                neighbor_batch_size = params['neighbor_batch_size']
            leafsize = params['leafsize']
            batch_size_x = params['batch_size_x']
            batch_size_y = params['batch_size_y']
            n_jobs = params['n_jobs']

        if cutoff is None:
            # h가 있으면 4*h, 없으면 임의
            if h is not None:
                cutoff = 4 * h
            else:
                cutoff = 1.0  # fallback

        if h is None:
            h = 1.0  # fallback
        if neighbor_batch_size is None:
            neighbor_batch_size = 100000

        # KD-Tree 생성
        tree = cKDTree(self.particles, leafsize=leafsize)

        # 격자 생성
        dx, dy = self.grid_spacing
        x_centers = np.arange(xmin + dx / 2, xmax, dx, dtype=np.float64)
        y_centers = np.arange(ymin + dy / 2, ymax, dy, dtype=np.float64)
        nx, ny = len(x_centers), len(y_centers)

        density_gpu = cp.zeros((nx, ny), dtype=cp.float64)
        x_gpu = cp.asarray(x_centers)
        y_gpu = cp.asarray(y_centers)

        h2 = h * h
        cutoff2 = cutoff * cutoff

        # 내부 함수(블록 단위)
        def process_block(block_x_range, block_y_range):
            start_x, end_x = block_x_range
            start_y, end_y = block_y_range

            block_x = x_gpu[start_x:end_x]
            block_y = y_gpu[start_y:end_y]

            block_x_cpu = cp.asnumpy(block_x)
            block_y_cpu = cp.asnumpy(block_y)

            local_density_gpu = cp.zeros((end_x - start_x, end_y - start_y), dtype=cp.float64)

            for i_x, xc in enumerate(block_x_cpu):
                for i_y, yc in enumerate(block_y_cpu):
                    neighbors_idx = tree.query_ball_point([xc, yc], r=cutoff)
                    if len(neighbors_idx) == 0:
                        continue

                    accum = cp.float64(0.0)

                    n_neighbors = len(neighbors_idx)
                    for start_n in range(0, n_neighbors, neighbor_batch_size):
                        end_n = min(start_n + neighbor_batch_size, n_neighbors)
                        subset_idx = neighbors_idx[start_n:end_n]

                        part_x_gpu = cp.asarray(self.particles[subset_idx, 0])
                        part_y_gpu = cp.asarray(self.particles[subset_idx, 1])

                        dx_gpu = (xc - part_x_gpu) + ((xmax - xmin) / 2)
                        dx_gpu = cp.mod(dx_gpu, (xmax - xmin)) - ((xmax - xmin) / 2)

                        dy_gpu = (yc - part_y_gpu) + ((ymax - ymin) / 2)
                        dy_gpu = cp.mod(dy_gpu, (ymax - ymin)) - ((ymax - ymin) / 2)

                        r2_gpu = dx_gpu**2 + dy_gpu**2
                        mask = r2_gpu <= cutoff2

                        if cp.any(mask):
                            accum += cp.sum(kernel_func_gpu(r2_gpu[mask], h2))

                    local_density_gpu[i_x, i_y] = accum

            return (start_x, end_x, start_y, end_y, local_density_gpu)

        # 블록 목록
        block_tasks = []
        for sx in range(0, nx, batch_size_x):
            ex = min(sx + batch_size_x, nx)
            for sy in range(0, ny, batch_size_y):
                ey = min(sy + batch_size_y, ny)
                block_tasks.append(((sx, ex), (sy, ey)))

        # 병렬 실행
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_block)(bx, by) for bx, by in block_tasks
        )

        for (sx, ex, sy, ey, local_gpu) in results:
            density_gpu[sx:ex, sy:ey] = local_gpu

        density_map = density_gpu.get()
        return x_centers, y_centers, density_map
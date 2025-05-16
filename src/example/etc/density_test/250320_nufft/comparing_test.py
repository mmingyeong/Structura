import os
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
import finufft
from numba import njit, prange

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------
# 사용자 정의 가우시안 커널 (실공간)
# --------------------------
def gaussian_kernel(r, h):
    """
    가우시안 커널 (실공간):
      K(r) = (1 / (2π h^2)) * exp(-0.5 * (r/h)^2)
    """
    norm = 1.0 / (2.0 * np.pi * h**2)
    return norm * np.exp(-0.5 * (r / h)**2)

# --------------------------
# 가우시안 커널의 푸리에 변환 (물리적 파동수 기준)
# --------------------------
def gaussian_kernel_ft(k2, h):
    """
    가우시안 커널의 푸리에 변환:
      K̂(k) = (1 / (2π h^2)) * exp(-0.5 * h^2 * k^2)
    여기서 k^2 = kx_phys^2 + ky_phys^2,  k_phys = (2π/L)*k.
    """
    norm = 1.0 / (2.0 * np.pi * h**2)
    return norm * np.exp(-0.5 * h**2 * k2)

# --------------------------
# 1. 직접 합산 방식 (주기적 경계, cutoff 제거)
# --------------------------
@njit(parallel=True, fastmath=True)
def compute_density_direct_periodic(particles, x_centers, y_centers, Lx, Ly, h):
    """
    직접 합산(Numba) 방식으로 2D 밀도장을 계산 (주기적 경계, cutoff 제거).
    각 격자점에서, 최소 이미지 방식을 적용해 파티클과의 거리 r를 구한 후,
    jitted된 gaussian_kernel_jit를 사용하여 모든 파티클 기여를 합산합니다.
    """
    nx = x_centers.shape[0]
    ny = y_centers.shape[0]
    density = np.zeros((nx, ny), dtype=np.float64)
    
    n_particles = particles.shape[0]
    
    for i in prange(nx):
        xc = x_centers[i]
        for j in prange(ny):
            yc = y_centers[j]
            val = 0.0
            for k in range(n_particles):
                dx = xc - particles[k, 0]
                dy = yc - particles[k, 1]
                # 최소 이미지 (주기적 경계)
                dx = (dx + Lx/2) % Lx - Lx/2
                dy = (dy + Ly/2) % Ly - Ly/2
                r = math.sqrt(dx*dx + dy*dy)
                val += (1.0 / (2.0 * math.pi * h * h)) * math.exp(-0.5 * (r / h)**2)
            density[i, j] = val
    return density

class DensityCalculator2D:
    """
    직접 합산(Numba) 방식으로 2D 밀도장을 계산하는 클래스 (주기적 경계, cutoff 제거).
    """
    def __init__(self, particles, grid_bounds, grid_spacing):
        self.particles = particles
        self.grid_bounds = grid_bounds  # 예: {"x": (-6,6), "y": (-6,6)}
        self.grid_spacing = grid_spacing  # 예: (0.05, 0.05)

    def calculate_density_map(self, h):
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing
        
        # 격자점 정의 (spacing 0.05)
        x_centers = np.arange(xmin + dx/2, xmax, dx, dtype=np.float64)
        y_centers = np.arange(ymin + dy/2, ymax, dy, dtype=np.float64)
        x_centers = np.ascontiguousarray(x_centers)
        y_centers = np.ascontiguousarray(y_centers)
        
        Lx = xmax - xmin
        Ly = ymax - ymin
        
        density_map = compute_density_direct_periodic(self.particles, x_centers, y_centers, Lx, Ly, h)
        return density_map

# --------------------------
# 2. NUFFT 방식 (주기적 경계, isign=-1, 상수 계수·스케일 보정, cutoff 제거, upsampfac 조정)
# --------------------------
class NUFFT_KDE:
    """
    2D NUFFT를 이용하여 가우시안 KDE를 계산하는 클래스 (주기적 경계).
    
    알고리즘:
      1) 파티클 좌표를 도메인 중앙 기준으로 [-π, π]로 선형 스케일링.
      2) Plan 기반 NUFFT를 사용하여, isign=-1 및 upsampfac=3.0으로 푸리에 계수를 계산.
      3) FFT 주파수를 물리적 파동수로 재스케일링한 후,
         사용자 정의 가우시안 커널의 푸리에 변환을 곱함:
         K̂(k) = (1/(2πh²)) * exp(-0.5 * h² * k_phys²), with k_phys = (2π/L)*k.
      4) Inverse FFT 후 (nx*ny) 보정하여 실공간 밀도장을 복원.
    """
    def __init__(self, particles, grid_bounds, grid_spacing, h):
        self.particles = particles
        self.grid_bounds = grid_bounds  # 예: {"x": (-6,6), "y": (-6,6)}
        self.grid_spacing = grid_spacing  # 예: (0.05, 0.05)
        self.h = h

        self.xmin, self.xmax = grid_bounds["x"]
        self.ymin, self.ymax = grid_bounds["y"]
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

        nx = int((self.xmax - self.xmin) / self.grid_spacing[0])
        ny = int((self.ymax - self.ymin) / self.grid_spacing[1])
        self.grid_size = (nx, ny)

        logging.info(f"Initialized NUFFT KDE with {particles.shape[0]} particles.")
        logging.info(f"Domain: x=({self.xmin},{self.xmax}), y=({self.ymin},{self.ymax}), grid_size={self.grid_size}")

    def compute_density_map(self):
        nx, ny = self.grid_size

        # (A) 좌표를 [-π, π]로 선형 스케일링 (중앙 정렬)
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        x_scaled = (x - (self.xmin + self.xmax) / 2) * (2 * np.pi / self.Lx)
        y_scaled = (y - (self.ymin + self.ymax) / 2) * (2 * np.pi / self.Ly)

        # (B) Plan 기반 NUFFT 사용: upsampfac=3.0 적용
        # n_modes를 튜플로 전달
        plan = finufft.Plan(1, 2, np.array([nx, ny], dtype=np.int64), isign=-1, eps=1e-6, upsampfac=3.0)
        plan.setpts(x_scaled, y_scaled)
        c = np.ones_like(x_scaled, dtype=np.complex128)
        F_k = plan.execute(c)

        # (C) FFT 주파수를 물리적 파동수로 재스케일링: k_phys = (2π / L)*k
        kx = np.fft.fftfreq(nx, d=1.0) * nx
        ky = np.fft.fftfreq(ny, d=1.0) * ny
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        kx_phys = (2 * np.pi / self.Lx) * kx
        ky_phys = (2 * np.pi / self.Ly) * ky
        k2_phys = kx_phys**2 + ky_phys**2

        # (D) 사용자 정의 푸리에 커널 적용
        kernel_ft = gaussian_kernel_ft(k2_phys, self.h)
        F_k_2d = F_k.reshape((nx, ny))
        G_k_2d = F_k_2d * kernel_ft

        # (E) Inverse FFT + (nx*ny) 보정
        density_complex = np.fft.ifft2(np.fft.ifftshift(G_k_2d))
        density_map = np.real(density_complex) * (nx * ny)
        
        return density_map

# --------------------------
# 3. 메인 테스트 코드
# --------------------------
def main():
    logging.info("Periodic Domain: Compare Direct Summation vs. NUFFT (Custom Gaussian Kernel, upsampfac=3.0)...")
    
    # A) 파티클 데이터 생성 (두 개의 가우시안 클러스터)
    data_filename = "particles_periodic.npy"
    if os.path.exists(data_filename):
        logging.info("Loading particle data from file...")
        particles = np.load(data_filename)
    else:
        logging.info("Generating new particle data (two Gaussian clusters)...")
        N = 5000
        N1 = N // 2
        N2 = N - N1
        cluster1 = np.random.randn(N1, 2) * 1.0 + np.array([1, 1])
        cluster2 = np.random.randn(N2, 2) * 0.5 + np.array([-2, -2])
        particles = np.vstack([cluster1, cluster2])
        np.save(data_filename, particles)
        logging.info(f"Generated and saved particle data to {data_filename}")

    # B) 파라미터 설정 (도메인 [-6,6], 주기적), 격자 spacing: 0.05 → 240x240 격자
    grid_bounds = {"x": (-6, 6), "y": (-6, 6)}
    grid_spacing = (0.05, 0.05)
    h = 0.5

    # C) 직접 합산 방식
    calc2d = DensityCalculator2D(particles, grid_bounds, grid_spacing)
    density_direct = calc2d.calculate_density_map(h)

    # D) NUFFT 방식 (Plan 기반, upsampfac=3.0 적용)
    kde_nufft = NUFFT_KDE(particles, grid_bounds, grid_spacing, h)
    density_nufft = kde_nufft.compute_density_map()

    # E) 시각화 (세 개 서브플롯)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # (1) 원본 입자 분포
    axs[0].scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.5)
    axs[0].set_title("Original Particle Distribution (Periodic)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_xlim(grid_bounds["x"])
    axs[0].set_ylim(grid_bounds["y"])

    # (2) NUFFT 결과
    im1 = axs[1].imshow(
        density_nufft,
        extent=[grid_bounds["x"][0], grid_bounds["x"][1],
                grid_bounds["y"][0], grid_bounds["y"][1]],
        origin="lower", cmap="inferno"
    )
    axs[1].set_title("NUFFT Density Field (upsampfac=3.0)")
    fig.colorbar(im1, ax=axs[1], label="Density")

    # (3) 직접 합산 결과
    im2 = axs[2].imshow(
        density_direct.T,  # 전치
        extent=[grid_bounds["x"][0], grid_bounds["x"][1],
                grid_bounds["y"][0], grid_bounds["y"][1]],
        origin="lower", cmap="inferno"
    )
    axs[2].set_title("Direct Summation (Custom Kernel)")
    fig.colorbar(im2, ax=axs[2], label="Density")

    plt.tight_layout()
    plt.savefig("compare_nufft_vs_direct_custom_kernel_upsampfac.png", dpi=300)
    plt.show()

    logging.info("Done. Figure saved as compare_nufft_vs_direct_custom_kernel_upsampfac.png")


if __name__ == "__main__":
    main()

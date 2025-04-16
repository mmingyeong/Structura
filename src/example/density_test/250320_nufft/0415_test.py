import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
import finufft

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def log_array_info(name, arr):
    logging.info(f"{name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

# -------------------------
# Configuration
# -------------------------
data_filename = "particles_periodic.npy"
h = 1.0
kernel_type = "triangle"
nx, ny = 128, 128

# Record total execution start time
total_start_time = time.time()

# -------------------------
# 1. Particle Data Loading
# -------------------------
original_filename = "particles_periodic.npy"

if not os.path.exists(original_filename):
    logging.info("Generating new particle data...")
    N = 5000
    N1 = N // 2
    N2 = N - N1
    cluster1 = np.random.randn(N1, 2) * 1.0 + np.array([1, 1])
    cluster2 = np.random.randn(N2, 2) * 0.5 + np.array([-2, -2])
    particles = np.vstack([cluster1, cluster2])
    np.save(original_filename, particles)
else:
    particles = np.load(original_filename)

log_array_info("Particles", particles)

# -------------------------
# 2. Grid Setup
# -------------------------
xmin, xmax = particles[:, 0].min(), particles[:, 0].max()
ymin, ymax = particles[:, 1].min(), particles[:, 1].max()
Lx, Ly = xmax - xmin, ymax - ymin
grid_bounds = {"x": (xmin, xmax), "y": (ymin, ymax)}
grid_spacing = (Lx / nx, Ly / ny)
extent = [xmin, xmax, ymin, ymax]

# -------------------------
# 3. Direct Summation
# -------------------------
def triangle_kernel(dx, dy, h):
    tx = (1 - abs(dx)/h)/h if abs(dx) < h else 0.0
    ty = (1 - abs(dy)/h)/h if abs(dy) < h else 0.0
    return tx * ty

def compute_density_direct(particles, grid_bounds, grid_spacing, h):
    x_centers = xmin + grid_spacing[0] * (0.5 + np.arange(nx))
    y_centers = ymin + grid_spacing[1] * (0.5 + np.arange(ny))
    density = np.zeros((len(x_centers), len(y_centers)))
    for i, xc in enumerate(x_centers):
        for j, yc in enumerate(y_centers):
            val = 0.0
            for k in range(particles.shape[0]):
                dx = ((xc - particles[k, 0] + Lx/2) % Lx) - Lx/2
                dy = ((yc - particles[k, 1] + Ly/2) % Ly) - Ly/2
                val += triangle_kernel(dx, dy, h)
            density[i, j] = val
    density /= (grid_spacing[0] * grid_spacing[1])
    return density

direct_start_time = time.time()
density_direct = compute_density_direct(particles, grid_bounds, grid_spacing, h)
direct_end_time = time.time()
logging.info("Direct density computation time: {:.4f} seconds".format(direct_end_time - direct_start_time))

# 원하는 평균 밀도
mean_n = len(particles) / (Lx * Ly)

# 현재 Direct 평균 밀도
mean_direct = density_direct.mean()

# 보정 계수 계산
correction_factor = mean_n / mean_direct
density_direct *= correction_factor

# -------------------------
# 4. NUFFT Density Estimation
# -------------------------
def triangle_kernel_ft(kx, ky, h):
    r = np.sqrt(kx**2 + ky**2)
    kernel = np.ones_like(r)
    nonzero = (r != 0)
    kernel[nonzero] = (np.sin(r[nonzero]*h/2)/(r[nonzero]*h/2))**2
    return kernel

def compute_density_nufft(particles, grid_bounds, grid_spacing, h):
    # 그리드 해상도 및 도메인 크기 계산
    nx = int((grid_bounds["x"][1] - grid_bounds["x"][0]) / grid_spacing[0])
    ny = int((grid_bounds["y"][1] - grid_bounds["y"][0]) / grid_spacing[1])
    Lx = grid_bounds["x"][1] - grid_bounds["x"][0]
    Ly = grid_bounds["y"][1] - grid_bounds["y"][0]

    # 입자 좌표 중심화 및 scaling (주기적 도메인 [-π, π)로)
    x = particles[:, 0]
    y = particles[:, 1]
    x_center = (grid_bounds["x"][0] + grid_bounds["x"][1]) / 2
    y_center = (grid_bounds["y"][0] + grid_bounds["y"][1]) / 2
    x_scaled = (x - x_center) * (2 * np.pi / Lx)
    y_scaled = (y - y_center) * (2 * np.pi / Ly)

    # NUFFT 계획 생성 (spread_kerevalmeth=0를 명시하여 upsampfac=3.0 문제 해결)
    plan = finufft.Plan(1, (nx, ny), isign=-1, eps=1e-6, upsampfac=3.0, spread_kerevalmeth=0)
    plan.setpts(x_scaled, y_scaled)

    # c: particle weights (normalized by number of particles)
    c = np.ones_like(x_scaled, dtype=np.complex128) / len(particles)
    F_k = plan.execute(c).reshape((nx, ny))
    F_k = np.fft.ifftshift(F_k)  # ifft2 대응을 위한 shift

    # 파동수 벡터 생성
    kx = np.fft.fftfreq(nx, d=1.0) * nx
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    kx_phys = (2 * np.pi / Lx) * kx
    ky_phys = (2 * np.pi / Ly) * ky

    # 삼각형 커널의 Fourier 변환 적용
    kernel_ft = triangle_kernel_ft(kx_phys, ky_phys, h)
    G_k = F_k * kernel_ft

    # 역 FFT 및 정규화: ifft2는 내부적으로 1/(nx*ny) 곱해짐 → 이를 상쇄하기 위해 (nx*ny)를 곱함
    density_complex = np.fft.ifft2(G_k)
    density_map = np.real(density_complex)
    density_map *= (nx * ny)
    
    # 면적당 밀도 (number density)로 환산: 전체 입자 수와 면적에 따른 스케일링
    density_map *= len(particles) / (Lx * Ly)

    return np.clip(density_map, 0, None)

nufft_start_time = time.time()
density_nufft = compute_density_nufft(particles, grid_bounds, grid_spacing, h)
density_nufft_shifted = np.fft.fftshift(density_nufft)
nufft_end_time = time.time()
logging.info("NUFFT density computation time: {:.4f} seconds".format(nufft_end_time - nufft_start_time))

# -------------------------
# 5. Comparison Metrics
# -------------------------
area = Lx * Ly
mean_n = len(particles) / area
mean_direct = density_direct.mean()
mean_nufft = density_nufft.mean()
vmax = max(density_direct.max(), density_nufft.max())

log_array_info("Direct Density", density_direct)
log_array_info("NUFFT Density", density_nufft)

# -------------------------
# 6. Visualization
# -------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.5, color='black')
axs[0, 0].set_title("Original Particle Distribution")
axs[0, 0].set_xlim(grid_bounds["x"])
axs[0, 0].set_ylim(grid_bounds["y"])
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")

# NUFFT Density Map
im1 = axs[0, 1].imshow(density_nufft_shifted.T, extent=extent, origin="lower", cmap="inferno")
axs[0, 1].set_title("NUFFT Density Map (Triangle)")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

# Direct Density Map
im2 = axs[1, 0].imshow(density_direct.T, extent=extent, origin="lower", cmap="inferno")
axs[1, 0].set_title("Direct Density Map (Triangle)")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)


dmap1 = np.log10(density_direct.flatten() + 1)
dmap2 = np.log10(density_nufft.flatten() + 1)
bins = np.linspace(1, 6.5, 150)
axs[1, 1].hist(dmap1, bins=bins, density=True, alpha=0.8,
               label=f"Direct (μ={mean_direct:.1f})", color="blue", histtype='step', linewidth=2)
axs[1, 1].hist(dmap2, bins=bins, density=True, alpha=0.8,
               label=f"NUFFT (μ={mean_nufft:.1f})", color="red", histtype='step', linewidth=2)
axs[1, 1].set_xlabel("log10(Density + 1)")
axs[1, 1].set_ylabel("Probability Density")
axs[1, 1].set_title("PDF Comparison of Density Maps")
axs[1, 1].legend()
axs[1, 1].grid(True)

# ➕ 텍스트 추가
textstr = (
    f"Mean Particle Density: {mean_n:.1f}\n"
    f"Mean Direct Density: {mean_direct:.1f}\n"
    f"Mean NUFFT Density: {mean_nufft:.1f}"
)
axs[1, 1].text(
    0.98, 0.70,  # <-- y좌표를 낮춰서 범례와 겹치지 않도록
    textstr,
    transform=axs[1, 1].transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9)
)


plt.tight_layout()
plt.savefig("new_combined_density_comparison.png", dpi=300)
plt.close()
logging.info("✅ Density maps and comparison plot saved.")

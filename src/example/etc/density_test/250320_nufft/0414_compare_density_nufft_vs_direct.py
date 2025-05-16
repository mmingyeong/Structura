import numpy as np
import matplotlib.pyplot as plt
import os
import time
from matplotlib.backends.backend_pdf import PdfPages  # Note: unused
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
# 1. Particle Data Loading or Generation
# -------------------------
if os.path.exists(data_filename):
    particles = np.load(data_filename)
    logging.info(f"Loaded particle data from {data_filename}")
else:
    logging.info("Generating new particle data...")
    N = 5000
    N1 = N // 2
    N2 = N - N1
    cluster1 = np.random.randn(N1, 2) * 1.0 + np.array([1, 1])
    cluster2 = np.random.randn(N2, 2) * 0.5 + np.array([-2, -2])
    particles = np.vstack([cluster1, cluster2])
    np.save(data_filename, particles)
    logging.info(f"Generated and saved particles to {data_filename}")
log_array_info("Particles", particles)

# -------------------------
# 2. Grid Setup and Boundaries
# -------------------------
xmin, xmax = particles[:, 0].min(), particles[:, 0].max()
ymin, ymax = particles[:, 1].min(), particles[:, 1].max()

# 도메인 크기 및 grid spacing 계산 (주기적 도메인 가정)
Lx = xmax - xmin
Ly = ymax - ymin
grid_bounds = {"x": (xmin, xmax), "y": (ymin, ymax)}
grid_spacing = (Lx / nx, Ly / ny)

# 각 격자 셀의 중심 (시각화를 위해)
x_centers = xmin + grid_spacing[0] * (0.5 + np.arange(nx))
y_centers = ymin + grid_spacing[1] * (0.5 + np.arange(ny))
extent = [xmin, xmax, ymin, ymax]

# -------------------------
# 3. Direct Summation (Triangle Kernel)
# -------------------------
# 여기서 삼각형 커널을 적분값 1이 되도록 정규화:
def triangle_kernel(dx, dy, h):
    # 1D: (1 - |x|/h)/h  (|x|<h), 0 else  -> 적분값 1.
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
                # 주기적 wrap를 적용하여 최소 거리 계산
                dx = ((xc - particles[k, 0] + Lx/2) % Lx) - Lx/2
                dy = ((yc - particles[k, 1] + Ly/2) % Ly) - Ly/2
                val += triangle_kernel(dx, dy, h)
            density[i, j] = val
    # 셀 면적: grid_spacing[0] * grid_spacing[1] = (Lx*Ly)/(nx*ny)
    # 면적당 밀도 = 합산값 / (셀 면적)
    density /= (grid_spacing[0] * grid_spacing[1])
    return density

# Measure Direct Density Computation Time
direct_start_time = time.time()
density_direct = compute_density_direct(particles, grid_bounds, grid_spacing, h)
direct_end_time = time.time()
direct_time = direct_end_time - direct_start_time
logging.info("Direct density computation time: {:.4f} seconds".format(direct_time))

# -------------------------
# 4. NUFFT Density Computation
# -------------------------
# Fourier 변환 측면에서, 정규화된 삼각형 커널의 Fourier 변환은:
# (sin(k*h/2)/(k*h/2))^2 를 사용합니다.
def triangle_kernel_ft(kx, ky, h):
    r = np.sqrt(kx**2 + ky**2)
    kernel = np.ones_like(r)
    nonzero = (r != 0)
    # h/2 사용: 정규화된 공간 커널 Fourier 변환
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

    # NUFFT 계획 생성 (비정규 -> 균일 그리드)
    plan = finufft.Plan(1, (nx, ny), isign=-1, eps=1e-6,
                        upsampfac=3.0, spread_kerevalmeth=0)
    plan.setpts(x_scaled, y_scaled)
    c = np.ones_like(x_scaled, dtype=np.complex128)
    F_k = plan.execute(c).reshape((nx, ny))
    
    # finufft가 반환한 Fourier 계수는 centered 형태이므로,
    # ifft2가 올바르게 처리하도록 ifftshift 적용
    F_k = np.fft.ifftshift(F_k)
    
    # 물리적 주파수 계산 (커널 적용을 위해)
    kx = np.fft.fftfreq(nx, d=1.0) * nx
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    kx_phys = (2 * np.pi / Lx) * kx
    ky_phys = (2 * np.pi / Ly) * ky

    # 정규화된 삼각 커널 Fourier 변환 적용
    kernel_ft = triangle_kernel_ft(kx_phys, ky_phys, h)
    G_k = F_k * kernel_ft
    
    # Inverse FFT 수행: ifft2는 1/(nx*ny) 정규화 포함 → (nx*ny)를 곱하여 원래 합 복원
    density_complex = np.fft.ifft2(G_k)
    density_map = np.real(density_complex) * (nx * ny)
    
    # 면적당 밀도: 셀 면적 = (Lx*Ly)/(nx*ny)
    density_map /= (Lx * Ly / (nx * ny))
    return np.clip(density_map, 0, None)

# Measure NUFFT Density Computation Time
nufft_start_time = time.time()
density_nufft = compute_density_nufft(particles, grid_bounds, grid_spacing, h)
nufft_end_time = time.time()
nufft_time = nufft_end_time - nufft_start_time
logging.info("NUFFT density computation time: {:.4f} seconds".format(nufft_time))

log_array_info("Direct Density", density_direct)
log_array_info("NUFFT Density", density_nufft)
np.save("density_direct.npy", density_direct)
np.save("density_nufft.npy", density_nufft)

# -------------------------
# 5. Visualization and Saving Results
# -------------------------
vis_start_time = time.time()
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# (1) Original Particle Distribution
axs[0, 0].scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.5, color='black')
axs[0, 0].set_title("Original Particle Distribution")
axs[0, 0].set_xlim(grid_bounds["x"])
axs[0, 0].set_ylim(grid_bounds["y"])
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")

# (2) NUFFT Density Map
im1 = axs[0, 1].imshow(density_nufft.T, extent=extent, origin="lower", cmap="inferno")
axs[0, 1].set_title("NUFFT Density Map (Triangle)")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

# (3) Direct Density Map
im2 = axs[1, 0].imshow(density_direct.T, extent=extent, origin="lower", cmap="inferno")
axs[1, 0].set_title("Direct Density Map (Triangle)")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

# (4) PDF Comparison Histogram
dmap1 = np.log10(density_direct.flatten() + 1)
dmap2 = np.log10(density_nufft.flatten() + 1)
mean1 = density_direct.mean()
mean2 = density_nufft.mean()
bins = np.linspace(1, 6.5, 150)
axs[1, 1].hist(dmap1, bins=bins, density=True, alpha=0.8, label=f"Direct (μ={mean1:.1f})",
               color="blue", histtype='step', linewidth=2)
axs[1, 1].hist(dmap2, bins=bins, density=True, alpha=0.8, label=f"NUFFT (μ={mean2:.1f})",
               color="red", histtype='step', linewidth=2)
axs[1, 1].set_xlabel("log10(Density + 1)")
axs[1, 1].set_ylabel("Probability Density")
axs[1, 1].set_title("PDF Comparison of Density Maps")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
merged_filename = "combined_density_comparison.png"
plt.savefig(merged_filename, dpi=300)
plt.close()
vis_end_time = time.time()
vis_time = vis_end_time - vis_start_time
logging.info("Visualization computation time: {:.4f} seconds".format(vis_time))

# -------------------------
# Total Execution Time
# -------------------------
total_end_time = time.time()
total_time = total_end_time - total_start_time
logging.info("Total execution time: {:.4f} seconds".format(total_time))
logging.info(f"✅ 모든 시각화 완료. 통합 그림 저장됨: {merged_filename}")

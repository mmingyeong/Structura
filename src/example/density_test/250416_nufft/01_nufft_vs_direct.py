import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
import finufft
import h5py
from dask import delayed, compute
import matplotlib.gridspec as gridspec

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
kernel_type = "triangle"  # "triangle" 또는 "uniform"
nx, ny = 128, 128
use_dask_parallel = True  # Dask 병렬화 사용 여부
save_hdf5 = False         # HDF5 저장 여부

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
logging.info(f"grid_bounds: {grid_bounds}")
grid_spacing = (Lx / nx, Ly / ny)
logging.info(f"grid_spacing: {grid_spacing}")
extent = [xmin, xmax, ymin, ymax]

# -------------------------
# 3. Direct Summation (with Dask Parallelization & Memory Optimization)
# -------------------------
def triangle_kernel(dx, dy, h):
    # 삼각형 커널: |dx| < h, |dy| < h 범위 내에서 선형 감쇠
    tx = (1 - abs(dx)/h)/h if abs(dx) < h else 0.0
    ty = (1 - abs(dy)/h)/h if abs(dy) < h else 0.0
    return tx * ty

def uniform_kernel(dx, dy, h):
    # uniform (상수) 커널: |dx|, |dy|가 h/2 이하이면 1/h^2, 아니면 0
    return 1.0 / (h**2) if (abs(dx) <= h/2 and abs(dy) <= h/2) else 0.0

def compute_density_direct(particles, grid_bounds, grid_spacing, h, use_dask=True, kernel_type="triangle"):
    x_centers = np.linspace(grid_bounds["x"][0] + grid_spacing[0]/2,
                            grid_bounds["x"][1] - grid_spacing[0]/2,
                            nx, dtype=np.float32)
    y_centers = np.linspace(grid_bounds["y"][0] + grid_spacing[1]/2,
                            grid_bounds["y"][1] - grid_spacing[1]/2,
                            ny, dtype=np.float32)
    density = np.zeros((nx, ny), dtype=np.float32)

    Lx = grid_bounds["x"][1] - grid_bounds["x"][0]
    Ly = grid_bounds["y"][1] - grid_bounds["y"][0]

    if kernel_type == "triangle":
        kernel_func = triangle_kernel
    elif kernel_type == "uniform":
        kernel_func = uniform_kernel
    else:
        raise ValueError("Unknown kernel type specified for direct method.")

    if use_dask:
        @delayed
        def compute_cell(xc, yc):
            val = 0.0
            for k in range(particles.shape[0]):
                dx = ((xc - particles[k, 0] + Lx/2) % Lx) - Lx/2
                dy = ((yc - particles[k, 1] + Ly/2) % Ly) - Ly/2
                val += kernel_func(dx, dy, h)
            return np.float32(val)

        delayed_rows = []
        for i, xc in enumerate(x_centers):
            row_tasks = [compute_cell(xc, yc) for yc in y_centers]
            delayed_rows.append(delayed(np.array)(row_tasks, dtype=np.float32))

        rows = compute(*delayed_rows)
        density = np.vstack(rows)
    else:
        for i, xc in enumerate(x_centers):
            for j, yc in enumerate(y_centers):
                val = 0.0
                for k in range(particles.shape[0]):
                    dx = ((xc - particles[k, 0] + Lx/2) % Lx) - Lx/2
                    dy = ((yc - particles[k, 1] + Ly/2) % Ly) - Ly/2
                    val += kernel_func(dx, dy, h)
                density[i, j] = val

    density /= (grid_spacing[0] * grid_spacing[1])
    return density

direct_start_time = time.time()
density_direct = compute_density_direct(particles, grid_bounds, grid_spacing, h,
                                        use_dask=use_dask_parallel, kernel_type=kernel_type)
direct_end_time = time.time()
logging.info("Direct density computation time: {:.4f} seconds".format(direct_end_time - direct_start_time))

# 평균 밀도와 보정 계수 적용
mean_n = len(particles) / (Lx * Ly)
mean_direct = density_direct.mean()
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

def uniform_kernel_ft(kx, ky, h):
    # uniform 커널의 Fourier 변환(2D sinc의 곱)
    kx_term = np.where(np.abs(kx) < 1e-10, 1.0, np.sin(kx * h/2)/(kx * h/2))
    ky_term = np.where(np.abs(ky) < 1e-10, 1.0, np.sin(ky * h/2)/(ky * h/2))
    return kx_term * ky_term

def compute_density_nufft(particles, grid_bounds, grid_spacing, h, kernel_type="triangle"):
    nx = int((grid_bounds["x"][1] - grid_bounds["x"][0]) / grid_spacing[0])
    ny = int((grid_bounds["y"][1] - grid_bounds["y"][0]) / grid_spacing[1])
    Lx = grid_bounds["x"][1] - grid_bounds["x"][0]
    Ly = grid_bounds["y"][1] - grid_bounds["y"][0]

    x = particles[:, 0]
    y = particles[:, 1]
    x_center = (grid_bounds["x"][0] + grid_bounds["x"][1]) / 2
    y_center = (grid_bounds["y"][0] + grid_bounds["y"][1]) / 2
    x_scaled = (x - x_center) * (2 * np.pi / Lx)
    y_scaled = (y - y_center) * (2 * np.pi / Ly)

    plan = finufft.Plan(1, (nx, ny), isign=-1, eps=1e-6, upsampfac=3.0, spread_kerevalmeth=0)
    plan.setpts(x_scaled, y_scaled)

    c = np.ones_like(x_scaled, dtype=np.complex128) / len(particles)
    F_k = plan.execute(c).reshape((nx, ny))
    F_k = np.fft.ifftshift(F_k)

    kx = np.fft.fftfreq(nx, d=1.0) * nx
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    kx_phys = (2 * np.pi / Lx) * kx
    ky_phys = (2 * np.pi / Ly) * ky

    if kernel_type == "triangle":
        kernel_ft = triangle_kernel_ft(kx_phys, ky_phys, h)
    elif kernel_type == "uniform":
        kernel_ft = uniform_kernel_ft(kx_phys, ky_phys, h)
    else:
        raise ValueError("Unknown kernel type specified for NUFFT method.")

    G_k = F_k * kernel_ft
    density_complex = np.fft.ifft2(G_k)
    density_map = np.real(density_complex)
    density_map *= (nx * ny)
    density_map *= len(particles) / (Lx * Ly)
    return np.clip(density_map, 0, None)

nufft_start_time = time.time()
density_nufft = compute_density_nufft(particles, grid_bounds, grid_spacing, h, kernel_type=kernel_type)
density_nufft_shifted = np.fft.fftshift(density_nufft)
nufft_end_time = time.time()
logging.info("NUFFT density computation time: {:.4f} seconds".format(nufft_end_time - nufft_start_time))

# -------------------------
# 5. Comparison Metrics
# -------------------------
area = Lx * Ly
mean_nufft = density_nufft.mean()
vmax = max(density_direct.max(), density_nufft.max())

log_array_info("Direct Density", density_direct)
log_array_info("NUFFT Density", density_nufft)

# -------------------------
# 6. Visualization of Density Maps and PDF Comparison (4 Subplots)
# -------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# (a) Original Particle Distribution
axs[0, 0].scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.5, color='black')
axs[0, 0].set_title("Original Particle Distribution")
axs[0, 0].set_xlim(grid_bounds["x"])
axs[0, 0].set_ylim(grid_bounds["y"])
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")

# (b) NUFFT Density Map
im1 = axs[0, 1].imshow(density_nufft_shifted.T, extent=extent, origin="lower", cmap="inferno")
axs[0, 1].set_title(f"NUFFT Density Map ({kernel_type.capitalize()})")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

# (c) Direct Density Map
im2 = axs[1, 0].imshow(density_direct.T, extent=extent, origin="lower", cmap="inferno")
axs[1, 0].set_title(f"Direct Density Map ({kernel_type.capitalize()})")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

# (d) PDF Comparison
dmap1 = np.log10(density_direct.flatten() + 1)
dmap2 = np.log10(density_nufft.flatten() + 1)
bins = np.linspace(1, 6.5, 150)
axs[1, 1].hist(dmap1, bins=bins, density=True, alpha=0.8,
               label=f"Direct (μ={density_direct.mean():.1f})", color="blue", histtype='step', linewidth=2)
axs[1, 1].hist(dmap2, bins=bins, density=True, alpha=0.8,
               label=f"NUFFT (μ={density_nufft.mean():.1f})", color="red", histtype='step', linewidth=2)
axs[1, 1].set_xlabel("log10(Density + 1)")
axs[1, 1].set_ylabel("Probability Density")
axs[1, 1].set_title("PDF Comparison of Density Maps")
axs[1, 1].legend()
axs[1, 1].grid(True)

textstr = (
    f"Mean Particle Density: {mean_n:.1f}\n"
    f"Mean Direct Density: {density_direct.mean():.1f}\n"
    f"Mean NUFFT Density: {density_nufft.mean():.1f}"
)
axs[1, 1].text(
    0.98, 0.70,
    textstr,
    transform=axs[1, 1].transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9)
)

plt.tight_layout()
plt.savefig(f"{kernel_type}_combined_density_comparison.png", dpi=300)
plt.close()
logging.info("Density maps and comparison plot saved.")

# -------------------------
# 7. HDF5 Saving Option
# -------------------------
if save_hdf5:
    hdf5_filename = "density_data.h5"
    with h5py.File(hdf5_filename, "w") as hf:
        hf.create_dataset("density_direct", data=density_direct, compression="gzip")
        hf.create_dataset("density_nufft", data=density_nufft, compression="gzip")
    logging.info(f"HDF5 file saved: {hdf5_filename}")


import numpy as np
import matplotlib.pyplot as plt
import logging

# ... (위쪽에서 density 계산까지의 코드는 생략) ...

# -------------------------
# 8. Kernel Function Visualization (2×2 layout)
# -------------------------
support_min, support_max = -h, h
n_vis = 200  # 2D 시각화 해상도
x_vals = np.linspace(support_min, support_max, n_vis)
y_vals = np.linspace(support_min, support_max, n_vis)
X, Y = np.meshgrid(x_vals, y_vals)

if kernel_type == "triangle":
    # 2D 실공간 커널
    Z_real = (np.where(np.abs(X) < h, (1 - np.abs(X)/h)/h, 0) *
              np.where(np.abs(Y) < h, (1 - np.abs(Y)/h)/h, 0))

    # 2D 푸리에 커널
    kmin, kmax = -5, 5
    n_k = 200
    kx_vals = np.linspace(kmin, kmax, n_k)
    ky_vals = np.linspace(kmin, kmax, n_k)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    R = np.sqrt(KX**2 + KY**2)
    Z_fourier = np.ones_like(R)
    mask = (R != 0)
    Z_fourier[mask] = (np.sin(R[mask]*h/2)/(R[mask]*h/2))**2

    eq_text_real_2d = r"$K(x,y)=\frac{(1-|x|/h)(1-|y|/h)}{h^2}\quad (|x|,|y|<h)$"
    eq_text_fourier_2d = r"$\widehat{K}(k_x,k_y)=\left[\frac{\sin(0.5\,h\,\sqrt{k_x^2+k_y^2})}{0.5\,h\,\sqrt{k_x^2+k_y^2}}\right]^2$"

    # 1D Cross-section (Real)
    x_1d = np.linspace(-h, h, 200)
    real_1d = np.where(np.abs(x_1d) < h, (1 - np.abs(x_1d)/h)/h, 0) * (1/h)
    eq_text_real_1d = r"$K(x,0)=\frac{1 - |x|/h}{h^2}\quad (|x|<h)$"

    # 1D Cross-section (Fourier)
    kx_1d = np.linspace(kmin, kmax, 200)
    fourier_1d = np.where(np.abs(kx_1d) < 1e-10,
                          1.0,
                          (np.sin(0.5*h*np.abs(kx_1d))/(0.5*h*np.abs(kx_1d)))**2)
    eq_text_fourier_1d = r"$\widehat{K}(k_x,0)=\left[\frac{\sin(0.5\,h\,|k_x|)}{0.5\,h\,|k_x|}\right]^2$"

elif kernel_type == "uniform":
    # 2D 실공간 커널
    Z_real = np.where((np.abs(X) <= h/2) & (np.abs(Y) <= h/2), 1.0/(h**2), 0)

    # 2D 푸리에 커널
    kmin, kmax = -5, 5
    n_k = 200
    kx_vals = np.linspace(kmin, kmax, n_k)
    ky_vals = np.linspace(kmin, kmax, n_k)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    def sinc(z):
        return np.where(np.abs(z) < 1e-10, 1.0, np.sin(z)/z)
    Z_fourier = sinc(KX*(h/2)) * sinc(KY*(h/2))

    eq_text_real_2d = r"$K(x,y)=\frac{1}{h^2}\quad (|x|,|y|\leq h/2)$"
    eq_text_fourier_2d = r"$\widehat{K}(k_x,k_y)=\mathrm{sinc}(0.5\,h\,k_x)\,\mathrm{sinc}(0.5\,h\,k_y)$"

    # 1D Cross-section (Real)
    x_1d = np.linspace(-h, h, 200)
    real_1d = np.where(np.abs(x_1d) <= h/2, 1.0/(h**2), 0)
    eq_text_real_1d = r"$K(x,0)=\frac{1}{h^2}\quad (|x|\leq h/2)$"

    # 1D Cross-section (Fourier)
    kmin, kmax = -5, 5
    kx_1d = np.linspace(kmin, kmax, 200)
    fourier_1d = sinc(kx_1d*(h/2))
    eq_text_fourier_1d = r"$\widehat{K}(k_x,0)=\mathrm{sinc}(0.5\,h\,k_x)$"

else:
    raise ValueError("Unknown kernel type specified for visualization.")


# --- 2x2 Figure Layout ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) Real-space 2D
im_real = axs[0, 0].imshow(Z_real, extent=[support_min, support_max, support_min, support_max],
                           origin='lower', cmap='viridis')
axs[0, 0].set_title(f"Real-Space Kernel ({kernel_type.capitalize()})")
axs[0, 0].set_xlabel(r"$\Delta x$")
axs[0, 0].set_ylabel(r"$\Delta y$")
plt.colorbar(im_real, ax=axs[0, 0], fraction=0.046, pad=0.04)
axs[0, 0].text(0.05, 0.95, eq_text_real_2d, transform=axs[0, 0].transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# (0,1) Fourier-space 2D
im_fourier = axs[0, 1].imshow(Z_fourier, extent=[kmin, kmax, kmin, kmax],
                              origin='lower', cmap='plasma')
axs[0, 1].set_title(f"Fourier-Space Kernel ({kernel_type.capitalize()})")
axs[0, 1].set_xlabel(r"$k_x$")
axs[0, 1].set_ylabel(r"$k_y$")
plt.colorbar(im_fourier, ax=axs[0, 1], fraction=0.046, pad=0.04)
axs[0, 1].text(0.05, 0.95, eq_text_fourier_2d, transform=axs[0, 1].transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# (1,0) Real-space 1D cross-section
axs[1, 0].plot(x_1d, real_1d, 'b-', linewidth=2)
axs[1, 0].set_title("1D Cross-Section (Real Space)")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel(r"$K(x,0)$")
axs[1, 0].text(0.05, 0.95, eq_text_real_1d, transform=axs[1, 0].transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
axs[1, 0].grid(True)

# (1,1) Fourier-space 1D cross-section
axs[1, 1].plot(kx_1d, fourier_1d, 'r-', linewidth=2)
axs[1, 1].set_title("1D Cross-Section (Fourier Space)")
axs[1, 1].set_xlabel(r"$k_x$")
axs[1, 1].set_ylabel(r"$\widehat{K}(k_x,0)$")
axs[1, 1].text(0.05, 0.95, eq_text_fourier_1d, transform=axs[1, 1].transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig(f"{kernel_type}_kernel_function_visualization.png", dpi=300)
plt.close()

logging.info("Kernel real/Fourier and 1D cross-section visualization plot saved.")


total_end_time = time.time()
logging.info("Total execution time: {:.4f} seconds".format(total_end_time - total_start_time))

import os
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
import finufft
from numba import njit, prange
import datetime

# Configure logging to output detailed information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_array_info(name, arr):
    """Log and print summary statistics (shape, min, max, mean) for an array."""
    info = (f"{name}: shape={arr.shape}, min={np.min(arr):.4e}, "
            f"max={np.max(arr):.4e}, mean={np.mean(arr):.4e}")
    logging.info(info)
    print(info)

# --------------------------
# Custom Gaussian Kernel (Real Space)
# --------------------------
def gaussian_kernel(r, h):
    """
    Gaussian kernel in real space:
      K(r) = (1 / (2π h^2)) * exp(-0.5 * (r/h)^2)
    """
    norm = 1.0 / (2.0 * np.pi * h**2)
    return norm * np.exp(-0.5 * (r / h)**2)

# --------------------------
# Fourier Transform of the Gaussian Kernel (Physical Wave Number)
# --------------------------
def gaussian_kernel_ft(k2, h):
    """
    Fourier transform of the Gaussian kernel:
      K̂(k) = (1 / (2π h^2)) * exp(-0.5 * h^2 * k^2)
    where k^2 = kx_phys^2 + ky_phys^2 and k_phys = (2π/L)*k.
    """
    norm = 1.0 / (2.0 * np.pi * h**2)
    return norm * np.exp(-0.5 * h**2 * k2)

# --------------------------
# 1. Direct Summation Method (Periodic Boundary, Cutoff Removed)
# --------------------------
@njit(parallel=True, fastmath=True)
def compute_density_direct_periodic(particles, x_centers, y_centers, Lx, Ly, h):
    """
    Compute the 2D density field via direct summation (Numba accelerated)
    under periodic boundary conditions.

    For each grid point, apply the minimum image convention to compute the
    distance r between a particle and the grid center, and sum the contributions
    using the Gaussian kernel.
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
                # Apply minimum image convention (periodic boundaries)
                dx = (dx + Lx/2) % Lx - Lx/2
                dy = (dy + Ly/2) % Ly - Ly/2
                r = math.sqrt(dx*dx + dy*dy)
                val += (1.0 / (2.0 * math.pi * h * h)) * math.exp(-0.5 * (r / h)**2)
            density[i, j] = val
    return density

class DensityCalculator2D:
    """
    Class for computing the 2D density field using a direct summation
    (Numba accelerated) with periodic boundary conditions.
    """
    def __init__(self, particles, grid_bounds, grid_spacing):
        self.particles = particles
        self.grid_bounds = grid_bounds  # Example: {"x": (-6,6), "y": (-6,6)}
        self.grid_spacing = grid_spacing  # Example: (0.05, 0.05)

    def calculate_density_map(self, h):
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing

        # Define grid centers (spacing 0.05)
        x_centers = np.arange(xmin + dx/2, xmax, dx, dtype=np.float64)
        y_centers = np.arange(ymin + dy/2, ymax, dy, dtype=np.float64)
        x_centers = np.ascontiguousarray(x_centers)
        y_centers = np.ascontiguousarray(y_centers)

        Lx = xmax - xmin
        Ly = ymax - ymin

        logging.info("Direct summation: Starting density computation.")
        density_map = compute_density_direct_periodic(self.particles, x_centers, y_centers, Lx, Ly, h)
        log_array_info("Direct Density Map", density_map)
        return density_map

# --------------------------
# 2. NUFFT Method (Periodic Boundary, isign=-1, constant coefficient scaling, cutoff removed, adjusted upsampfac)
# --------------------------
class NUFFT_KDE:
    """
    Class to compute the Gaussian Kernel Density Estimate (KDE) in 2D using NUFFT
    under periodic boundary conditions.

    Algorithm Steps:
      1) Rescale particle coordinates to the interval [-π, π] (centered).
      2) Utilize a plan-based NUFFT with isign=-1 and upsampfac=3.0 to compute
         the Fourier coefficients.
      3) Rescale the FFT frequencies to physical wave numbers and multiply by
         the Fourier-transformed Gaussian kernel:
         K̂(k) = (1/(2π h^2)) * exp(-0.5 * h^2 * k_phys^2)
      4) Perform an inverse FFT and multiply by the number of particles
         to reconstruct the real-space density field.
    """
    def __init__(self, particles, grid_bounds, grid_spacing, h):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.h = h

        self.xmin, self.xmax = grid_bounds["x"]
        self.ymin, self.ymax = grid_bounds["y"]
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

        nx = int((self.xmax - self.xmin) / self.grid_spacing[0])
        ny = int((self.ymax - self.ymin) / self.grid_spacing[1])
        self.grid_size = (nx, ny)

        logging.info(f"Initialized NUFFT KDE with {particles.shape[0]} particles.")
        logging.info(f"Domain: x=({self.xmin}, {self.xmax}), y=({self.ymin}, {self.ymax}), grid_size={self.grid_size}")

    def compute_density_map(self):
        nx, ny = self.grid_size
        logging.info("NUFFT: Starting density computation using NUFFT method.")

        # (A) Rescale particle coordinates to [-π, π] (centered)
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        x_center = (self.xmin + self.xmax) / 2
        y_center = (self.ymin + self.ymax) / 2
        x_scaled = (x - x_center) * (2 * np.pi / self.Lx)
        y_scaled = (y - y_center) * (2 * np.pi / self.Ly)
        log_array_info("x_scaled", x_scaled)
        log_array_info("y_scaled", y_scaled)

        # (B) Execute the plan-based NUFFT with upsampfac=3.0 and isign=-1
        plan = finufft.Plan(
            1, (nx, ny), isign=-1, eps=1e-6,
            upsampfac=3.0,
            spread_kerevalmeth=0
        )
        plan.setpts(x_scaled, y_scaled)
        c = np.ones_like(x_scaled, dtype=np.complex128)
        F_k = plan.execute(c)
        logging.info(f"NUFFT: Execution complete. F_k has shape: {F_k.shape}")
        print(f"F_k summary: shape = {F_k.shape}, min = {F_k.min():.4e}, max = {F_k.max():.4e}, mean = {F_k.mean():.4e}")

        # (C) Rescale FFT frequencies to physical wave numbers: k_phys = (2π / L)*k
        kx = np.fft.fftfreq(nx, d=1.0) * nx
        ky = np.fft.fftfreq(ny, d=1.0) * ny
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        kx_phys = (2 * np.pi / self.Lx) * kx
        ky_phys = (2 * np.pi / self.Ly) * ky
        k2_phys = kx_phys**2 + ky_phys**2
        log_array_info("kx_phys", kx_phys)
        log_array_info("ky_phys", ky_phys)
        log_array_info("k2_phys", k2_phys)

        # (D) Apply the Fourier-transformed Gaussian kernel
        kernel_ft = gaussian_kernel_ft(k2_phys, self.h)
        log_array_info("kernel_ft", kernel_ft)
        F_k_2d = F_k.reshape((nx, ny))
        G_k_2d = F_k_2d * kernel_ft
        log_array_info("G_k_2d (after kernel multiplication)", G_k_2d)

        # (E) Inverse FFT and multiply by the number of particles
        density_complex = np.fft.ifft2(np.fft.ifftshift(G_k_2d))
        density_map = np.real(density_complex)
        # Normalize by total particle number to match direct summation scale
        density_map *= self.particles.shape[0]

        log_array_info("density_map (NUFFT)", density_map)
        return density_map

# --------------------------
# 3. Main Testing Code
# --------------------------
def main():
    logging.info("Periodic Domain: Comparing Direct Summation vs. NUFFT (Custom Gaussian Kernel, upsampfac=3.0)...")

    # (A) Particle Data Generation (Two Gaussian Clusters)
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
    log_array_info("Particles", particles)

    # (B) Set parameters: Domain [-6, 6] (periodic) and grid spacing of 0.05 → 240x240 grid
    grid_bounds = {"x": (-6, 6), "y": (-6, 6)}
    grid_spacing = (0.05, 0.05)
    h = 0.5

    # (C) Direct Summation Method
    calc2d = DensityCalculator2D(particles, grid_bounds, grid_spacing)
    density_direct = calc2d.calculate_density_map(h)

    # (D) NUFFT Method (Plan-based with upsampfac=3.0)
    kde_nufft = NUFFT_KDE(particles, grid_bounds, grid_spacing, h)
    density_nufft = kde_nufft.compute_density_map()

    # (E) Visualization: Three subplots to compare the particle distribution and density fields
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Original Particle Distribution
    axs[0].scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.5)
    axs[0].set_title("Original Particle Distribution (Periodic)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_xlim(grid_bounds["x"])
    axs[0].set_ylim(grid_bounds["y"])

    # (2) NUFFT Result
    im1 = axs[1].imshow(
        density_nufft,
        extent=[grid_bounds["x"][0], grid_bounds["x"][1],
                grid_bounds["y"][0], grid_bounds["y"][1]],
        origin="lower", cmap="inferno"
    )
    axs[1].set_title("NUFFT Density Field (upsampfac=3.0)")
    fig.colorbar(im1, ax=axs[1], label="Density")

    # (3) Direct Summation Result
    im2 = axs[2].imshow(
        density_direct.T,  # Transpose to match orientation
        extent=[grid_bounds["x"][0], grid_bounds["x"][1],
                grid_bounds["y"][0], grid_bounds["y"][1]],
        origin="lower", cmap="inferno"
    )
    axs[2].set_title("Direct Summation (Custom Kernel)")
    fig.colorbar(im2, ax=axs[2], label="Density")

    # Compute difference metrics
    diff = density_nufft - density_direct
    log_array_info("Difference (NUFFT - Direct)", diff)
    print("RMSE:", np.sqrt(np.mean(diff**2)))
    print("Max absolute error:", np.max(np.abs(diff)))

    # 현재 시간을 'YYYYMMDD_HHMMSS' 형태로 포맷팅
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 저장할 파일 이름에 timestamp_str 삽입
    filename = f"compare_nufft_vs_direct_{timestamp_str}.png"

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    logging.info(f"Completed. Figure saved as {filename}")

if __name__ == "__main__":
    main()

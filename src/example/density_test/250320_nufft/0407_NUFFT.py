import os
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
import finufft
from numba import njit, prange
import datetime
from scipy.special import j1  # For top-hat kernel Fourier transform

# Configure logging to output detailed information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_array_info(name, arr):
    """Log and print summary statistics (shape, min, max, mean) for an array."""
    info = (f"{name}: shape={arr.shape}, min={np.min(arr):.4e}, "
            f"max={np.max(arr):.4e}, mean={np.mean(arr):.4e}")
    logging.info(info)
    print(info)

# --------------------------
# Gaussian Kernel Functions
# --------------------------
def gaussian_kernel(r, h):
    """
    Gaussian kernel in real space:
      K(r) = (1/(2πh^2)) exp(-0.5*(r/h)^2)
    """
    norm = 1.0 / (2.0 * np.pi * h**2)
    return norm * np.exp(-0.5 * (r / h)**2)

def gaussian_kernel_ft(k2, h):
    """
    Fourier transform of the Gaussian kernel:
      K̂(k) = (1/(2πh^2)) exp(-0.5*h^2*k^2)
    """
    norm = 1.0 / (2.0 * np.pi * h**2)
    return norm * np.exp(-0.5 * h**2 * k2)

# --------------------------
# Top-Hat Kernel Functions
# --------------------------
def tophat_kernel(r, h):
    """
    Top-hat kernel in real space:
      K(r) = 1/(πh^2) for r ≤ h, 0 otherwise.
    """
    return 1.0/(np.pi * h**2) if r <= h else 0.0

def tophat_kernel_ft(k2, h):
    """
    Fourier transform of the top-hat kernel in 2D:
      K̂(k) = 2*J1(k*h)/(k*h)
    where J1 is the first-order Bessel function.
    """
    k = np.sqrt(k2)
    kernel_ft = np.empty_like(k)
    kernel_ft[k==0] = 1.0  # limit as k -> 0
    nonzero = k != 0
    kernel_ft[nonzero] = 2 * j1(k[nonzero] * h) / (k[nonzero] * h)
    return kernel_ft

# --------------------------
# Triangle Kernel Functions (Separable)
# --------------------------
def triangle_kernel(dx, dy, h):
    """
    Separable triangle kernel in real space:
      T(x,y) = max(0, 1 - |x|/h) * max(0, 1 - |y|/h)
    """
    tx = (1 - abs(dx)/h) if abs(dx) < h else 0.0
    ty = (1 - abs(dy)/h) if abs(dy) < h else 0.0
    return tx * ty

def triangle_kernel_ft(kx, ky, h):
    """
    Fourier transform of the separable triangle kernel:
      T̂(k_x, k_y) = h^2 * sinc^2(k_x*h/2) * sinc^2(k_y*h/2)
    where sinc(x) = sin(x)/x, with np.sinc(x/np.pi) implementing that definition.
    """
    # np.sinc expects the argument in units of pi, i.e., np.sinc(x/np.pi) = sin(x)/x.
    return h**2 * np.sinc((kx * h / 2) / np.pi)**2 * np.sinc((ky * h / 2) / np.pi)**2

# --------------------------
# 1. Direct Summation Method with Selectable Kernel (Updated to include Triangle)
# --------------------------
@njit(parallel=True, fastmath=True)
def compute_density_direct_periodic(particles, x_centers, y_centers, Lx, Ly, h, kernel_flag):
    """
    Compute the 2D density field via direct summation (Numba accelerated)
    under periodic boundary conditions.
    
    kernel_flag: 0 for Gaussian, 1 for Top-hat, 2 for Triangle.
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
                if kernel_flag == 0:
                    # Gaussian kernel
                    r = math.sqrt(dx*dx+dy*dy)
                    val += (1.0 / (2.0 * math.pi * h * h)) * math.exp(-0.5 * (r/h)**2)
                elif kernel_flag == 1:
                    # Top-hat kernel: radial test based on distance r
                    r = math.sqrt(dx*dx + dy*dy)
                    if r <= h:
                        val += 1.0/(math.pi * h * h)
                    else:
                        val += 0.0
                elif kernel_flag == 2:
                    # Triangle kernel (separable in x and y)
                    tmp1 = (1 - abs(dx)/h) if abs(dx) < h else 0.0
                    tmp2 = (1 - abs(dy)/h) if abs(dy) < h else 0.0
                    val += tmp1 * tmp2
            density[i, j] = val
    return density

class DensityCalculator2D:
    """
    Class for computing the 2D density field using a direct summation
    (Numba accelerated) with periodic boundary conditions and selectable kernel.
    """
    def __init__(self, particles, grid_bounds, grid_spacing, kernel_type="gaussian"):
        self.particles = particles
        self.grid_bounds = grid_bounds  # Example: {"x": (-6,6), "y": (-6,6)}
        self.grid_spacing = grid_spacing  # Example: (0.05, 0.05)
        self.kernel_type = kernel_type.lower()

    def calculate_density_map(self, h):
        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing

        # Define grid centers
        x_centers = np.arange(xmin + dx/2, xmax, dx, dtype=np.float64)
        y_centers = np.arange(ymin + dy/2, ymax, dy, dtype=np.float64)
        x_centers = np.ascontiguousarray(x_centers)
        y_centers = np.ascontiguousarray(y_centers)

        Lx = xmax - xmin
        Ly = ymax - ymin

        # Determine kernel flag: 0 for Gaussian, 1 for Top-hat, 2 for Triangle
        if self.kernel_type == "gaussian":
            kernel_flag = 0
        elif self.kernel_type == "tophat":
            kernel_flag = 1
        elif self.kernel_type == "triangle":
            kernel_flag = 2
        else:
            raise ValueError("Unsupported kernel type. Choose 'gaussian', 'tophat', or 'triangle'.")

        logging.info("Direct summation: Starting density computation.")
        density_map = compute_density_direct_periodic(self.particles, x_centers, y_centers, Lx, Ly, h, kernel_flag)
        log_array_info("Direct Density Map", density_map)
        return density_map

# --------------------------
# 2. NUFFT Method with Selectable Kernel (Updated for Triangle Kernel)
# --------------------------
class NUFFT_KDE:
    """
    Class to compute the Kernel Density Estimate (KDE) in 2D using NUFFT
    under periodic boundary conditions. Selectable kernel: 'gaussian', 'tophat', or 'triangle'.
    
    Algorithm Steps:
      1) Rescale particle coordinates to [-π, π] (centered).
      2) Utilize a plan-based NUFFT with isign=-1 and upsampfac=3.0.
      3) Rescale the FFT frequencies to physical wave numbers and multiply by the
         Fourier-transformed kernel.
      4) Perform an inverse FFT and apply proper normalization.
      5) Save Fourier-space visualization as a PNG file.
    """
    def __init__(self, particles, grid_bounds, grid_spacing, h, kernel_type="gaussian"):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.h = h
        self.kernel_type = kernel_type.lower()

        self.xmin, self.xmax = grid_bounds["x"]
        self.ymin, self.ymax = grid_bounds["y"]
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

        nx = int((self.xmax - self.xmin) / self.grid_spacing[0])
        ny = int((self.ymax - self.ymin) / self.grid_spacing[1])
        self.grid_size = (nx, ny)

        logging.info(f"Initialized NUFFT KDE with {particles.shape[0]} particles using {self.kernel_type} kernel.")
        logging.info(f"Domain: x=({self.xmin}, {self.xmax}), y=({self.ymin}, {self.ymax}), grid_size={self.grid_size}")

    def compute_density_map(self, plot_fourier=True):
        nx, ny = self.grid_size
        logging.info("NUFFT: Starting density computation using NUFFT method.")

        # (A) Rescale particle coordinates to [-π, π]
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
        log_array_info("kx_phys", kx_phys)
        log_array_info("ky_phys", ky_phys)

        # (D) Apply the Fourier-transformed kernel
        if self.kernel_type == "gaussian":
            kernel_ft = gaussian_kernel_ft(kx_phys**2+ky_phys**2, self.h)
        elif self.kernel_type == "tophat":
            kernel_ft = tophat_kernel_ft(kx_phys**2+ky_phys**2, self.h)
        elif self.kernel_type == "triangle":
            kernel_ft = triangle_kernel_ft(kx_phys, ky_phys, self.h)
        else:
            raise ValueError("Unsupported kernel type. Choose 'gaussian', 'tophat', or 'triangle'.")
        log_array_info("kernel_ft", kernel_ft)

        F_k_2d = F_k.reshape((nx, ny))
        G_k_2d = F_k_2d * kernel_ft
        log_array_info("G_k_2d (after kernel multiplication)", G_k_2d)

        # (E) Plot and save Fourier space data as a PNG file
        if plot_fourier:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(np.log10(np.abs(F_k_2d) + 1e-12), origin="lower", cmap="viridis")
            plt.title("NUFFT F_k (log10 |F_k|)")
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(np.log10(np.abs(G_k_2d) + 1e-12), origin="lower", cmap="viridis")
            plt.title("G_k after kernel multiplication (log10 |G_k|)")
            plt.colorbar()
            plt.tight_layout()
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fourier_filename = f"fourier_space_{timestamp_str}.png"
            plt.savefig(fourier_filename, dpi=300)
            logging.info(f"Fourier space visualization saved as {fourier_filename}")
            plt.close()

        # (F) Inverse FFT and proper normalization (multiply by nx*ny)
        density_complex = np.fft.ifft2(np.fft.ifftshift(G_k_2d))
        density_map = np.real(density_complex) * (nx * ny)
        log_array_info("density_map (NUFFT)", density_map)
        return density_map

# --------------------------
# 3. Main Testing Code
# --------------------------
def main():
    logging.info("Periodic Domain: Comparing Direct Summation vs. NUFFT with selectable kernel and Fourier space visualization...")

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

    # (B) Set parameters: Domain [-6, 6] (periodic), grid spacing 0.05, and h=0.5
    grid_bounds = {"x": (-6, 6), "y": (-6, 6)}
    grid_spacing = (0.05, 0.05)
    h = 0.5

    # Select kernel type: 'gaussian', 'tophat', or 'triangle'
    kernel_type = "tophat"

    # (C) Direct Summation Method
    calc2d = DensityCalculator2D(particles, grid_bounds, grid_spacing, kernel_type=kernel_type)
    density_direct = calc2d.calculate_density_map(h)

    # (D) NUFFT Method (Plan-based with upsampfac=3.0)
    kde_nufft = NUFFT_KDE(particles, grid_bounds, grid_spacing, h, kernel_type=kernel_type)
    density_nufft = kde_nufft.compute_density_map(plot_fourier=True)

    # (E) Visualization: Compare particle distribution and density fields
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Original Particle Distribution
    axs[0].scatter(particles[:, 0], particles[:, 1], s=2, alpha=0.5)
    axs[0].set_title("Original Particle Distribution (Periodic)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_xlim(grid_bounds["x"])
    axs[0].set_ylim(grid_bounds["y"])

    # NUFFT Result
    im1 = axs[1].imshow(
        density_nufft,
        extent=[grid_bounds["x"][0], grid_bounds["x"][1],
                grid_bounds["y"][0], grid_bounds["y"][1]],
        origin="lower", cmap="inferno"
    )
    axs[1].set_title(f"NUFFT Density Field ({kernel_type.capitalize()} Kernel)")
    fig.colorbar(im1, ax=axs[1], label="Density")

    # Direct Summation Result
    im2 = axs[2].imshow(
        density_direct.T,  # Transpose to match orientation
        extent=[grid_bounds["x"][0], grid_bounds["x"][1],
                grid_bounds["y"][0], grid_bounds["y"][1]],
        origin="lower", cmap="inferno"
    )
    axs[2].set_title(f"Direct Summation ({kernel_type.capitalize()} Kernel)")
    fig.colorbar(im2, ax=axs[2], label="Density")

    # Compute error metrics
    diff = density_nufft - density_direct
    log_array_info("Difference (NUFFT - Direct)", diff)
    print("RMSE:", np.sqrt(np.mean(diff**2)))
    print("Max absolute error:", np.max(np.abs(diff)))

    # Save the comparison figure with a timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"0414_compare_nufft_vs_direct_{timestamp_str}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    logging.info(f"Completed. Figure saved as {filename}")

if __name__ == "__main__":
    main()

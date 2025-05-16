import os
import numpy as np
import matplotlib.pyplot as plt
import finufft
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class NUFFT_KDE:
    """
    A class to perform 2D kernel density estimation using NUFFT (CPU-based FINUFFT),
    with domain embedding & cropping to mitigate wrap-around effects.
    
    Parameters
    ----------
    particles : ndarray, shape (N,2)
        The nonuniform particle positions (x, y).
    grid_bounds : dict
        {'x':(xmin, xmax), 'y':(ymin, ymax)}  (the final region you want to visualize)
    grid_spacing : tuple
        (dx, dy)
    h : float
        Gaussian kernel bandwidth.
    """

    def __init__(self, particles, grid_bounds, grid_spacing, h):
        self.particles = particles
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.h = h

        # ---------------------------
        # 1) Define "extended" bounds
        #    (expand domain to reduce wrap-around at edges)
        #    예: [-6,6] -> [-8,8]
        # ---------------------------
        margin = 2.0  # 확장할 여유 범위
        self.extended_bounds = {
            "x": (grid_bounds["x"][0] - margin, grid_bounds["x"][1] + margin),
            "y": (grid_bounds["y"][0] - margin, grid_bounds["y"][1] + margin)
        }

        # Original domain size
        self.Lx = grid_bounds["x"][1] - grid_bounds["x"][0]
        self.Ly = grid_bounds["y"][1] - grid_bounds["y"][0]

        # Extended domain size
        self.Lx_ext = self.extended_bounds["x"][1] - self.extended_bounds["x"][0]
        self.Ly_ext = self.extended_bounds["y"][1] - self.extended_bounds["y"][0]

        # 2) Compute extended grid size
        self.extended_grid_size = (
            int(self.Lx_ext / grid_spacing[0]),
            int(self.Ly_ext / grid_spacing[1]),
        )

        # 3) Compute final (original) grid size
        self.grid_size = (
            int(self.Lx / grid_spacing[0]),
            int(self.Ly / grid_spacing[1]),
        )

        logging.info(f"Initialized NUFFT KDE with {particles.shape[0]} particles.")
        logging.info(f"Original bounds: {self.grid_bounds}, extended bounds: {self.extended_bounds}")
        logging.info(f"Grid size (original): {self.grid_size}, extended grid size: {self.extended_grid_size}")

    def compute_density_map(self):
        """
        NUFFT-based 2D KDE with Gaussian kernel, using domain embedding & cropping.
        
        Steps:
          1) Rescale particles to extended domain -> [-π, π].
          2) Type-1 NUFFT to get sum of delta functions in Fourier space.
          3) Multiply by Gaussian kernel's Fourier transform.
          4) Inverse FFT -> extended density map.
          5) Crop the extended map to the original domain.
        """

        # --------------------------
        # 0) 좌표를 "확장 도메인" 기준 [-π, π]로 스케일링
        # --------------------------
        x = self.particles[:, 0]
        y = self.particles[:, 1]

        # Extended domain
        xmin_ext, xmax_ext = self.extended_bounds["x"]
        ymin_ext, ymax_ext = self.extended_bounds["y"]

        Lx_ext = xmax_ext - xmin_ext
        Ly_ext = ymax_ext - ymin_ext

        # 중앙을 (0,0)으로 맞춘 뒤 [-π, π]로 스케일링
        x_scaled = (x - 0.5*(xmin_ext + xmax_ext)) * (2*np.pi / Lx_ext)
        y_scaled = (y - 0.5*(ymin_ext + ymax_ext)) * (2*np.pi / Ly_ext)

        # --------------------------
        # 1) Type-1 NUFFT (sum of deltas)
        # --------------------------
        c = np.ones_like(x_scaled, dtype=np.complex128)  # weights = 1
        ms_ext, mt_ext = self.extended_grid_size  # Fourier grid size for extended domain

        logging.info("Performing Type-1 NUFFT on extended domain...")
        F_k = finufft.nufft2d1(x_scaled, y_scaled, c, (ms_ext, mt_ext), eps=1e-6)

        # --------------------------
        # 2) 가우시안 커널의 푸리에 변환을 곱해주기
        # --------------------------
        kx = np.fft.fftfreq(ms_ext, d=1.0) * ms_ext
        ky = np.fft.fftfreq(mt_ext, d=1.0) * mt_ext
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        # 물리적 wave-number (extended domain)
        kx_phys = (2*np.pi / Lx_ext) * kx
        ky_phys = (2*np.pi / Ly_ext) * ky
        k2_phys = kx_phys**2 + ky_phys**2

        # Gaussian kernel in Fourier space
        kernel_ft = np.exp(-0.5 * (self.h**2) * k2_phys)

        F_k_2d = F_k.reshape((ms_ext, mt_ext))
        G_k_2d = F_k_2d * kernel_ft

        # --------------------------
        # 3) Inverse FFT -> extended density map
        # --------------------------
        density_complex = np.fft.ifft2(np.fft.ifftshift(G_k_2d))
        density_map_extended = np.abs(density_complex)

        # --------------------------
        # 4) Crop the extended map to original domain
        # --------------------------
        # 예: extended domain = [-8,8], original = [-6,6] => margin=2
        # => grid spacing=0.1 => 2 units = 20 grid cells
        # => crop indices: [20 : 20+120] (if original domain is 12 wide => 120 cells)
        margin_cells_x = int((self.extended_bounds["x"][0] - self.grid_bounds["x"][0]) / self.grid_spacing[0] * -1)
        margin_cells_y = int((self.extended_bounds["y"][0] - self.grid_bounds["y"][0]) / self.grid_spacing[1] * -1)

        # 원본 grid_size
        ms, mt = self.grid_size
        # crop
        density_map_cropped = density_map_extended[
            margin_cells_x : margin_cells_x + ms,
            margin_cells_y : margin_cells_y + mt
        ]

        # --------------------------
        # 5) 정규화 (선택)
        # --------------------------
        density_map_cropped /= np.max(density_map_cropped)

        return density_map_cropped

    def save_density(self, density_field, filename="density_field.npy"):
        """Save the computed density field to an npy file."""
        np.save(filename, density_field)
        logging.info(f"Density field saved to {filename}")

    def visualize_results(self, density_field, savefile="kde_result.png"):
        """Visualize both the original particle distribution and the computed density map."""
        logging.info("Visualizing results...")

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot of original particle positions
        axs[0].scatter(self.particles[:, 0], self.particles[:, 1], s=2, alpha=0.5)
        axs[0].set_title("Original Particle Distribution")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")

        # Density field heatmap (cropped domain: self.grid_bounds)
        im = axs[1].imshow(
            density_field,
            extent=[
                self.grid_bounds["x"][0], self.grid_bounds["x"][1],
                self.grid_bounds["y"][0], self.grid_bounds["y"][1]
            ],
            origin="lower", cmap="inferno"
        )
        axs[1].set_title("NUFFT KDE Density Field (Cropped)")
        fig.colorbar(im, ax=axs[1], label="Density")

        plt.tight_layout()
        plt.savefig(savefile, dpi=300)
        plt.show()
        logging.info(f"Visualization saved to {savefile}")


def main():
    """Main function to execute NUFFT KDE computation."""
    data_filename = "particles.npy"
    
    # Step 1: Load or Generate Data
    if os.path.exists(data_filename):
        logging.info("Loading particle data from file...")
        particles = np.load(data_filename)
    else:
        logging.info("Generating new particle data...")
        # (A) 데이터: 2개 가우시안 클러스터
        N = 5000  
        N1 = N // 2
        N2 = N - N1

        cluster1 = np.random.randn(N1, 2)*1.0 + np.array([1, 1])
        cluster2 = np.random.randn(N2, 2)*0.5 + np.array([-2, -2])
        particles = np.vstack([cluster1, cluster2])

        np.save(data_filename, particles)
        logging.info(f"Generated and saved particle data to {data_filename}")

    # Step 2: Define KDE Parameters
    grid_bounds = {"x": (-6, 6), "y": (-6, 6)}
    grid_spacing = (0.1, 0.1)
    h = 0.5  # Kernel bandwidth

    # Step 3: Initialize KDE Object (with embedding)
    kde = NUFFT_KDE(particles, grid_bounds, grid_spacing, h)

    # Step 4: Compute Density Map (NUFFT-based KDE with domain extension & cropping)
    density_field = kde.compute_density_map()

    # Step 5: Save & Visualize Results
    kde.save_density(density_field)
    kde.visualize_results(density_field)


if __name__ == "__main__":
    main()

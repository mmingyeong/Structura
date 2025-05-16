import os
import glob
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.gridspec as gridspec

# 설정 (필요에 따라 수정)
data_dirs = {
    "z=0": "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5",
    "z=127": "/caefs/data/IllustrisTNG/densitymap-ics-hdf5"
}
box_size = 205.0  # cMpc/h
num_bins = 50
output_fig = "power_spectrum_relative_density.png"
theory_file = "test_matterpower.dat"
num_workers = os.cpu_count() or 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_theory_pk(filepath):
    try:
        data = np.loadtxt(filepath)
        return data[:,0], data[:,1]
    except Exception as e:
        logging.error(f"Failed to load theory file {filepath}: {e}")
        return None, None

def compute_power_spectrum_from_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'density' in f:
                density = f['density'][:]
            else:
                key = list(f.keys())[0]
                density = f[key][:]

        rho_bar = np.mean(density)
        # 상대밀도 계산 (1 + delta)
        relative_density = density / rho_bar

        N = density.shape[0]
        box_volume = box_size ** 3

        # FFT 및 파워 스펙트럼 계산
        delta_k = np.fft.fftn(relative_density)
        delta_k = np.fft.fftshift(delta_k)
        power_k = (np.abs(delta_k) ** 2) / box_volume

        k_vals = np.fft.fftfreq(N, d=box_size / N)
        kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag = np.fft.fftshift(k_mag)

        k_flat = k_mag.flatten()
        p_flat = power_k.flatten()

        valid = k_flat > 0
        k_flat = k_flat[valid]
        p_flat = p_flat[valid]

        k_min_clip = max(k_flat.min(), 1e-5)
        k_bins = np.logspace(np.log10(k_min_clip), np.log10(k_flat.max()), num_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        P_k = np.full_like(k_centers, np.nan)

        for i in range(len(k_centers)):
            in_bin = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            if np.any(in_bin):
                P_k[i] = np.mean(p_flat[in_bin])

        filename = os.path.basename(filepath).rsplit('.',1)[0]
        return (k_centers, P_k, filename)

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None

def main():
    start_time = time.time()

    hdf5_files = []
    file_zlabels = []
    for z_label, dir_path in data_dirs.items():
        files = sorted(glob.glob(os.path.join(dir_path, "*.hdf5")))
        if not files:
            logging.warning(f"No HDF5 files found in {dir_path} for {z_label}")
        else:
            logging.info(f"Found {len(files)} files in {dir_path} for {z_label}")
        hdf5_files.extend(files)
        file_zlabels.extend([z_label]*len(files))

    if not hdf5_files:
        logging.error("No HDF5 files found.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result, z_label in tqdm(zip(executor.map(compute_power_spectrum_from_file, hdf5_files), file_zlabels), total=len(hdf5_files)):
            if result:
                k_centers, P_k, filename = result
                results.append((k_centers, P_k, filename, z_label))

    if not results:
        logging.error("No valid results.")
        return

    fig = plt.figure(figsize=(10,9))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1], hspace=0.05)

    ax0 = plt.subplot(gs[0])
    reference_k = None

    for k_vals, P_k, label, z_label in results:
        valid = ~np.isnan(P_k)
        ax0.loglog(k_vals[valid], P_k[valid], label=f"{z_label}: {label}")
        if reference_k is None:
            reference_k = k_vals

    k_theory, Pk_theory = load_theory_pk(theory_file)
    if k_theory is not None and Pk_theory is not None and reference_k is not None:
        valid = (reference_k >= k_theory.min()) & (reference_k <= k_theory.max())
        k_plot = reference_k[valid]
        Pk_interp = np.interp(k_plot, k_theory, Pk_theory)
        ax0.loglog(k_plot, Pk_interp, 'k--', linewidth=2, label='Theory (CAMB)')

        ax1 = plt.subplot(gs[1], sharex=ax0)
        for k_vals, P_k, label, z_label in results:
            valid = (k_vals >= k_theory.min()) & (k_vals <= k_theory.max())
            valid &= ~np.isnan(P_k)
            if np.any(valid):
                ratio = P_k[valid] / np.interp(k_vals[valid], k_theory, Pk_theory)
                ax1.semilogx(k_vals[valid], ratio, label=f"{z_label}: {label}")

        ax1.axhline(1.0, color='k', linestyle='--')
        ax1.set_ylabel(r'$P_{\mathrm{sim}}/P_{\mathrm{CAMB}}$')
        ax1.set_xlabel(r'Wavenumber $k$ [$h\,\mathrm{Mpc}^{-1}$]')
        ax1.grid(True, which='both', linestyle=':')
        ax1.set_ylim(0.01, 100)

    ax0.set_ylabel(r"$P(k)$ [$({h}^{-1}\,\mathrm{Mpc})^3$]")
    ax0.set_title("3D Power Spectrum from Relative Density")
    ax0.grid(True, which='both', linestyle=':')
    ax0.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    logging.info(f"Saved figure: {output_fig}")
    plt.show()

    elapsed = time.time() - start_time
    logging.info(f"Total runtime: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

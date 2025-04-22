#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.gridspec as gridspec  # 상단/하단 plot 나누기 위해 추가

# ------------------------------
# 설정
# ------------------------------
data_dir = "/caefs/data/IllustrisTNG/density_map_99"
box_size = 205.0  # in cMpc/h
num_bins = 50
output_fig = "tng300_power_spectra.png"
theory_file = "test_matterpower.dat"  # 이론 파워 스펙트럼 경로
num_workers = os.cpu_count() or 4  # 병렬 처리에 사용할 코어 수

# ------------------------------
# 로깅 설정
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------
# 이론 P(k) 불러오기 함수
# ------------------------------
def load_theory_pk(filepath):
    try:
        data = np.loadtxt(filepath)
        k_theory = data[:, 0]
        Pk_theory = data[:, 1]
        logging.info(f"Theory P(k) loaded from: {filepath}")
        return k_theory, Pk_theory
    except Exception as e:
        logging.error(f"Failed to load theory file {filepath}: {e}")
        return None, None

# ------------------------------
# FFT → Power Spectrum 함수
# ------------------------------
def compute_power_spectrum_from_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'density' in f:
                density = f['density'][:]
            else:
                key = list(f.keys())[0]
                density = f[key][:]

        rho_bar = np.mean(density)
        delta = (density - rho_bar) / rho_bar

        N = density.shape[0]
        box_volume = box_size ** 3  # 전체 부피 [Mpc/h]^3

        # FFT and proper normalization
        delta_k = np.fft.fftn(delta)
        delta_k = np.fft.fftshift(delta_k)
        power_k = (np.abs(delta_k)**2) / box_volume  # ✅ 정규화 적용됨

        # k-vector 계산
        k_vals = np.fft.fftfreq(N, d=box_size/N)
        kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag = np.fft.fftshift(k_mag)

        k_flat = k_mag.flatten()
        p_flat = power_k.flatten()

        valid = k_flat > 0
        k_flat = k_flat[valid]
        p_flat = p_flat[valid]

        k_bins = np.logspace(np.log10(k_flat.min()), np.log10(k_flat.max()), num_bins)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        P_k = np.zeros_like(k_centers)

        for i in range(len(k_centers)):
            in_bin = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
            if np.any(in_bin):
                P_k[i] = np.mean(p_flat[in_bin])
            else:
                P_k[i] = np.nan

        filename = os.path.basename(filepath).replace(".hdf5", "")
        return (k_centers, P_k, filename)

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None

# ------------------------------
# 메인 실행
# ------------------------------
def main():
    start_time = time.time()
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    if not hdf5_files:
        logging.error("No HDF5 files found.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(compute_power_spectrum_from_file, hdf5_files), total=len(hdf5_files)):
            if result:
                results.append(result)

    # 시각화: 메인 + ratio plot
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # --- (1) Main power spectrum plot ---
    ax0 = plt.subplot(gs[0])
    reference_k = None
    ratios = []

    for k_vals, P_k, label in results:
        ax0.loglog(k_vals, P_k, label=label)
        if reference_k is None:
            reference_k = k_vals

    # Load and interpolate theory
    k_theory, Pk_theory = load_theory_pk(theory_file)
    if k_theory is not None and Pk_theory is not None and reference_k is not None:
        valid = (reference_k >= k_theory.min()) & (reference_k <= k_theory.max())
        k_plot = reference_k[valid]
        Pk_interp = np.interp(k_plot, k_theory, Pk_theory)
        ax0.loglog(k_plot, Pk_interp, 'k--', linewidth=2.0, label='Theory (CAMB)')

        # --- (2) Ratio plot ---
        ax1 = plt.subplot(gs[1], sharex=ax0)
        for k_vals, P_k, label in results:
            valid = (k_vals >= k_theory.min()) & (k_vals <= k_theory.max())
            ratio = P_k[valid] / np.interp(k_vals[valid], k_theory, Pk_theory)
            ax1.semilogx(k_vals[valid], ratio, label=label)

        ax1.axhline(1.0, color='k', linestyle='--', linewidth=1)
        ax1.set_ylabel(r'$P_{\mathrm{sim}}/P_{\mathrm{CAMB}}$', fontsize=11)
        ax1.set_xlabel(r'Wavenumber $k$ [$h\,\mathrm{Mpc}^{-1}$]', fontsize=13)
        ax1.grid(True, which='both', linestyle=':')
        ax1.set_ylim(0.01, 100)

    # 메인 plot 라벨
    ax0.set_ylabel(r"$P(k)$ [$({h}^{-1}\,\mathrm{Mpc})^3$]", fontsize=13)
    ax0.set_title("3D Power Spectrum and Ratio to CAMB", fontsize=15)
    ax0.grid(True, which='both', ls=':')
    ax0.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig("tng300_power_spectra_with_ratio.png", dpi=300)
    logging.info("Figure with ratio plot saved as: tng300_power_spectra_with_ratio.png")
    plt.show()


    elapsed = time.time() - start_time
    logging.info(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def check_density_hdf5(file_path, plot=True):
    print(f"\n📂 Checking: {os.path.basename(file_path)}")

    with h5py.File(file_path, 'r') as f:
        # ─────────────────────────────────────
        # 1. 밀도 데이터 가져오기
        keys = ['density_map', 'density', 'kde_density', 'density_kde']
        dataset_key = next((k for k in keys if k in f), None)

        if not dataset_key:
            print("✗ No valid density dataset found.")
            return

        dset = f[dataset_key]
        density = dset[:]

        # ─────────────────────────────────────
        # 2. 메타데이터
        box_size = f.attrs.get("box_size", 205.0)
        grid_spacing = f.attrs.get("grid_spacing", box_size / density.shape[0])
        bandwidth = f.attrs.get("bandwidth", -1.0)
        cutoff_sigma = f.attrs.get("cutoff_sigma", 3.0)
        kernel = f.attrs.get("kernel", "unknown").decode() if isinstance(f.attrs.get("kernel"), bytes) else f.attrs.get("kernel", "unknown")

        N_particles = f.attrs.get("N_particles", None)
        N_cells = np.prod(density.shape)

        # ─────────────────────────────────────
        # 3. 통계 분석
        print("🔍 Density Statistics:")
        print(f" - shape          : {density.shape}")
        print(f" - dtype          : {density.dtype}")
        print(f" - min            : {density.min():.4e}")
        print(f" - max            : {density.max():.4e}")
        print(f" - mean           : {density.mean():.4e}")
        print(f" - sum            : {density.sum():.4e}")
        print(f" - has NaN        : {np.isnan(density).any()}")
        print(f" - has Inf        : {np.isinf(density).any()}")

        if N_particles:
            expected_mean = N_particles / N_cells
            print(f" - expected mean  : {expected_mean:.4e} (given N_particles = {N_particles})")

        # ─────────────────────────────────────
        # 4. 로그 변환 통계
        log_offset = 1e-4
        log_density = np.log10(density + log_offset)
        print("🧮 Log10(Density + 1e-4) Statistics:")
        print(f" - min            : {log_density.min():.4f}")
        print(f" - max            : {log_density.max():.4f}")
        print(f" - mean           : {log_density.mean():.4f}")
        print(f" - std            : {log_density.std():.4f}")
        print(f" - p1 / p99       : {np.percentile(log_density, 1):.4f} / {np.percentile(log_density, 99):.4f}")

        # ─────────────────────────────────────
        # 5. 커널 관련 정보
        print("🧾 KDE Metadata:")
        print(f" - kernel         : {kernel}")
        print(f" - grid_spacing   : {grid_spacing}")
        print(f" - bandwidth      : {bandwidth}")
        print(f" - cutoff_radius  : {cutoff_sigma} × bandwidth = {cutoff_sigma * bandwidth if bandwidth > 0 else 'N/A'}")
        print(f" - bandwidth / dx : {bandwidth / grid_spacing if bandwidth > 0 else 'N/A'}")

        # ─────────────────────────────────────
        # 6. 시각화
        if plot:
            slice_idx = density.shape[0] // 2
            proj = np.sum(density, axis=0)
            log_proj = np.log10(proj + log_offset)

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(density[slice_idx, :, :], cmap='inferno', origin='lower')
            axs[0].set_title(f"Slice at X={slice_idx}")
            axs[1].imshow(log_proj, cmap='cividis', origin='lower')
            axs[1].set_title("Log10 Projection along X")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_density_map_hdf5.py <density_file.hdf5>")
    else:
        check_density_hdf5(sys.argv[1])

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_densitymap_hdf5_sample.py

This script scans up to 3 .hdf5 density map files in the specified directory and reports:
- Group keys
- Basic statistics (shape, dtype, min, max, sum)
- NaN / Inf presence

Author: Mingyeong Yang
Date: 2025-04-22
"""

import os
import h5py
import numpy as np
import random

def report_density_map(filepath):
    with h5py.File(filepath, 'r') as f:
        for key in f:
            obj = f[key]

            if isinstance(obj, h5py.Group):
                if 'density_map' in obj:
                    print(f"[✓] Found 'density_map' in group '{key}'")
                    data = obj['density_map'][:]
                    label = f"{key}/density_map"
                else:
                    print(f"[✗] No 'density_map' in group '{key}'")
                    continue
            elif isinstance(obj, h5py.Dataset):
                if key in ['density_map', 'x_centers', 'y_centers', 'z_centers']:
                    print(f"[✓] Found top-level dataset: '{key}'")
                    data = obj[:]
                    label = key
                else:
                    print(f"[!] '{key}' is not a recognized dataset")
                    continue
            else:
                print(f"[!] '{key}' is not a group or dataset (type: {type(obj)})")
                continue

            print(f"  └─ Path: {label}")
            print(f"     - shape : {data.shape}")
            print(f"     - dtype : {data.dtype}")
            print(f"     - min   : {np.nanmin(data):.4e}")
            print(f"     - max   : {np.nanmax(data):.4e}")
            print(f"     - mean  : {np.nanmean(data):.4e}")
            print(f"     - sum   : {np.nansum(data):.4e}")
            print(f"     - has NaN: {np.isnan(data).any()}")
            print(f"     - has Inf: {np.isinf(data).any()}")



def main():
    #test_dir = '/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5/uniform_dx0.82'
    test_dir = '/caefs/data/IllustrisTNG/densitymap-ics-hdf5/uniform_dx0.82'
    #test_dir = 'test'
    hdf5_files = [f for f in os.listdir(test_dir) if f.endswith('.hdf5')]

    print(f"📦 Total HDF5 files in directory: {len(hdf5_files)}")

    if not hdf5_files:
        print("❌ No HDF5 files found.")
        return

    selected_files = random.sample(hdf5_files, min(1, len(hdf5_files)))
    print(f"🧪 Selected files for report: {selected_files}\n")

    for fname in selected_files:
        print(f"==> Reporting: {fname}")
        report_density_map(os.path.join(test_dir, fname))
        print("")

if __name__ == '__main__':
    main()

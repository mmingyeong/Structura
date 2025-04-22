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
        for group_name in f:
            grp = f[group_name]
            if isinstance(grp, h5py.Group):
                if 'density_map' in grp:
                    print(f"[✓] Found 'density_map' in group '{group_name}'")
                else:
                    print(f"[✗] No 'density_map' in group '{group_name}'")
            else:
                print(f"[!] '{group_name}' is not a group (type: {type(grp)})")
                continue

            data = grp['density_map'][:]
            print(f"  └─ Group: {group_name}")
            print(f"     - shape : {data.shape}")
            print(f"     - dtype : {data.dtype}")
            print(f"     - min   : {np.nanmin(data):.4e}")
            print(f"     - max   : {np.nanmax(data):.4e}")
            print(f"     - sum   : {np.nansum(data):.4e}")
            print(f"     - has NaN: {np.isnan(data).any()}")
            print(f"     - has Inf: {np.isinf(data).any()}")

def main():
    test_dir = '/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5'
    hdf5_files = [f for f in os.listdir(test_dir) if f.endswith('.hdf5')]

    print(f"📦 Total HDF5 files in directory: {len(hdf5_files)}")

    if not hdf5_files:
        print("❌ No HDF5 files found.")
        return

    selected_files = random.sample(hdf5_files, min(3, len(hdf5_files)))
    print(f"🧪 Selected files for report: {selected_files}\n")

    for fname in selected_files:
        print(f"==> Reporting: {fname}")
        report_density_map(os.path.join(test_dir, fname))
        print("")

if __name__ == '__main__':
    main()

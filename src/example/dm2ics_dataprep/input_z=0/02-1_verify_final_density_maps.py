#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_final_density_maps.py

HDF5 밀도 맵 병합 결과의 정확성을 확인하는 스크립트입니다.
- 총합 비교 (원본 vs 병합본)
- NaN, Inf 값 여부 확인

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Date: 2025-05-12
"""

import os
import h5py
import numpy as np
from tqdm import tqdm

# 경로 설정
base_dir = "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5"
pairs = [
    ("triangular_dx0.41", "final_triangular_dx0.41.hdf5"),
    ("triangular_dx0.82", "final_triangular_dx0.82.hdf5"),
    ("uniform_dx0.41", "final_uniform_dx0.41.hdf5"),
    ("uniform_dx0.82", "final_uniform_dx0.82.hdf5"),
]

def sum_all_density_values_in_folder(folder_path):
    total_sum = 0.0
    for fname in tqdm(sorted(os.listdir(folder_path)), desc=f"Summing {os.path.basename(folder_path)}"):
        if fname.endswith(".hdf5") or fname.endswith(".h5"):
            path = os.path.join(folder_path, fname)
            try:
                with h5py.File(path, "r") as f:
                    total_sum += np.sum(f["density_map"])
            except Exception as e:
                print(f"  ⚠️ 오류 발생: {path} - {e}")
    return total_sum

def inspect_final_file(final_path):
    with h5py.File(final_path, "r") as f:
        data = f["density_map"][:]
        total = np.sum(data)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        return total, has_nan, has_inf, data.shape, np.min(data), np.max(data)

# 검증 루프
for folder, final_file in pairs:
    folder_path = os.path.join(base_dir, folder)
    final_path = os.path.join(base_dir, final_file)

    print(f"\n📁 검증 시작: {folder} -> {final_file}")
    folder_sum = sum_all_density_values_in_folder(folder_path)
    final_sum, has_nan, has_inf, shape, min_val, max_val = inspect_final_file(final_path)

    print(f"  📦 원본 총합    : {folder_sum:.6e}")
    print(f"  ✅ 병합본 총합   : {final_sum:.6e}")
    print(f"  🔍 총합 차이     : {abs(folder_sum - final_sum):.6e}")
    print(f"  🧼 NaN 포함 여부 : {has_nan}")
    print(f"  🧼 Inf 포함 여부 : {has_inf}")
    print(f"  📐 shape         : {shape}")
    print(f"  ⬇ 최소값         : {min_val:.4e}")
    print(f"  ⬆ 최대값         : {max_val:.4e}")
    print("-" * 60)

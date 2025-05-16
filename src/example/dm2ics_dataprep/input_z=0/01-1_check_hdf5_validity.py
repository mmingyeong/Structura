#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

TARGET_DIR = "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5/"
INVALID_LOG_PATH = "invalid_density_files.txt"
SKIP_DIR_NAME = "res_0.16"
NUM_WORKERS = min(cpu_count(), 8)

# -------------------------------------------------------------------------------------
def fast_check(file_path):
    if time.time() - os.path.getmtime(file_path) < 10:
        return (file_path, "SKIPPED_RECENTLY_MODIFIED")
    try:
        with h5py.File(file_path, "r") as f:
            if "density_map" not in f:
                return (file_path, "MISSING_density_map")
    except Exception as e:
        return (file_path, "CORRUPTED")
    return (file_path, "OK")

# -------------------------------------------------------------------------------------
def detailed_check(file_path):
    result = {
        "file": file_path,
        "status": "OK",
        "min": None,
        "max": None,
        "mean": None,
        "error": None
    }
    try:
        with h5py.File(file_path, "r") as f:
            dset = f["density_map"]
            data = dset[...]
            result["min"] = float(np.nanmin(data))
            result["max"] = float(np.nanmax(data))
            result["mean"] = float(np.nanmean(data))

            if np.isnan(data).all():
                result["status"] = "ALL_NAN"
            elif np.count_nonzero(data) == 0:
                result["status"] = "ALL_ZERO"
            elif result["min"] == result["max"]:
                result["status"] = "MIN_EQ_MAX"
    except Exception as e:
        result["status"] = "CORRUPTED"
        result["error"] = str(e)
    return result

# -------------------------------------------------------------------------------------
def gather_hdf5_files():
    hdf5_files = []
    for root, _, files in os.walk(TARGET_DIR):
        if SKIP_DIR_NAME in root:
            continue
        for fname in files:
            if fname.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, fname))
    return sorted(hdf5_files)

# -------------------------------------------------------------------------------------
def run_with_progress(pool, func, iterable, desc):
    results = []
    for res in tqdm(pool.imap_unordered(func, iterable), total=len(iterable), desc=desc):
        results.append(res)
    return results

# -------------------------------------------------------------------------------------
def main():
    all_files = gather_hdf5_files()
    print(f"📦 Found {len(all_files)} HDF5 files (excluding {SKIP_DIR_NAME}).")

    with Pool(NUM_WORKERS) as pool:
        fast_results = run_with_progress(pool, fast_check, all_files, desc="🔍 Fast checking")

    to_check_detailed = [f for f, status in fast_results if status != "OK"]
    print(f"⚠️  {len(to_check_detailed)} files require detailed validation...")

    invalid_results = []
    if to_check_detailed:
        with Pool(NUM_WORKERS) as pool:
            detailed_results = run_with_progress(pool, detailed_check, to_check_detailed, desc="🔬 Detailed checking")

        for res in detailed_results:
            if res["status"] != "OK":
                msg = f"[✗] {res['file']} - {res['status']} | min={res['min']} max={res['max']} mean={res['mean']} | {res.get('error')}"
                print(msg)
                invalid_results.append(msg)

    if invalid_results:
        with open(INVALID_LOG_PATH, "w") as f:
            for line in invalid_results:
                f.write(line + "\n")
        print(f"\n⚠️  {len(invalid_results)} invalid files logged to {INVALID_LOG_PATH}")
    else:
        print("✅ All files passed metadata and value checks.")

if __name__ == "__main__":
    main()

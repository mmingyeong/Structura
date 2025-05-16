#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import h5py
import time
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

TARGET_DIR = "/caefs/data/IllustrisTNG/densitymap-ics-hdf5/sum"
SKIP_DIR_NAME = None
NUM_WORKERS = min(cpu_count(), 8)

def setup_logger(task_id):
    log_filename = f"check_task{task_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

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

def gather_hdf5_files():
    hdf5_files = []
    for root, _, files in os.walk(TARGET_DIR):
        if SKIP_DIR_NAME and SKIP_DIR_NAME in root:
            continue
        for fname in files:
            if fname.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, fname))
    return sorted(hdf5_files)


def run_with_progress(pool, func, iterable, desc):
    results = []
    for res in tqdm(pool.imap_unordered(func, iterable), total=len(iterable), desc=desc):
        results.append(res)
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python 02_check_density_values_array.py <TASK_ID>")
        sys.exit(1)

    task_id = int(sys.argv[1])
    setup_logger(task_id)

    total_chunks = 4
    all_files = gather_hdf5_files()
    chunk_size = len(all_files) // total_chunks
    start = task_id * chunk_size
    end = (task_id + 1) * chunk_size if task_id < total_chunks - 1 else len(all_files)
    target_files = all_files[start:end]

    logging.info(f"📦 Task {task_id} handling files {start} to {end-1} ({len(target_files)} files)")

    invalid_results = []
    with Pool(NUM_WORKERS) as pool:
        detailed_results = run_with_progress(pool, detailed_check, target_files, desc=f"🔬 Checking Task {task_id}")

    for res in detailed_results:
        if res["status"] != "OK":
            msg = f"[✗] {res['file']} - {res['status']} | min={res['min']} max={res['max']} mean={res['mean']} | {res.get('error')}"
            logging.warning(msg)
            invalid_results.append(msg)

    log_path = f"invalid_density_files_task{task_id}.txt"
    if invalid_results:
        with open(log_path, "w") as f:
            for line in invalid_results:
                f.write(line + "\n")
        logging.warning(f"⚠️ Task {task_id} wrote {len(invalid_results)} invalid files to {log_path}")
    else:
        logging.info(f"✅ Task {task_id}: all files OK.")

if __name__ == "__main__":
    main()

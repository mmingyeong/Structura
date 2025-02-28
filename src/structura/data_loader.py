#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: data_loader.py
# structura/data_loader.py

import os
import numpy as np
import cupy as cp
import time
import gc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from structura.logger import logger  # 공통 로거 가져오기
from structura.config import clear_gpu_memory

class DataLoader:
    """Loads TNG300-1 dark matter simulation data with X-range filtering."""

    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_chunk(self, file_path, x_min=None, x_max=None):
        """Loads a data file and applies X-range filtering if specified."""
        try:
            data_cpu = np.load(file_path)['data']
            if x_min is not None and x_max is not None:
                data_cpu = data_cpu[(data_cpu[:, 0] >= x_min) & (data_cpu[:, 0] <= x_max)]
            return data_cpu
        except KeyError:
            logger.error(f"Key 'data' not found in {file_path}. Skipping file.")
            return None

    def load_all_chunks(self, x_min=None, x_max=None):
        """Loads all data chunks in the folder using parallel processing."""
        file_list = sorted([os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(".npz")])
        if not file_list:
            logger.error(f"No .npz files found in {self.folder_path}. Exiting.")
            exit(1)

        start_time = time.time()
        filtered_chunks = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            for chunk in tqdm(executor.map(self.load_chunk, file_list, [x_min] * len(file_list), [x_max] * len(file_list)), 
                              desc="Loading chunks", total=len(file_list)):
                if chunk is not None:
                    filtered_chunks.append(chunk)
                    if len(filtered_chunks) % 1000 == 0:
                        clear_gpu_memory()

        if not filtered_chunks:
            logger.error("No valid data loaded. Exiting.")
            exit(1)

        total_time = time.time() - start_time
        logger.info(f"✅ Time taken to load all chunks: {total_time:.2f}s")

        return cp.vstack([cp.asarray(chunk) for chunk in filtered_chunks])

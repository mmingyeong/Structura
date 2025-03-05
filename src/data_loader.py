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
from concurrent.futures import ProcessPoolExecutor
from logger import logger  # 공통 로거 가져오기
from config import clear_gpu_memory

def load_chunk_worker(args):
    """
    Helper function to load a single data chunk with memory mapping,
    X-range filtering, and optional sampling.

    Parameters
    ----------
    args : tuple
        A tuple containing (file_path, x_min, x_max, sampling_rate).
        - file_path (str): Path to the .npy or .npz file.
        - x_min (float or None): Minimum X for filtering.
        - x_max (float or None): Maximum X for filtering.
        - sampling_rate (float): Fraction of data to sample, 0.0 < sampling_rate <= 1.0.
                                1.0 means full data (no sampling).
    
    Returns
    -------
    np.ndarray or None
        Filtered and sampled data array (as a regular numpy array), or None if loading fails.
    """
    file_path, x_min, x_max, sampling_rate = args
    try:
        # 1) Load file with memory mapping (for .npy).
        if file_path.endswith(".npz"):
            data_cpu = np.load(file_path)['data']
        elif file_path.endswith(".npy"):
            data_cpu = np.load(file_path, mmap_mode='r')
        else:
            logger.warning(f"Skipping non-npz/npy file: {file_path}")
            return None

        # 2) X-range filtering (CPU-based).
        if x_min is not None and x_max is not None:
            data_cpu = data_cpu[(data_cpu[:, 0] >= x_min) & (data_cpu[:, 0] <= x_max)]

        # 3) Sampling (if sampling_rate < 1.0).
        #    sampling_rate=1.0 => 100% (no sampling)
        if 0.0 < sampling_rate < 1.0:
            num_samples = int(len(data_cpu) * sampling_rate)
            if num_samples < len(data_cpu):
                data_cpu = data_cpu[np.random.choice(len(data_cpu), num_samples, replace=False)]

        # For safety, copy memory-mapped array to a normal array.
        data_cpu = np.copy(data_cpu)
        
        return data_cpu

    except KeyError:
        logger.error(f"Key 'data' not found in {file_path}. (Applies to .npz files with a different key.) Skipping.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        return None

class DataLoader:
    """
    Loads TNG300-1 dark matter simulation data (in npz or npy format) with X-range filtering
    and optional sampling.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .npz or .npy files.
    use_gpu : bool, optional
        If True, the final combined array is converted to a CuPy array for GPU processing.
        If False, a NumPy array is returned. Default is True.
    """

    def __init__(self, folder_path: str, use_gpu: bool = True) -> None:
        self.folder_path = folder_path
        self.use_gpu = use_gpu

    def load_all_chunks(
        self,
        x_min: float = None,
        x_max: float = None,
        sampling_rate: float = 1.0
    ):
        """
        Loads all data chunks (.npz or .npy) in the folder using process-based parallel processing
        with optional X-range filtering and sampling.

        Parameters
        ----------
        x_min : float, optional
            Minimum X value for filtering.
        x_max : float, optional
            Maximum X value for filtering.
        sampling_rate : float, optional
            Fraction of data to sample (0.0 < sampling_rate <= 1.0).
            - 1.0 means full data (no sampling).
            - 0.5 means 50% of the data.
            - 0.01 means 1% of the data.
            Default is 1.0.

        Returns
        -------
        cupy.ndarray or numpy.ndarray
            Combined array of all filtered and sampled chunks. If use_gpu is True, returns a CuPy array;
            otherwise, returns a NumPy array.
        """
        # Find all .npz or .npy files in the folder.
        file_list = sorted([
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith(".npz") or f.endswith(".npy")
        ])

        if not file_list:
            logger.error(f"No .npz or .npy files found in {self.folder_path}. Exiting.")
            exit(1)

        start_time = time.time()
        filtered_chunks = []

        # Prepare arguments for parallel loading.
        args_list = [(f, x_min, x_max, sampling_rate) for f in file_list]

        # Use process-based parallel loading to bypass the GIL.
        with ProcessPoolExecutor(max_workers=4) as executor:
            for chunk in tqdm(
                executor.map(load_chunk_worker, args_list),
                desc="Loading chunks",
                total=len(args_list)
            ):
                if chunk is not None:
                    filtered_chunks.append(chunk)
                    # Periodically clear GPU memory (if relevant).
                    if len(filtered_chunks) % 1000 == 0:
                        clear_gpu_memory()

        if not filtered_chunks:
            logger.error("No valid data loaded. Exiting.")
            exit(1)

        total_time = time.time() - start_time
        logger.info(f"✅ Time taken to load all chunks: {total_time:.2f}s")

        # Combine all chunks into a single contiguous array.
        combined_np = np.vstack([np.ascontiguousarray(chunk) for chunk in filtered_chunks])
        if self.use_gpu:
            return cp.asarray(combined_np)
        else:
            return combined_np

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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from logger import logger
from config import clear_gpu_memory


def load_chunk_worker(args):
    """
    Load a single data chunk using memory mapping, with optional filtering along a specified projection axis and sampling.

    Parameters
    ----------
    args : tuple
        A tuple containing:
            file_path (str): Path to the .npy or .npz file.
            x_min (float or None): Minimum value for filtering.
            x_max (float or None): Maximum value for filtering.
            sampling_rate (float): Fraction of data to sample (0.0 < sampling_rate <= 1.0). A value of 1.0 indicates no sampling.
            projection_axis (int): Column index used for filtering.

    Returns
    -------
    np.ndarray or None
        The filtered and sampled data as a NumPy array, or None if the file could not be loaded.
    """
    file_path, x_min, x_max, sampling_rate, projection_axis = args
    try:
        # Load the file using memory mapping (for .npy) or full loading (for .npz).
        if file_path.endswith(".npz"):
            data_cpu = np.load(file_path)["data"]
        elif file_path.endswith(".npy"):
            data_cpu = np.load(file_path, mmap_mode="r")
        else:
            logger.warning(f"Skipping unsupported file format: {file_path}")
            return None

        # Apply filtering based on the specified projection axis.
        if x_min is not None and x_max is not None:
            data_cpu = data_cpu[
                (data_cpu[:, projection_axis] >= x_min)
                & (data_cpu[:, projection_axis] <= x_max)
            ]

        # Apply sampling if a sampling rate less than 1.0 is specified.
        if 0.0 < sampling_rate < 1.0:
            num_samples = int(len(data_cpu) * sampling_rate)
            if num_samples < len(data_cpu):
                data_cpu = data_cpu[
                    np.random.choice(len(data_cpu), num_samples, replace=False)
                ]

        # Convert memory-mapped array to a standard NumPy array.
        data_cpu = np.copy(data_cpu)
        return data_cpu

    except KeyError:
        logger.error(
            f"Key 'data' not found in {file_path}. This may occur with .npz files using a different key. Skipping file."
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error while loading {file_path}: {e}")
        return None


class DataLoader:
    """
    Class for loading and processing TNG300-1 dark matter simulation data from .npz or .npy files.

    The class supports filtering data along a specified projection axis and optional random sampling,
    with support for GPU-accelerated processing using CuPy.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing .npz or .npy files.
    use_gpu : bool, optional
        If True, the final combined array is converted to a CuPy array for GPU processing;
        otherwise, a NumPy array is returned. Default is True.
    """

    def __init__(self, folder_path: str, use_gpu: bool = True) -> None:
        self.folder_path = folder_path
        self.use_gpu = use_gpu

    def load_all_chunks(
        self,
        x_min: float = None,
        x_max: float = None,
        sampling_rate: float = 1.0,
        projection_axis: int = 0,
    ):
        """
        Load and combine all data chunks (.npz or .npy files) in the specified folder using process-based parallel processing.

        Optional filtering along a specified projection axis and random sampling can be applied.

        Parameters
        ----------
        x_min : float, optional
            Minimum value for filtering data along the projection axis.
        x_max : float, optional
            Maximum value for filtering data along the projection axis.
        sampling_rate : float, optional
            Fraction of data to sample (0.0 < sampling_rate <= 1.0). A value of 1.0 indicates no sampling.
        projection_axis : int, optional
            Column index used for filtering the data. Default is 0.

        Returns
        -------
        cupy.ndarray or numpy.ndarray
            A combined array containing all the filtered and sampled data chunks.
            If 'use_gpu' is True, the data is returned as a CuPy array; otherwise, as a NumPy array.
        """
        # Identify all .npz or .npy files in the specified folder.
        file_list = sorted(
            [
                os.path.join(self.folder_path, f)
                for f in os.listdir(self.folder_path)
                if f.endswith(".npz") or f.endswith(".npy")
            ]
        )

        if not file_list:
            logger.error(
                f"No .npz or .npy files found in {self.folder_path}. Terminating data loading process."
            )
            exit(1)

        start_time = time.time()
        filtered_chunks = []

        # Prepare the list of arguments for parallel processing.
        args_list = [
            (f, x_min, x_max, sampling_rate, projection_axis) for f in file_list
        ]

        # Use process-based parallelism to bypass the Global Interpreter Lock (GIL).
        with ProcessPoolExecutor(max_workers=4) as executor:
            for chunk in tqdm(
                executor.map(load_chunk_worker, args_list),
                desc="Loading data chunks",
                total=len(args_list),
            ):
                if chunk is not None:
                    filtered_chunks.append(chunk)
                    # Periodically clear GPU memory if necessary.
                    if len(filtered_chunks) % 1000 == 0:
                        clear_gpu_memory()

        if not filtered_chunks:
            logger.error(
                "No valid data chunks were loaded. Terminating data loading process."
            )
            exit(1)

        total_time = time.time() - start_time
        logger.info(
            f"Time elapsed for loading all data chunks: {total_time:.2f} seconds"
        )

        # Combine all chunks into a single contiguous array.
        combined_np = np.vstack(
            [np.ascontiguousarray(chunk) for chunk in filtered_chunks]
        )

        # Log combined data information and statistics.
        logger.info(f"Combined data shape: {combined_np.shape}")
        if combined_np.shape[0] > 0:
            try:
                data_mean = np.mean(combined_np, axis=0)
                data_median = np.median(combined_np, axis=0)
                data_min = np.min(combined_np, axis=0)
                data_max = np.max(combined_np, axis=0)
                logger.info(
                    f"Data statistics per column:\n"
                    f"  Mean: {data_mean}\n"
                    f"  Median: {data_median}\n"
                    f"  Minimum: {data_min}\n"
                    f"  Maximum: {data_max}"
                )
                logger.info(
                    f"Projection axis {projection_axis} range: min = {data_min[projection_axis]}, max = {data_max[projection_axis]}"
                )
            except Exception as e:
                logger.warning(f"Unable to compute detailed data statistics: {e}")
        else:
            logger.error(
                "No data available after filtering. Please verify the filtering parameters or input files."
            )

        if self.use_gpu:
            return cp.asarray(combined_np)
        else:
            return combined_np

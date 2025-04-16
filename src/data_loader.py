#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-28
# @Filename: data_loader.py
# Description: Data loading and processing for TNG300-1 dark matter simulation.
#              Provides filtering, sampling, GPU conversion, and subcube partitioning/saving.

import os
import time
import gc
from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np
import cupy as cp
import h5py  # For HDF5 saving support
from logger import logger

# Dask-related modules
from dask import delayed, compute
try:
    from dask.distributed import Client, as_completed
except ImportError:
    Client = None
from dask.diagnostics import ProgressBar  # For progress monitoring
from tqdm import tqdm
import dask.config  # For adjusting Dask settings

# Global configuration parameters
GPU_CACHE_VALID_PERIOD: int = 5  # GPU cache validity period in seconds
_gpu_memory_cache: dict = {}

BATCH_SIZE: int = 5  # Batch size for processing files
GROUP_SIZE: int = 2  # Group size to reduce input dependency per task


def get_least_used_gpu() -> int:
    """
    Selects the GPU with the most available memory.
    
    Returns
    -------
    int
        GPU device ID with the largest free memory.
    """
    global _gpu_memory_cache
    try:
        current_time: float = time.time()
        # Use cached result if within valid period
        if _gpu_memory_cache.get("timestamp", 0) + GPU_CACHE_VALID_PERIOD > current_time:
            return _gpu_memory_cache.get("best_device", 0)

        num_devices: int = cp.cuda.runtime.getDeviceCount()
        best_device: int = 0
        best_free: int = 0
        for i in range(num_devices):
            with cp.cuda.Device(i):
                free, _ = cp.cuda.runtime.memGetInfo()
                if free > best_free:
                    best_free = free
                    best_device = i
        logger.info(f"Selected GPU {best_device} with free memory {best_free} bytes.")
        _gpu_memory_cache = {"best_device": best_device, "timestamp": current_time}
        return best_device
    except Exception as e:
        logger.warning(f"Failed to get least used GPU: {e}. Defaulting to GPU 0.")
        return 0


def load_chunk_worker(
    file_path: str,
    x_min: Optional[float],
    x_max: Optional[float],
    sampling_rate: float,
    projection_axis: int,
) -> Optional[np.ndarray]:
    """
    Loads, filters, samples, and copies data from a single .npy file.
    """
    try:
        if not file_path.endswith(".npy"):
            logger.warning(f"Skipping unsupported file format: {file_path}")
            return None

        data_cpu: np.ndarray = np.load(file_path, mmap_mode="r")
        if x_min is not None and x_max is not None:
            data_cpu = data_cpu[
                (data_cpu[:, projection_axis] >= x_min) &
                (data_cpu[:, projection_axis] <= x_max)
            ]
        if 0.0 < sampling_rate < 1.0:
            num_samples: int = int(len(data_cpu) * sampling_rate)
            if num_samples < len(data_cpu):
                indices = np.random.choice(len(data_cpu), num_samples, replace=False)
                data_cpu = data_cpu[indices]
        result: np.ndarray = np.copy(data_cpu)
        del data_cpu
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Unexpected error while loading {file_path}: {e}")
        return None


def load_batch_worker(
    file_paths: List[str],
    x_min: Optional[float],
    x_max: Optional[float],
    sampling_rate: float,
    projection_axis: int,
) -> List[np.ndarray]:
    """
    Processes a batch of files by loading each file sequentially.
    """
    batch_results: List[np.ndarray] = []
    for file_path in file_paths:
        result = load_chunk_worker(file_path, x_min, x_max, sampling_rate, projection_axis)
        if result is not None:
            batch_results.append(result)
    gc.collect()
    return batch_results


def load_cube_worker_periodic(
    file_path: str,
    cube_origin: List[float],
    cube_size: float,
    full_length: float,
    sampling_rate: float,
) -> Optional[np.ndarray]:
    """
    Loads particles from a file that fall within a subcube under periodic boundary conditions.
    """
    try:
        if not file_path.endswith(".npy"):
            logger.warning(f"Skipping unsupported file format: {file_path}")
            return None

        data_cpu: np.ndarray = np.load(file_path, mmap_mode="r")
        masks = []
        for dim in range(3):
            origin_dim: float = cube_origin[dim]
            end_dim: float = origin_dim + cube_size
            if end_dim <= full_length:
                mask_dim = (data_cpu[:, dim] >= origin_dim) & (data_cpu[:, dim] < end_dim)
            else:
                mask_dim = (data_cpu[:, dim] >= origin_dim) | (data_cpu[:, dim] < (end_dim - full_length))
            masks.append(mask_dim)
        cube_mask = np.logical_and.reduce(masks)
        cube_data: np.ndarray = data_cpu[cube_mask]
        
        if 0.0 < sampling_rate < 1.0:
            num_samples: int = int(len(cube_data) * sampling_rate)
            if num_samples < len(cube_data):
                indices = np.random.choice(len(cube_data), num_samples, replace=False)
                cube_data = cube_data[indices]
        result: np.ndarray = cube_data.copy()
        del data_cpu, cube_data
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Unexpected error while loading periodic cube from {file_path}: {e}")
        return None


def get_initial_workers() -> int:
    """
    Determines the initial number of workers as half of the available CPU cores.
    """
    cpu_count: Optional[int] = os.cpu_count() or 1
    return max(1, cpu_count // 2)


def group_tasks(tasks: List, group_size: int) -> List:
    """
    Groups several delayed tasks into batches of the specified group size.
    """
    grouped = []
    for i in range(0, len(tasks), group_size):
        group = tasks[i : i + group_size]
        grouped_task = delayed(lambda results: [item for sublist in results for item in sublist])(group)
        grouped.append(grouped_task)
    return grouped


def concatenate_and_convert(chunks: List[np.ndarray], use_gpu: bool) -> Union[np.ndarray, cp.ndarray]:
    """
    Concatenates a list of NumPy arrays and optionally converts the result to a CuPy array.
    Uses an asynchronous stream for GPU data transfer.
    """
    try:
        combined_np: np.ndarray = np.concatenate(chunks, axis=0)
    except MemoryError as me:
        logger.error(f"MemoryError during data concatenation: {me}")
        raise
    logger.info(f"Combined data shape: {combined_np.shape}")
    if use_gpu:
        gpu_id: int = get_least_used_gpu()
        cp.cuda.Device(gpu_id).use()
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            gpu_data: cp.ndarray = cp.asarray(combined_np)
        stream.synchronize()
        del combined_np
        gc.collect()
        return gpu_data
    else:
        return combined_np


def process_cube_batch(
    batch: List[str],
    cube_origin: List[float],
    cube_size: float,
    full_length: float,
    sampling_rate: float,
) -> List[Optional[np.ndarray]]:
    """
    Processes a batch of files for periodic subcube loading.
    """
    return [
        load_cube_worker_periodic(fp, cube_origin, cube_size, full_length, sampling_rate)
        for fp in batch
    ]


class DataLoader:
    """
    Loads and processes TNG300-1 dark matter simulation data from .npy files.
    Provides options for filtering, sampling, and returns the final data as a CuPy or NumPy array.
    Also supports subcube partitioning and saving functionalities.
    """

    def __init__(self, folder_path: str, use_gpu: bool = True) -> None:
        self.folder_path: str = folder_path
        self.use_gpu: bool = use_gpu

    ############################################################################
    # 1) Data Loading Methods
    ############################################################################

    def load_all_chunks(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        sampling_rate: float = 1.0,
        projection_axis: int = 0,
        workers: Optional[int] = None,
        statistics: bool = False,
    ) -> Union[np.ndarray, cp.ndarray]:
        """
        Loads all .npy files in the folder, applies filtering and sampling,
        and concatenates them into a single array.
        """
        file_list: List[str] = sorted(
            [
                os.path.join(self.folder_path, f)
                for f in os.listdir(self.folder_path)
                if f.endswith(".npy")
            ]
        )
        print(f"Total number of files: {len(file_list)}")
        if not file_list:
            logger.error(f"No .npy files found in {self.folder_path}.")
            raise RuntimeError("No .npy files found in the specified folder.")

        start_time: float = time.time()

        batches: List[List[str]] = [
            file_list[i : i + BATCH_SIZE] for i in range(0, len(file_list), BATCH_SIZE)
        ]
        delayed_tasks = [
            delayed(load_batch_worker)(batch, x_min, x_max, sampling_rate, projection_axis)
            for batch in batches
        ]
        grouped_tasks = group_tasks(delayed_tasks, GROUP_SIZE)

        filtered_chunks: List[np.ndarray] = []
        if Client is not None:
            if workers is None:
                workers = get_initial_workers()
            with Client(n_workers=workers, threads_per_worker=1, memory_limit="4GB", dashboard_address=":8788") as client:
                dask.config.set({
                    "distributed.worker.heartbeat_interval": "1s",
                    "distributed.worker.timeout": "60s"
                })
                logger.info(f"Using Dask distributed client with {workers} workers.")
                futures = client.compute(grouped_tasks)
                with tqdm(total=len(futures), desc="Loading batches") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            filtered_chunks.extend(result)
                        except Exception as e:
                            logger.error(f"Error in a Dask task: {e}")
                        pbar.update(1)
                client.cancel(futures)
        else:
            logger.info("Dask distributed client not available. Using default scheduler.")
            with ProgressBar():
                batch_results_grouped = compute(*grouped_tasks)
            for group in batch_results_grouped:
                filtered_chunks.extend(group)

        if not filtered_chunks:
            logger.error("No valid data chunks were loaded.")
            raise RuntimeError("No valid data chunks were loaded.")

        total_time = time.time() - start_time
        logger.info(f"Time elapsed for loading all data chunks: {total_time:.2f} seconds")

        try:
            combined_np: np.ndarray = np.concatenate(filtered_chunks, axis=0)
        except MemoryError as me:
            logger.error(f"MemoryError during data concatenation: {me}")
            raise

        if statistics:
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
                    f"Projection axis {projection_axis} range: min = {data_min[projection_axis]}, "
                    f"max = {data_max[projection_axis]}"
                )
            except Exception as e:
                logger.warning(f"Unable to compute data statistics: {e}")

        del filtered_chunks
        gc.collect()

        return concatenate_and_convert([combined_np], self.use_gpu)

    def load_cube_data(
        self,
        cube_origin: List[float],
        cube_size: float,
        full_length: float,
        sampling_rate: float = 1.0,
        workers: Optional[int] = None,
    ) -> Union[np.ndarray, cp.ndarray]:
        """
        Loads and concatenates data from all .npy files that fall within a specified cube,
        applying periodic boundary conditions.
        """
        file_list: List[str] = sorted(
            [
                os.path.join(self.folder_path, f)
                for f in os.listdir(self.folder_path)
                if f.endswith(".npy")
            ]
        )
        print(f"Total number of files: {len(file_list)}")
        if not file_list:
            logger.error(f"No .npy files found in {self.folder_path}.")
            raise RuntimeError("No .npy files found in the specified folder.")

        start_time: float = time.time()

        batches: List[List[str]] = [
            file_list[i : i + BATCH_SIZE] for i in range(0, len(file_list), BATCH_SIZE)
        ]
        delayed_tasks = [
            delayed(process_cube_batch)(batch, cube_origin, cube_size, full_length, sampling_rate)
            for batch in batches
        ]
        grouped_tasks = group_tasks(delayed_tasks, GROUP_SIZE)

        cube_chunks: List[np.ndarray] = []
        if Client is not None:
            if workers is None:
                workers = get_initial_workers()
            with Client(n_workers=workers, threads_per_worker=1, memory_limit="4GB", dashboard_address=":8788") as client:
                dask.config.set({
                    "distributed.worker.heartbeat_interval": "1s",
                    "distributed.worker.timeout": "60s"
                })
                logger.info(f"Using Dask distributed client with {workers} workers.")
                futures = client.compute(grouped_tasks)
                with tqdm(total=len(futures), desc="Loading cube batches") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            cube_chunks.extend(result)
                        except Exception as e:
                            logger.error(f"Error in a Dask task: {e}")
                        pbar.update(1)
                client.cancel(futures)
        else:
            logger.info("Dask distributed client not available. Using default scheduler.")
            with ProgressBar():
                batch_results_grouped = compute(*grouped_tasks)
            for group in batch_results_grouped:
                cube_chunks.extend(group)

        if not cube_chunks:
            logger.error("No valid cube data chunks were loaded.")
            raise RuntimeError("No valid data chunks were loaded.")

        total_time = time.time() - start_time
        logger.info(f"Time elapsed for loading cube data chunks: {total_time:.2f} seconds")

        try:
            combined_cube: np.ndarray = np.concatenate(cube_chunks, axis=0)
        except MemoryError as me:
            logger.error(f"MemoryError during cube data concatenation: {me}")
            raise

        del cube_chunks
        gc.collect()

        return concatenate_and_convert([combined_cube], self.use_gpu)

    ############################################################################
    # 2) Subcube Partitioning and Saving Methods
    ############################################################################

    @staticmethod
    def compute_subcube_parameters(
        full_length: float, central_size: float, overlap: float
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """
        Computes subcube centers and load bounds along a single axis.
        """
        N: int = int(np.ceil(full_length / central_size))
        # Vectorized calculation for centers
        centers: List[float] = (np.arange(N) * central_size + central_size / 2).tolist()
        load_bounds: List[Tuple[float, float]] = []
        for center in centers:
            load_start: float = (center - central_size / 2 - overlap) % full_length
            load_end: float = (center + central_size / 2 + overlap) % full_length
            load_bounds.append((load_start, load_end))
        return centers, load_bounds

    @staticmethod
    def extract_central_region(
        data: Union[np.ndarray, cp.ndarray],
        origin: List[float],
        central_size: float,
        overlap: float,
        full_length: float,
    ) -> Union[np.ndarray, cp.ndarray]:
        """
        Extracts the central region from a subcube, handling periodic wrap-around.
        If the input data is a CuPy array, GPU operations are used.
        """
        # Determine whether to use numpy or cupy operations
        use_cp = isinstance(data, cp.ndarray)
        masks = []
        for dim in range(3):
            start: float = (origin[dim] + overlap) % full_length
            end: float = (start + central_size) % full_length
            if start < end:
                mask_dim = (data[:, dim] >= start) & (data[:, dim] < end)
            else:
                mask_dim = (data[:, dim] >= start) | (data[:, dim] < end)
            masks.append(mask_dim)
        if use_cp:
            # Combine masks using iterative logical_and instead of reduce.
            central_mask = masks[0]
            for m in masks[1:]:
                central_mask = cp.logical_and(central_mask, m)
        else:
            central_mask = np.logical_and.reduce(masks)
        return data[central_mask]


    @staticmethod
    def save_npy(data: np.ndarray, filename: str) -> None:
        """
        Saves data as a NumPy binary file.
        """
        np.save(filename, data)

    @staticmethod
    def save_hdf5(data: np.ndarray, filename: str) -> None:
        """
        Saves data in HDF5 format with gzip compression.
        """
        with h5py.File(filename, 'w') as f:
            f.create_dataset('positions', data=data, compression='gzip')

    def save_subcubes(
        self,
        output_dir: str,
        full_length: float = 205.0,
        central_size: float = 10.0,
        overlap: Optional[float] = None,
        bandwidth: Optional[float] = None,
        sampling_rate: float = 1.0,
        workers: int = 48,
        use_gpu_override: Optional[bool] = None,
        save_format: str = "npy",
        extract_center_only: bool = False,
    ) -> None:
        """
        Partitions the entire domain into multiple subcubes and saves each to a file.
        """
        if overlap is None:
            if bandwidth is not None:
                overlap = 3.0 * bandwidth
                logger.info(f"Automatically computed overlap = 3 * bandwidth = {overlap}")
            else:
                raise ValueError("Either overlap or bandwidth must be specified.")

        if use_gpu_override is not None:
            original_use_gpu: bool = self.use_gpu
            self.use_gpu = use_gpu_override
        else:
            original_use_gpu = self.use_gpu

        os.makedirs(output_dir, exist_ok=True)

        centers, bounds = self.compute_subcube_parameters(full_length, central_size, overlap)
        origins_1d: List[float] = [b[0] for b in bounds]
        subcube_origins: List[Tuple[float, float, float]] = list(product(origins_1d, repeat=3))

        logger.info(f"Total number of subcubes: {len(subcube_origins)}")

        for idx, origin in enumerate(subcube_origins):
            logger.info(f"[{idx+1}/{len(subcube_origins)}] Loading subcube at origin={origin}")
            cube_size_total: float = central_size + 2 * overlap

            data = self.load_cube_data(
                cube_origin=list(origin),
                cube_size=cube_size_total,
                full_length=full_length,
                sampling_rate=sampling_rate,
                workers=workers,
            )

            if data is None or len(data) == 0:
                logger.warning(f"No data found for subcube {idx} at {origin}. Skipping.")
                continue

            if extract_center_only:
                data = self.extract_central_region(data, list(origin), central_size, overlap, full_length)
                logger.info(f"Extracted central region shape: {data.shape}")

            filename_base: str = os.path.join(output_dir, f"subcube_{idx:04d}")

            if save_format == "npy":
                self.save_npy(data, f"{filename_base}.npy")
                logger.info(f"Saved subcube {idx} to {filename_base}.npy with shape {data.shape}")
            elif save_format == "hdf5":
                self.save_hdf5(data, f"{filename_base}.h5")
                logger.info(f"Saved subcube {idx} to {filename_base}.h5 with shape {data.shape}")
            else:
                raise ValueError("Unsupported save format. Choose 'npy' or 'hdf5'.")

        self.use_gpu = original_use_gpu

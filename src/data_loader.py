#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-28
# @Filename: data_loader.py
# structura/data_loader.py

import os
import time
import gc
import numpy as np
import cupy as cp
from logger import logger

# Dask 관련 모듈 import
from dask import delayed, compute
try:
    from dask.distributed import Client, as_completed
except ImportError:
    Client = None
from dask.diagnostics import ProgressBar  # 진행 상황 표시
from tqdm import tqdm
import dask.config  # Dask 설정 조정을 위해 임포트

# GPU 캐시 유효 기간 (초)
GPU_CACHE_VALID_PERIOD = 5
_gpu_memory_cache = {}

# 배치 처리에 사용할 batch size (예: 5로 축소)
BATCH_SIZE = 5
# 그룹화 크기를 줄여서 한 태스크당 input dependency 크기를 낮춥니다 (예: 2)
GROUP_SIZE = 2


def get_least_used_gpu() -> int:
    """
    현재 가장 여유 메모리가 많은 GPU를 선택합니다.
    
    Returns
    -------
    int
        가장 free memory가 큰 GPU device ID.
    """
    global _gpu_memory_cache
    try:
        current_time = time.time()
        if _gpu_memory_cache.get("timestamp", 0) + GPU_CACHE_VALID_PERIOD > current_time:
            return _gpu_memory_cache.get("best_device", 0)

        num_devices = cp.cuda.runtime.getDeviceCount()
        best_device = 0
        best_free = 0
        for i in range(num_devices):
            with cp.cuda.Device(i):
                free, total = cp.cuda.runtime.memGetInfo()
                if free > best_free:
                    best_free = free
                    best_device = i
        logger.info(f"Selected GPU {best_device} with free memory {best_free} bytes.")
        _gpu_memory_cache = {"best_device": best_device, "timestamp": current_time}
        return best_device
    except Exception as e:
        logger.warning(f"Failed to get least used GPU: {e}. Defaulting to GPU 0.")
        return 0


def load_chunk_worker(file_path, x_min, x_max, sampling_rate, projection_axis):
    """
    단일 파일에 대해 데이터를 로딩, 필터링, 샘플링하고 메모리 복사를 수행합니다.
    
    Parameters
    ----------
    file_path : str
        .npy 파일 경로.
    x_min : float or None
        필터링용 최소값.
    x_max : float or None
        필터링용 최대값.
    sampling_rate : float
        샘플링 비율 (0.0 < sampling_rate <= 1.0).
    projection_axis : int
        필터링에 사용될 열 인덱스.
    
    Returns
    -------
    np.ndarray or None
        처리된 NumPy 배열, 또는 오류 발생 시 None.
    """
    try:
        if not file_path.endswith(".npy"):
            logger.warning(f"Skipping unsupported file format: {file_path}")
            return None

        data_cpu = np.load(file_path, mmap_mode="r")
        if x_min is not None and x_max is not None:
            data_cpu = data_cpu[
                (data_cpu[:, projection_axis] >= x_min) &
                (data_cpu[:, projection_axis] <= x_max)
            ]
        if 0.0 < sampling_rate < 1.0:
            num_samples = int(len(data_cpu) * sampling_rate)
            if num_samples < len(data_cpu):
                indices = np.random.choice(len(data_cpu), num_samples, replace=False)
                data_cpu = data_cpu[indices]
        result = np.copy(data_cpu)
        del data_cpu
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Unexpected error while loading {file_path}: {e}")
        return None


def load_batch_worker(file_paths, x_min, x_max, sampling_rate, projection_axis):
    """
    여러 파일을 한 번에 로딩하는 배치 작업 함수.
    
    Parameters
    ----------
    file_paths : list of str
        로딩할 .npy 파일 경로 리스트.
    x_min : float or None
        필터링용 최소값.
    x_max : float or None
        필터링용 최대값.
    sampling_rate : float
        샘플링 비율.
    projection_axis : int
        필터링에 사용될 열 인덱스.
    
    Returns
    -------
    list of np.ndarray
        각 파일에서 로딩된 데이터 배열의 리스트 (오류 발생한 파일은 제외).
    """
    batch_results = []
    for file_path in file_paths:
        result = load_chunk_worker(file_path, x_min, x_max, sampling_rate, projection_axis)
        if result is not None:
            batch_results.append(result)
    gc.collect()
    return batch_results


def load_cube_worker_periodic(file_path, cube_origin, cube_size, full_length, sampling_rate):
    """
    Loads data from a single .npy file and filters points that lie within the cube
    defined by cube_origin (lower corner) and cube_size (edge length), 
    applying periodic boundary conditions.
    
    If the cube spans the domain boundary, the function selects points from both
    the high-end and low-end portions of the domain.
    
    Parameters
    ----------
    file_path : str
        Path to the .npy file.
    cube_origin : tuple of float
        Lower corner coordinates (x0, y0, z0) of the cube.
    cube_size : float
        Edge length of the cube.
    full_length : float
        Size of the full domain (e.g., 205 Mpc/h).
    sampling_rate : float
        Fraction of points to sample (0.0 < sampling_rate <= 1.0).
    
    Returns
    -------
    np.ndarray or None
        Filtered data points within the periodic cube as a NumPy array; None on error.
    """
    try:
        if not file_path.endswith(".npy"):
            logger.warning(f"Skipping unsupported file format: {file_path}")
            return None

        data_cpu = np.load(file_path, mmap_mode="r")
        masks = []
        for dim in range(3):
            origin = cube_origin[dim]
            end = origin + cube_size
            if end <= full_length:
                mask = (data_cpu[:, dim] >= origin) & (data_cpu[:, dim] < end)
            else:
                # Wrap-around: combine points from [origin, full_length) and [0, end - full_length)
                mask1 = data_cpu[:, dim] >= origin
                mask2 = data_cpu[:, dim] < (end - full_length)
                mask = mask1 | mask2
            masks.append(mask)
        # Combine masks across all dimensions.
        cube_mask = masks[0] & masks[1] & masks[2]
        cube_data = data_cpu[cube_mask]
        
        if 0.0 < sampling_rate < 1.0:
            num_samples = int(len(cube_data) * sampling_rate)
            if num_samples < len(cube_data):
                indices = np.random.choice(len(cube_data), num_samples, replace=False)
                cube_data = cube_data[indices]
        result = np.copy(cube_data)
        del data_cpu, cube_data
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Unexpected error while loading periodic cube from {file_path}: {e}")
        return None


def get_initial_workers() -> int:
    """
    사용 가능한 CPU 코어 수의 절반을 사용하여 초기 worker 수를 결정합니다.
    
    Returns
    -------
    int
        초기 worker 수.
    """
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // 2)


def group_tasks(tasks, group_size):
    """
    여러 delayed 태스크를 지정한 크기로 그룹화하여 하나의 태스크로 합칩니다.
    
    Parameters
    ----------
    tasks : list
        delayed 태스크 리스트.
    group_size : int
        그룹 당 태스크 개수.
        
    Returns
    -------
    list
        그룹화된 delayed 태스크 리스트.
    """
    grouped = []
    for i in range(0, len(tasks), group_size):
        group = tasks[i:i+group_size]
        grouped_task = delayed(lambda results: [item for sublist in results for item in sublist])(group)
        grouped.append(grouped_task)
    return grouped


class DataLoader:
    """
    .npy 파일로부터 TNG300-1 dark matter simulation 데이터를 로딩 및 처리합니다.
    필터링 및 샘플링 옵션을 제공하며, 최종 결합 결과는 CuPy 또는 NumPy 배열로 반환됩니다.
    
    Parameters
    ----------
    folder_path : str
        .npy 파일들이 있는 디렉토리 경로.
    use_gpu : bool, optional
        True이면 최종 배열을 CuPy 배열로 변환하여 GPU 처리. Default는 True.
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
        workers: int = None,
        statistics: bool = False,
    ):
        """
        폴더 내의 모든 .npy 파일을 로딩하고 필터링/샘플링 후 하나의 배열로 결합합니다.
        
        Parameters
        ----------
        x_min : float, optional
            필터링용 최소 값.
        x_max : float, optional
            필터링용 최대 값.
        sampling_rate : float, optional
            샘플링 비율 (0.0 < sampling_rate <= 1.0).
        projection_axis : int, optional
            필터링에 사용될 열 인덱스.
        workers : int, optional
            사용될 worker 수.
        statistics : bool, optional
            True이면 로딩된 데이터 통계 정보를 출력.
        
        Returns
        -------
        cupy.ndarray or numpy.ndarray
            결합된 데이터 배열 (use_gpu가 True이면 CuPy 배열).
        """
        file_list = sorted([
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith(".npy")
        ])
        print(f"Total number of files: {len(file_list)}")
        if not file_list:
            logger.error(f"No .npy files found in {self.folder_path}. Terminating data loading process.")
            raise RuntimeError("No .npy files found in the specified folder.")

        start_time = time.time()

        batches = [file_list[i:i+BATCH_SIZE] for i in range(0, len(file_list), BATCH_SIZE)]
        delayed_tasks = [
            delayed(load_batch_worker)(batch, x_min, x_max, sampling_rate, projection_axis)
            for batch in batches
        ]
        grouped_tasks = group_tasks(delayed_tasks, GROUP_SIZE)

        filtered_chunks = []
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
            logger.error("No valid data chunks were loaded. Terminating data loading process.")
            raise RuntimeError("No valid data chunks were loaded.")

        total_time = time.time() - start_time
        logger.info(f"Time elapsed for loading all data chunks: {total_time:.2f} seconds")

        try:
            combined_np = np.concatenate(filtered_chunks, axis=0)
        except MemoryError as me:
            logger.error(f"MemoryError during data concatenation: {me}")
            raise

        logger.info(f"Combined data shape: {combined_np.shape}")

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
                logger.warning(f"Unable to compute detailed data statistics: {e}")

        del filtered_chunks
        gc.collect()

        if self.use_gpu:
            gpu_id = get_least_used_gpu()
            cp.cuda.Device(gpu_id).use()
            gpu_data = cp.asarray(combined_np)
            del combined_np
            gc.collect()
            return gpu_data
        else:
            return combined_np

    def load_cube_data(self, cube_origin, cube_size, full_length, sampling_rate=1.0, workers=None):
        """
        Loads and concatenates data from all .npy files that fall within a specified cube,
        applying periodic boundary conditions.
        
        Parameters
        ----------
        cube_origin : tuple of float
            The lower corner (x0, y0, z0) of the cube.
        cube_size : float
            Edge length of the cube.
        full_length : float
            Full domain size (e.g., 205 Mpc/h).
        sampling_rate : float, optional
            Sampling rate (0.0 < sampling_rate <= 1.0) for data reduction. Default is 1.0.
        workers : int, optional
            Number of parallel workers. If None, a default is chosen.
        
        Returns
        -------
        cupy.ndarray or np.ndarray
            Concatenated data points that fall within the cube (periodic boundary applied).
            If self.use_gpu is True, the data is returned as a CuPy array.
        """
        file_list = sorted([
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith(".npy")
        ])
        print(f"Total number of files: {len(file_list)}")
        if not file_list:
            logger.error(f"No .npy files found in {self.folder_path}.")
            raise RuntimeError("No .npy files found in the specified folder.")

        start_time = time.time()

        batches = [file_list[i:i+BATCH_SIZE] for i in range(0, len(file_list), BATCH_SIZE)]
        # Create delayed tasks using the periodic cube worker.
        delayed_tasks = [
            delayed(lambda batch: [load_cube_worker_periodic(fp, cube_origin, cube_size, full_length, sampling_rate) 
                                    for fp in batch])(batch)
            for batch in batches
        ]
        grouped_tasks = group_tasks(delayed_tasks, GROUP_SIZE)

        cube_chunks = []
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
            combined_np = np.concatenate(cube_chunks, axis=0)
        except MemoryError as me:
            logger.error(f"MemoryError during cube data concatenation: {me}")
            raise

        logger.info(f"Combined cube data shape: {combined_np.shape}")

        del cube_chunks
        gc.collect()

        if self.use_gpu:
            gpu_id = get_least_used_gpu()
            cp.cuda.Device(gpu_id).use()
            gpu_data = cp.asarray(combined_np)
            del combined_np
            gc.collect()
            return gpu_data
        else:
            return combined_np

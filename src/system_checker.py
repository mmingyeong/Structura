#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang
# @Date: 2025-03-04
# @Filename: system_checker.py

import platform
import psutil
import numpy as np
import cupy as cp
from datetime import date
from logger import logger

# Handle PyTorch import and version retrieval.
try:
    import torch

    torch_version = torch.__version__
except ModuleNotFoundError:
    torch_version = "Not Installed"

# Automatically set the last update date to today's date.
LAST_UPDATE_DATE = date.today().strftime("%Y-%m-%d")

# Define recommended GPUs for different computational purposes.
RECOMMENDED_GPUS = {
    "Deep Learning": [
        "NVIDIA H100",
        "NVIDIA A100",
        "RTX 4090",
        "RTX 3090",
        "Tesla V100",
    ],
    "High Performance Computing": ["NVIDIA H100", "A100", "Tesla P100"],
    "General Computing": ["RTX 4060", "RTX 3060", "GTX 1660 Super"],
}


class SystemChecker:
    """
    A class to evaluate the system environment by checking GPU availability, CPU and memory specifications,
    and the versions of key Python libraries. It also provides recommendations for GPU upgrades based on performance.
    """

    def __init__(self, verbose=False):
        """
        Initializes the SystemChecker instance.

        Parameters
        ----------
        verbose : bool, optional
            If True, detailed environment analysis is logged. Default is False.
        """
        self.use_gpu = self.check_gpu_availability()
        self.verbose = verbose
        self.gpu_info = []
        self.recommended_gpus = set()
        self.cpu_cores = None
        self.total_memory = None
        self.python_version = None
        self.numpy_version = None
        self.cupy_version = None
        self.torch_version = torch_version

    def check_gpu_availability(self):
        """
        Checks if a GPU is available.

        Returns
        -------
        bool
            True if at least one GPU is available, False otherwise.
        """
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return True
        except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
            pass
        return False

    def check_gpu(self):
        """
        Analyzes the GPU performance and generates upgrade recommendations if applicable.
        The method collects GPU details such as name, total memory, CUDA cores, and clock speed.
        """
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus > 0:
                for i in range(num_gpus):
                    # device = cp.cuda.Device(i)
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    name = props["name"].decode("utf-8")
                    total_mem = props["totalGlobalMem"] / (
                        1024**3
                    )  # Convert bytes to GB
                    cuda_cores = props["multiProcessorCount"]
                    clock_speed = props["clockRate"] / 1e6  # Convert to GHz

                    gpu_details = (
                        f"GPU {i}: {name} ({total_mem:.2f} GB, "
                        f"{cuda_cores} CUDA cores, {clock_speed:.2f} GHz)"
                    )
                    self.gpu_info.append(gpu_details)

                    recommended_gpu = self.recommend_gpu(name, total_mem, cuda_cores)
                    if recommended_gpu and recommended_gpu != name:
                        self.recommended_gpus.add(recommended_gpu)
        except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
            self.gpu_info.append("No GPU detected; falling back to CPU (NumPy mode).")
            self.use_gpu = False

    def recommend_gpu(self, current_gpu, memory_gb, cuda_cores):
        """
        Analyzes the current GPU performance and returns a recommended GPU upgrade if necessary.

        Parameters
        ----------
        current_gpu : str
            The name of the current GPU.
        memory_gb : float
            The total GPU memory in GB.
        cuda_cores : int
            The number of CUDA cores.

        Returns
        -------
        str or None
            The recommended GPU model, or None if no upgrade is advised.
        """
        # Do not recommend an upgrade if an A100 variant is already in use.
        if any(
            a100_variant in current_gpu
            for a100_variant in ["A100", "A100-PCIE", "A100 80GB"]
        ):
            return None

        # Do not recommend an upgrade if H100 is already in use.
        if "H100" in current_gpu:
            return None

        # Recommend a higher-end GPU if memory is less than 8GB or CUDA cores are insufficient.
        if memory_gb < 8 or cuda_cores < 2500:
            return RECOMMENDED_GPUS["Deep Learning"][0]
        elif "GTX" in current_gpu or "RTX 2060" in current_gpu:
            return RECOMMENDED_GPUS["General Computing"][0]
        elif "Tesla" in current_gpu and memory_gb < 32:
            return RECOMMENDED_GPUS["High Performance Computing"][0]
        return None

    def check_cpu(self):
        """
        Retrieves CPU core count and total system memory.
        """
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.total_memory = psutil.virtual_memory().total / (
            1024**3
        )  # Convert bytes to GB

    def check_python_libraries(self):
        """
        Retrieves version information for Python and key libraries (NumPy, CuPy, PyTorch).
        """
        self.python_version = platform.python_version()
        self.numpy_version = np.__version__
        self.cupy_version = cp.__version__ if cp else "Not Installed"

    def log_environment_analysis(self):
        """
        Logs detailed environment analysis if verbose mode is enabled.
        """
        if not self.verbose:
            return

        logger.info(
            f"Environment analysis based on SystemChecker update: {LAST_UPDATE_DATE}"
        )

        if self.python_version < "3.9":
            logger.warning(
                f"Python {self.python_version} is outdated. Upgrade to Python 3.9+ is recommended."
            )

        if self.torch_version == "Not Installed":
            logger.warning(
                "PyTorch is not installed. Consider installing it for deep learning applications."
            )

        logger.info(f"Python Version: {self.python_version}")
        logger.info(f"NumPy Version: {self.numpy_version}")
        logger.info(f"CuPy Version: {self.cupy_version}")
        logger.info(f"PyTorch Version: {self.torch_version}")

    def run_all_checks(self):
        """
        Executes all system checks including CPU, GPU, and library version checks.
        """
        self.check_cpu()
        self.check_gpu()
        self.check_python_libraries()
        self.log_environment_analysis()

    def log_results(self):
        """
        Logs the results of the system checks.
        """
        logger.info("Running system check...")

        # Log GPU information.
        if self.gpu_info:
            for gpu in self.gpu_info:
                logger.info(gpu)

        # Log recommended GPU upgrades.
        if self.recommended_gpus:
            for rec_gpu in self.recommended_gpus:
                logger.warning(f"Recommended GPU upgrade: {rec_gpu}")

        logger.info(f"CPU Cores: {self.cpu_cores}")
        logger.info(f"Total RAM: {self.total_memory:.2f} GB")
        logger.info(f"GPU Usage Enabled: {self.use_gpu}")
        logger.info("System check complete.")

    def get_use_gpu(self):
        """
        Returns the current GPU usage setting.

        Returns
        -------
        bool
            True if GPU usage is enabled, False otherwise.
        """
        return self.use_gpu

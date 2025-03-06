#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: config.py

"""
Configuration module for Structura.

This module defines simulation constants for the IllustrisTNG simulation and loads configuration files containing
user-defined settings as well as TNG-specific parameters. It also configures GPU usage, validates file paths for input
and output data, and provides a utility function for GPU memory management.
"""

import os
import yaml
import cupy as cp
from logger import logger
from pathlib import Path

# Simulation Constants (Fixed)
# IllustrisTNG Simulation Box dimensions
LBOX_MPC = 302.6  # Comoving Mpc/h (cMpc/h)
LBOX_CKPCH = 205000  # ckpc/h

# Default Grid Resolution
DEFAULT_RESOLUTIONS = 1  # cMpc/h
DEFAULT_GRID_SIZE = 302  # Grid points per axis

# Load User Configurations
# Set the path to 'etc/config.yml' relative to the current file's directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "etc", "config.yml")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as file:
    user_config = yaml.safe_load(file)

# Load TNG Configurations
# Set the path to 'etc/IllustrisTNG_config.yml' relative to the current file's directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "etc", "IllustrisTNG_config.yml")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as file:
    tng_config = yaml.safe_load(file)

# Histogram settings (User Configurable)
DEFAULT_BINS = user_config["DEFAULT_BINS"]

# GPU Settings
USE_GPU = user_config["USE_GPU"]
GPU_DEVICE = user_config["GPU_DEVICE"]

if USE_GPU:
    try:
        cp.cuda.Device(GPU_DEVICE).use()
        logger.info(f"Using GPU device {GPU_DEVICE}.")
    except cp.cuda.runtime.CUDARuntimeError:
        logger.warning(
            f"GPU device {GPU_DEVICE} not accessible. Falling back to default device."
        )
        cp.cuda.Device(0).use()
        logger.info("Using GPU device 0 as fallback.")

# File Paths (User Configurable)
INPUT_DATA_PATHS = user_config["INPUT_DATA_PATHS"]
OUTPUT_DATA_PATHS = user_config["OUTPUT_DATA_PATHS"]

# Validate existence and write permission for output directories.
for key, path in OUTPUT_DATA_PATHS.items():
    if not os.path.exists(path):
        logger.warning(f"Output directory '{key}' does not exist: {path}")
    else:
        if os.access(path, os.W_OK):
            logger.info(f"Output directory '{key}' is set to: {path}")
        else:
            logger.error(f"Permission denied: Cannot write to output directory '{key}'")

# Define the results directory relative to the module's parent directory.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "src/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu_memory():
    """
    Clears all GPU memory pools managed by CuPy.

    This function frees all memory blocks held by both the default memory pool and the pinned memory pool.
    It logs an informational message upon successful completion.
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    logger.info("GPU memory cleared.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: config.py

import os
import cupy as cp
from structura.logger import logger  # 공통 로거 가져오기

# -------------------------------
# 🔹 Simulation box settings (TNG300-1)
# -------------------------------
LBOX_CMPCH = 302.6   # Comoving Mpc/h (cMpc/h)
LBOX_CKPCH = 205000  # ckpc/h

# -------------------------------
# 🔹 Grid resolution (1 Mpc/h per grid cell)
# -------------------------------
DEFAULT_RESOLUTIONS = 1  # cMpc/h
DEFAULT_GRID_SIZE = int(LBOX_CMPCH / DEFAULT_RESOLUTIONS)  # 302 grid points per axis

# -------------------------------
# 🔹 Histogram settings
# -------------------------------
DEFAULT_BINS = DEFAULT_GRID_SIZE  # Auto bins based on resolution

# -------------------------------
# 🔹 GPU settings
# -------------------------------
USE_GPU = True
try:
    cp.cuda.Device(1).use()
    logger.info("✅ Using GPU device 1.")
except cp.cuda.runtime.CUDARuntimeError:
    logger.warning("⚠️ GPU device 1 not accessible. Using default device.")
    cp.cuda.Device(0).use()
    logger.info("✅ Using GPU device 0 as fallback.")

# -------------------------------
# 🔹 Ensure results folder exists (Always save in Structura/results/)
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Structura root directory
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "results")  # Always save results here
os.makedirs(RESULTS_FOLDER, exist_ok=True)

logger.info(f"📁 Results will be saved in: {RESULTS_FOLDER}")

# -------------------------------
# 🔹 GPU memory management
# -------------------------------
def clear_gpu_memory():
    """Clears GPU memory pools."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    logger.info("✅ GPU memory cleared.")

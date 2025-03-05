#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: config.py

import os
import yaml
import cupy as cp
from logger import logger
from pathlib import Path

# -------------------------------
# 🔹 Simulation Constants (Fixed)
# -------------------------------
# IllustrisTNG Simulation Box
LBOX_CMPCH = 302.6   # Comoving Mpc/h (cMpc/h)
LBOX_CKPCH = 205000  # ckpc/h

# Default Grid Resolution
DEFAULT_RESOLUTIONS = 1  # cMpc/h
DEFAULT_GRID_SIZE = 302  # Grid points per axis

# -------------------------------
# 🔹 Load User Configurations
# -------------------------------
# 🔍 src/ 폴더의 상위 디렉토리를 기준으로 etc/config.yml 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/ 기준
CONFIG_PATH = os.path.join(BASE_DIR, "etc", "config.yml")  # src/etc/config.yml

# ✅ config.yml 파일 존재 여부 확인
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

# 📂 YAML 파일 로드
with open(CONFIG_PATH, "r") as file:
    user_config = yaml.safe_load(file)


# -------------------------------
# 🔹 Histogram settings (User Configurable)
# -------------------------------
DEFAULT_BINS = user_config["DEFAULT_BINS"]

# -------------------------------
# 🔹 GPU settings
# -------------------------------
USE_GPU = user_config["USE_GPU"]
GPU_DEVICE = user_config["GPU_DEVICE"]

if USE_GPU:
    try:
        cp.cuda.Device(GPU_DEVICE).use()
        logger.info(f"✅ Using GPU device {GPU_DEVICE}.")
    except cp.cuda.runtime.CUDARuntimeError:
        logger.warning(f"⚠️ GPU device {GPU_DEVICE} not accessible. Using default device.")
        cp.cuda.Device(0).use()
        logger.info("✅ Using GPU device 0 as fallback.")

# -------------------------------
# 🔹 File Paths (User Configurable)
# -------------------------------
# 🔹 File Paths
INPUT_DATA_PATHS = user_config["INPUT_DATA_PATHS"]
OUTPUT_DATA_PATHS = user_config["OUTPUT_DATA_PATHS"]

# ✅ 출력 디렉터리 존재 여부 및 접근 권한 확인
for key, path in OUTPUT_DATA_PATHS.items():
    if not os.path.exists(path):
        logger.warning(f"⚠️ Output directory '{key}' does not exist: {path}")
    else:
        # ✅ 접근 권한 확인
        if os.access(path, os.W_OK):
            logger.info(f"📁 Output directory '{key}' is set to: {path}")
        else:
            logger.error(f"❌ Permission denied: Cannot write to output directory '{key}'")


# 현재 파일(모듈)의 위치를 기준으로 Structura 내부 results 폴더 경로 지정
RESULTS_DIR = Path(__file__).resolve().parent.parent / "src/results"

# 경로가 없으면 생성
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 🔹 GPU memory management
# -------------------------------
def clear_gpu_memory():
    """Clears GPU memory pools."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    logger.info("✅ GPU memory cleared.")


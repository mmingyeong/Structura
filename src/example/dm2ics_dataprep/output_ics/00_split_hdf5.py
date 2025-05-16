#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_ics_chunks.py

Splits a large HDF5 dataset (e.g., from IllustrisTNG initial conditions)
into smaller HDF5 chunks for efficient parallel access.

Author: Mingyeong Yang
Date: 2025-04-22
"""

import os
import sys
import logging

# Add parent directory (src) to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from convert import SimulationDataConverter

# ====== 사용자 설정 ======
INPUT_FILE = "/caefs/data/IllustrisTNG/ics.hdf5"
OUTPUT_DIR = "/caefs/data/IllustrisTNG/snapshot-0-ics"
DATASET_NAME = "PartType1/Coordinates"  # 사용자에 따라 다를 수 있음
CHUNK_SIZE = 10_000_000  # 예: 1천만 개씩 나누기

# ====== 로깅 설정 ======
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file_path = os.path.join(OUTPUT_DIR, "split_chunks.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ====== 실행 ======
if __name__ == "__main__":
    logger.info("Starting HDF5 chunk split...")
    
    converter = SimulationDataConverter(
        input_path=INPUT_FILE,
        output_folder=OUTPUT_DIR,
        chunk_size=None,
        use_gpu=False  # GPU 가속이 필요 없다면 False
    )

    # 선택된 데이터셋 기준으로 분할 저장
    converter.split_hdf5_to_chunks(dataset_name=DATASET_NAME)
    
    logger.info("Chunk split complete.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: converter_ex.py

import os
import sys
import time

# 🔧 현재 스크립트의 상위 디렉터리(src)를 Python 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import logger
from convert import SimulationDataConverter
from config import INPUT_DATA_PATHS, OUTPUT_DIRECTORIES
from system_checker import SystemChecker  # ✅ 시스템 체크 (선택사항)

# ✅ 시스템 체크 실행 (원한다면 활성화)
checker = SystemChecker(verbose=True)
# checker.run_all_checks()
# checker.log_results()

use_gpu = checker.get_use_gpu()
logger.info(f"🚀 Using GPU: {use_gpu}")

start_time = time.time()  # 시작 시간 저장

# 🔹 HDF5 파일 경로 및 변환된 파일 저장 경로
hdf5_file_path = INPUT_DATA_PATHS["HDF5"]
output_folder = OUTPUT_DIRECTORIES["NPY"]

logger.info(f"📂 Input HDF5 file: {hdf5_file_path}")
logger.info(f"📁 Output folder: {output_folder}")

# ✅ 변환된 데이터가 이미 존재하는지 확인
try:
    output_files = [f for f in os.listdir(output_folder) if f.endswith((".npy", ".npz"))]
except PermissionError:
    logger.error(f"❌ Permission denied: Cannot access output folder {output_folder}")
    logger.info("⏩ Skipping conversion due to permission issue.")
    sys.exit(0)

if output_files:
    file_count = len(output_files)
    total_size = sum(os.path.getsize(os.path.join(output_folder, f)) for f in output_files)
    avg_size = total_size / file_count if file_count > 0 else 0

    total_size_mb = total_size / (1024 * 1024)  # MB 단위 변환
    avg_size_mb = avg_size / (1024 * 1024)     # MB 단위 변환

    # ✅ 예시로, 첫 번째 파일 이름을 e.g. 부분에 넣어주기 (e.g., 'chunk_89.npy')
    example_file = output_files[0] if output_files else "chunk_??.npy"

    # ✅ 여기서 보여주신 예시처럼 로그 출력
    logger.info(f"⚠️ Output folder '{output_folder}' contains {file_count} files (e.g., '{example_file}').")
    logger.info(f"📊 Total size: {total_size_mb:.2f} MB, Average file size: {avg_size_mb:.2f} MB")
    logger.info("⏩ Skipping conversion process.")
    sys.exit(0)

# 🚀 변환 실행
try:
    logger.info("🚀 Starting HDF5 to NPY conversion...")
    converter = SimulationDataConverter(hdf5_file_path, output_folder, use_gpu=use_gpu)
    converter.convert_hdf5(npyornpz="npy")
    logger.info(f"✅ Conversion completed. Output saved in: {output_folder}")
except Exception as e:
    logger.error(f"❌ Error during conversion: {e}")
    sys.exit(1)

end_time = time.time()
elapsed_time = end_time - start_time

formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
logger.info(f"⏳ Total execution time: {formatted_time} ({elapsed_time:.2f} seconds)")
logger.info("✅ HDF5 to NPY conversion completed successfully!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: converter_ex.py

import os
import sys
import time

# Append the parent directory (src) to the Python module search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import logger
from convert import SimulationDataConverter
from config import INPUT_DATA_PATHS, OUTPUT_DATA_PATHS
from system_checker import SystemChecker  # Optional system check

# Optionally, perform a system check.
checker = SystemChecker(verbose=True)
# Uncomment the following lines if detailed system analysis is desired:
# checker.run_all_checks()
# checker.log_results()

use_gpu = checker.get_use_gpu()
logger.info(f"Using GPU: {use_gpu}")

start_time = time.time()  # Record the start time.

# Define the HDF5 input file path and the output folder for the converted files.
hdf5_file_path = INPUT_DATA_PATHS["HDF5"]
output_folder = OUTPUT_DATA_PATHS["NPY"]

logger.info(f"Input HDF5 file: {hdf5_file_path}")
logger.info(f"Output folder: {output_folder}")

# Check if converted data already exists in the output folder.
try:
    output_files = [
        f for f in os.listdir(output_folder) if f.endswith((".npy", ".npz"))
    ]
except PermissionError:
    logger.error(f"Permission denied: Cannot access output folder {output_folder}")
    logger.info("Skipping conversion due to permission issues.")
    sys.exit(0)

if output_files:
    file_count = len(output_files)
    total_size = sum(
        os.path.getsize(os.path.join(output_folder, f)) for f in output_files
    )
    avg_size = total_size / file_count if file_count > 0 else 0

    total_size_mb = total_size / (1024 * 1024)  # Convert to MB.
    avg_size_mb = avg_size / (1024 * 1024)  # Convert to MB.

    example_file = output_files[0] if output_files else "chunk_??.npy"

    logger.info(
        f"Output folder '{output_folder}' contains {file_count} files (e.g., '{example_file}')."
    )
    logger.info(
        f"Total size: {total_size_mb:.2f} MB, Average file size: {avg_size_mb:.2f} MB"
    )
    logger.info("Skipping conversion process as output data already exists.")
    sys.exit(0)

# Execute the conversion process.
try:
    logger.info("Starting HDF5 to NPY conversion...")
    converter = SimulationDataConverter(hdf5_file_path, output_folder, use_gpu=use_gpu)
    converter.convert_hdf5(npyornpz="npy")
    logger.info(f"Conversion completed. Output saved in: {output_folder}")
except Exception as e:
    logger.error(f"Error during conversion: {e}")
    sys.exit(1)

end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
logger.info(f"Total execution time: {formatted_time} ({elapsed_time:.2f} seconds)")
logger.info("HDF5 to NPY conversion completed successfully.")

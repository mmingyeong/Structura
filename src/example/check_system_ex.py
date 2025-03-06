#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-04
# @Filename: check_system_ex.py

import os
import sys
import time  # Used for measuring execution time

# Add the parent directory (src) of the current script to the Python module search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import logger
from system_checker import SystemChecker, LAST_UPDATE_DATE

if __name__ == "__main__":
    logger.info(
        f"Starting system check (SystemChecker last updated: {LAST_UPDATE_DATE})"
    )

    start_time = time.time()  # Record the start time

    # Instantiate and run the SystemChecker with verbose mode enabled for detailed analysis.
    checker = SystemChecker(verbose=True)
    checker.run_all_checks()
    checker.log_results()

    # Check and log the GPU availability.
    use_gpu = checker.get_use_gpu()
    if use_gpu:
        logger.info("GPU is available. Computation will be accelerated.")
    else:
        logger.warning(
            "No GPU detected. Using CPU. Computation may be significantly slower."
        )

    # Measure and log the elapsed time for the system check.
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info(
        f"System check completed in {formatted_time} ({elapsed_time:.2f} seconds)."
    )

    # Terminate the script explicitly.
    sys.exit(0)

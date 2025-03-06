#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: logger.py

"""
Logger configuration for Structura.

This module establishes a logger that directs log messages to both the console and a file.
The log file is created in a dedicated 'log' directory relative to the module location,
and its filename is based on the executing script's name and the current timestamp.
Unhandled exceptions are automatically captured and logged.
"""

import logging
import os
import sys
from datetime import datetime

# Set the base directory to the directory containing this module (e.g., Structura/src)
BASE_DIR = os.path.dirname(__file__)
# Define the directory for log files (e.g., Structura/src/log)
LOG_DIR = os.path.join(BASE_DIR, "log")

# Create the log directory if it does not already exist
os.makedirs(LOG_DIR, exist_ok=True)

# Determine the name of the executing script; default to 'main' if unavailable.
if len(sys.argv) > 0:
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
else:
    script_name = "main"

# Generate a log filename based on the current date and time (e.g., converter_ex_2025-03-05_00-39-53.log)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{script_name}_{timestamp}.log"
LOG_FILE = os.path.join(LOG_DIR, log_filename)

# Configure the logger with the name 'Structura' and a global log level of DEBUG.
logger = logging.getLogger("Structura")
logger.setLevel(
    logging.INFO
)  # Collect all log messages; filtering is applied in individual handlers.

# Define the log message format.
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler: display messages with level INFO and above.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler: write messages with level DEBUG and above to the log file.
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Prevent duplicate handlers from being added.
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Redirect standard error output to standard output.
sys.stderr = sys.stdout


def log_exception(exc_type, exc_value, exc_traceback):
    """
    Logs unhandled exceptions to the log file.

    Parameters
    ----------
    exc_type : type
        The type of the exception.
    exc_value : Exception
        The exception instance.
    exc_traceback : traceback
        The traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error(
        "Unhandled exception occurred:", exc_info=(exc_type, exc_value, exc_traceback)
    )


# Set the global exception hook to ensure that unhandled exceptions are logged.
sys.excepthook = log_exception

if __name__ == "__main__":
    logger.info("Logger is successfully set up!")

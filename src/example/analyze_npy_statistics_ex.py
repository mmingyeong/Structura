#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: analyze_npy_statistics_ex.py

import os
import numpy as np
from config import RESULTS_DIR

# Ensure the results directory exists; create it if necessary.
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# List to store output lines.
output_lines = []


def log_line(line):
    """
    Print the given line to the console and store it in the output list.

    Parameters
    ----------
    line : str
        The text line to be printed and stored.
    """
    print(line)
    output_lines.append(line)


# Define the folder containing the data to be analyzed.
# For example, this folder holds snapshot-99 data.
test_folder = "/caefs/data/IllustrisTNG/snapshot-99-dm-npy/cache"


# Define a filter function to select only files with the pattern "snapshot-99.*.npy".
def file_filter(f):
    return f.startswith("snapshot-99.") and f.endswith(".npy")


# Retrieve a sorted list of npy files in the specified folder that match the filter.
npy_files = sorted([f for f in os.listdir(test_folder) if file_filter(f)])
if not npy_files:
    raise FileNotFoundError(
        f"No npy files matching the criteria were found in {test_folder}."
    )

n_total = len(npy_files)
log_line(f"Total number of files: {n_total}")

# Always include the first and last file indices; also, select 10 evenly spaced indices from the middle.
if n_total > 2:
    middle_indices = np.linspace(1, n_total - 2, num=10, dtype=int).tolist()
else:
    middle_indices = []

# Combine indices: first file, middle samples, and last file.
selected_indices = [0] + middle_indices + [n_total - 1]
log_line("Selected file indices: " + str(selected_indices))
log_line("")

# Compute and log statistics for each selected file.
for idx in selected_indices:
    filename = npy_files[idx]
    file_path = os.path.join(test_folder, filename)
    data = np.load(file_path)

    # If the data is one-dimensional, reshape it into a two-dimensional array.
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    log_line(f"File index {idx} - {filename}")
    log_line(f"  Data shape: {data.shape}")

    n_columns = data.shape[1]
    for col in range(n_columns):
        col_data = data[:, col]
        col_min = np.min(col_data)
        col_max = np.max(col_data)
        col_mean = np.mean(col_data)
        log_line(f"    Column {col}: min={col_min}, max={col_max}, mean={col_mean}")
    log_line("-" * 60)

# Save the collected statistics to a text file.
output_file_path = os.path.join(RESULTS_DIR, "data_statistics.txt")
with open(output_file_path, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

log_line(f"\nBasic statistical information has been saved to '{output_file_path}'.")

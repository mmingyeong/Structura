#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: convert.py
# structura/convert.py

import h5py
import numpy as np
import cupy as cp
import os
import psutil
import struct
from tqdm import tqdm
from logger import logger
from multiprocessing import Pool, cpu_count


class SimulationDataConverter:
    """
    Converts various cosmological simulation formats (HDF5, GADGET, ASCII) to .npy or .npz.

    This class supports conversion of simulation data files into NumPy binary formats,
    with optional GPU acceleration and parallel processing.
    """

    def __init__(
        self,
        input_path,
        output_folder,
        chunk_size=None,
        num_processes=None,
        size_threshold=500 * 1e6,
        use_gpu=True,
    ):
        """
        Parameters
        ----------
        input_path : str
            Path to the input data file or folder.
        output_folder : str
            Folder where the converted .npy or .npz files will be stored.
        chunk_size : int, optional
            Number of rows per chunk. If None, an optimal value is determined automatically.
        num_processes : int, optional
            Number of parallel processes to use (default: number of CPU cores).
        size_threshold : int, optional
            File size threshold in bytes below which conversion is skipped (default: 500 MB).
        use_gpu : bool, optional
            If True, CuPy is utilized for GPU acceleration (default: True).
        """
        self.input_path = input_path
        self.output_folder = output_folder
        self.use_gpu = use_gpu
        self.num_processes = num_processes or cpu_count()
        self.size_threshold = (
            size_threshold  # Threshold for skipping conversion (default: 500 MB)
        )
        os.makedirs(self.output_folder, exist_ok=True)

        # Automatically detect data format.
        self.data_format = self.detect_format()
        if not self.data_format:
            raise ValueError("Unsupported data format.")

        # Automatically determine optimal chunk_size.
        self.chunk_size = chunk_size or self.get_optimal_chunk_size()
        logger.info(f"Automatically determined chunk_size: {self.chunk_size:,}")

        # Determine file size with exception handling.
        try:
            file_size = os.path.getsize(self.input_path)
            logger.info(f"File size: {file_size / 1e6:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to determine file size: {e}")
            file_size = float("inf")  # Proceed with conversion in case of error

        # Decide whether conversion is needed.
        self.needs_conversion = file_size >= self.size_threshold
        if not self.needs_conversion:
            logger.info(
                "File size is below threshold; using original data without conversion."
            )

    def get_optimal_chunk_size(self):
        """
        Automatically determines the optimal chunk size based on available RAM and I/O performance.

        Returns
        -------
        int
            The determined chunk size (number of rows).
        """
        total_ram = psutil.virtual_memory().total  # Total RAM in bytes
        if total_ram < 16 * 1e9:  # Less than 16 GB RAM
            return 1_000_000  # 1 million entries
        elif total_ram < 64 * 1e9:  # Between 16 GB and 64 GB RAM
            return 10_000_000  # 10 million entries
        else:  # 64 GB RAM or more
            return 50_000_000  # 50 million entries

    def detect_format(self):
        """
        Detects the input file format based on its extension.

        Returns
        -------
        str or None
            The detected format ("HDF5", "GADGET", "ASCII", "NUMPY"), or None if unsupported.
        """
        if self.input_path.endswith((".hdf5", ".h5")):
            return "HDF5"
        elif self.input_path.endswith((".bin", ".dat")):
            return "GADGET"
        elif self.input_path.endswith((".csv", ".txt")):
            return "ASCII"
        elif self.input_path.endswith((".npy", ".npz")):
            return "NUMPY"
        return None

    def convert(self, npyornpz="npy"):
        """
        Converts the detected simulation data format into .npy or .npz.

        Parameters
        ----------
        npyornpz : str, optional
            Specifies the target format ('npy' or 'npz'). Default is 'npy'.
        """
        if not self.needs_conversion:
            logger.info("Using original data without conversion.")
            return

        logger.info(
            f"Converting {self.input_path} ({self.data_format}) to {npyornpz.upper()} format..."
        )

        if self.data_format == "HDF5":
            self.convert_hdf5(npyornpz)
        elif self.data_format == "GADGET":
            self.convert_gadget(npyornpz)
        elif self.data_format == "ASCII":
            self.convert_ascii(npyornpz)
        elif self.data_format == "NUMPY":
            logger.info("No conversion needed; data is already in NumPy format.")
        else:
            raise ValueError("Unsupported data format.")

    def _find_hdf5_datasets(self):
        """
        Returns a list of datasets contained within the HDF5 file, with progress display.

        Returns
        -------
        list of str
            List of dataset paths within the HDF5 file.
        """
        datasets = []
        with h5py.File(self.input_path, "r") as hdf5_file:
            groups = list(hdf5_file.keys())
            for group in tqdm(groups, desc="Searching HDF5 datasets"):
                if isinstance(hdf5_file[group], h5py.Group):
                    for dataset in hdf5_file[group].keys():
                        full_path = f"{group}/{dataset}"
                        datasets.append(full_path)
                        logger.info(f"Found dataset: {full_path}")

        return datasets

    def convert_hdf5(self, npyornpz, dataset_name=None):
        """
        Converts HDF5 data to .npy or .npz format with an option for user dataset selection.

        Parameters
        ----------
        npyornpz : str
            Target format ('npy' or 'npz').
        dataset_name : str, optional
            Specific dataset to convert. If None, the user is prompted to select from available datasets.
        """
        if dataset_name is None:
            available_datasets = self._find_hdf5_datasets()
            if not available_datasets:
                raise ValueError("No valid datasets found in the HDF5 file.")

            logger.info("Available datasets:")
            for idx, ds in enumerate(available_datasets):
                logger.info(f"  [{idx}] {ds}")

            try:
                selected_idx = int(
                    input("Enter the index number of the dataset to convert: ")
                )
                dataset_name = available_datasets[selected_idx]
            except (ValueError, IndexError):
                raise ValueError("Invalid input. Please enter a valid dataset index.")

        logger.info(f"Selected dataset: {dataset_name}")

        with h5py.File(self.input_path, "r") as hdf5_file:
            dataset_size = hdf5_file[dataset_name].shape[
                0
            ]  # Determine total number of data points

        # Pass index information to the multiprocessing pool instead of the dataset itself.
        chunk_indices = list(
            tqdm(
                [
                    (
                        self.input_path,
                        dataset_name,
                        i,
                        min(i + self.chunk_size, dataset_size),
                        idx,
                        npyornpz,
                    )
                    for idx, i in enumerate(range(0, dataset_size, self.chunk_size))
                ],
                desc="Preparing chunk indices",
            )
        )

        with Pool(self.num_processes) as pool:
            list(
                tqdm(
                    pool.starmap(self._process_hdf5_chunk, chunk_indices),
                    total=len(chunk_indices),
                    desc="Converting HDF5 chunks",
                )
            )

    def _process_hdf5_chunk(
        self, hdf5_path, dataset_name, start_idx, end_idx, chunk_id, npyornpz
    ):
        """
        Processes a chunk of an HDF5 file and saves it in the specified format.

        Parameters
        ----------
        hdf5_path : str
            Path to the HDF5 file.
        dataset_name : str
            Name of the dataset within the file.
        start_idx : int
            Starting index of the chunk.
        end_idx : int
            Ending index of the chunk.
        chunk_id : int
            Identifier for the current chunk.
        npyornpz : str
            Target format ('npy' or 'npz').
        """
        # Reopen the HDF5 file for processing this chunk.
        with h5py.File(hdf5_path, "r") as f:
            dataset = f[dataset_name]
            chunk = dataset[start_idx:end_idx]

        if self.use_gpu:
            chunk = cp.asarray(
                chunk
            )  # Convert NumPy array to CuPy array for GPU acceleration.
            logger.info(f"GPU acceleration enabled for chunk {chunk_id}")

        chunk_file = os.path.join(self.output_folder, f"chunk_{chunk_id}.{npyornpz}")
        if npyornpz == "npy":
            np.save(chunk_file, chunk)
        else:
            np.savez_compressed(chunk_file, data=chunk)

        logger.info(f"Saved chunk {chunk_id} to {chunk_file}")

    def convert_gadget(self, npyornpz):
        """
        Converts GADGET binary data to .npy or .npz format.

        Parameters
        ----------
        npyornpz : str
            Target format ('npy' or 'npz').
        """
        with open(self.input_path, "rb") as f:
            # Dynamically determine header size.
            header_size = struct.unpack("I", f.read(4))[0]
            f.seek(header_size)
            data = np.fromfile(f, dtype=np.float32).reshape(-1, 3)

        output_file = os.path.join(self.output_folder, f"gadget_converted.{npyornpz}")
        if npyornpz == "npy":
            np.save(output_file, data)
        else:
            np.savez_compressed(output_file, data=data)
        logger.info(f"Converted GADGET data to {npyornpz.upper()} format.")

    def convert_ascii(self, npyornpz):
        """
        Converts ASCII (CSV or TXT) data to .npy or .npz format.

        Parameters
        ----------
        npyornpz : str
            Target format ('npy' or 'npz').
        """
        data = np.loadtxt(
            self.input_path, delimiter=None
        )  # Automatically detect delimiter.
        output_file = os.path.join(self.output_folder, f"converted.{npyornpz}")

        if npyornpz == "npy":
            np.save(output_file, data)
        else:
            np.savez_compressed(output_file, data=data)
        logger.info(f"Converted ASCII data to {npyornpz.upper()} format.")

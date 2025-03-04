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
from logger import logger  # 공통 로거 가져오기
from multiprocessing import Pool, cpu_count

class SimulationDataConverter:
    """Converts various cosmological simulation formats (HDF5, GADGET, ASCII) to .npy or .npz."""

    def __init__(self, input_path, output_folder, chunk_size=None, num_processes=None, size_threshold=500 * 1e6, use_gpu=True):
        """
        Parameters
        ----------
        input_path : str
            Path to the input data file or folder.
        output_folder : str
            Folder where the converted .npy or .npz files will be stored.
        chunk_size : int, optional
            Number of rows per chunk. If None, it is determined automatically.
        num_processes : int, optional
            Number of parallel processes (default: CPU count).
        size_threshold : int, optional
            File size threshold (in bytes) under which conversion is skipped.
        use_gpu : bool, optional
            If True, CuPy will be used for GPU acceleration (default: True).
        """
        self.input_path = input_path
        self.output_folder = output_folder
        self.use_gpu = use_gpu
        self.num_processes = num_processes or cpu_count()
        self.size_threshold = size_threshold  # 변환 생략 기준 (기본값: 500MB)
        os.makedirs(self.output_folder, exist_ok=True)

        # 데이터 형식 자동 감지
        self.data_format = self.detect_format()
        if not self.data_format:
            raise ValueError("❌ Unsupported data format!")

        # 최적의 chunk_size 자동 설정
        self.chunk_size = chunk_size or self.get_optimal_chunk_size()
        logger.info(f"🔧 자동 설정된 chunk_size: {self.chunk_size:,}")

        # 파일 크기 확인 (예외 처리 추가)
        try:
            file_size = os.path.getsize(self.input_path)
            logger.info(f"📂 파일 크기: {file_size / 1e6:.2f} MB")
        except Exception as e:
            logger.error(f"❌ 파일 크기 확인 실패: {e}")
            file_size = float("inf")  # 예외 발생 시 변환 수행

        # 변환 여부 결정
        self.needs_conversion = file_size >= self.size_threshold
        if not self.needs_conversion:
            logger.info("✅ 파일 크기가 작아서 변환 없이 원본 사용")

    def get_optimal_chunk_size(self):
        """RAM 용량과 I/O 속도를 고려하여 자동으로 chunk_size 설정"""
        total_ram = psutil.virtual_memory().total  # 전체 RAM 용량 (bytes)
        if total_ram < 16 * 1e9:  # RAM 16GB 미만
            return 1_000_000  # 1M 개
        elif total_ram < 64 * 1e9:  # RAM 16GB ~ 64GB
            return 10_000_000  # 10M 개
        else:  # RAM 64GB 이상
            return 50_000_000  # 50M 개

    def detect_format(self):
        """Detects the input file format."""
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
        """Converts the detected format into .npy or .npz."""

        if not self.needs_conversion:
            logger.info("✅ 변환 없이 원본 사용.")
            return
        
        # 변환 수행
        logger.info(f"🔄 Converting {self.input_path} ({self.data_format}) to {npyornpz.upper()}...")

        if self.data_format == "HDF5":
            self.convert_hdf5(npyornpz)
        elif self.data_format == "GADGET":
            self.convert_gadget(npyornpz)
        elif self.data_format == "ASCII":
            self.convert_ascii(npyornpz)
        elif self.data_format == "NUMPY":
            logger.info("✅ No conversion needed. Already in NumPy format.")
        else:
            raise ValueError("❌ Unsupported data format!")

    def _find_hdf5_datasets(self):
        """HDF5 내부의 데이터셋 목록을 반환 (진행률 표시 포함)"""
        datasets = []
        with h5py.File(self.input_path, "r") as hdf5_file:
            groups = list(hdf5_file.keys())
            for group in tqdm(groups, desc="🔍 Searching HDF5 datasets"):
                if isinstance(hdf5_file[group], h5py.Group):
                    for dataset in hdf5_file[group].keys():
                        full_path = f"{group}/{dataset}"
                        datasets.append(full_path)
                        logger.info(f"📌 Found dataset: {full_path}")

        return datasets

    def convert_hdf5(self, npyornpz, dataset_name=None):
        """Converts HDF5 data to .npy or .npz with user dataset selection."""
        
        if dataset_name is None:
            available_datasets = self._find_hdf5_datasets()
            
            if not available_datasets:
                raise ValueError("❌ HDF5 파일에서 유효한 데이터셋을 찾을 수 없음!")

            # 사용자가 직접 선택하도록 옵션 제공
            logger.info("💡 Available datasets:")
            for idx, ds in enumerate(available_datasets):
                logger.info(f"  [{idx}] {ds}")

            # CLI 환경에서 실행할 경우
            try:
                selected_idx = int(input("👉 변환할 데이터셋의 번호를 입력하세요: "))
                dataset_name = available_datasets[selected_idx]
            except (ValueError, IndexError):
                raise ValueError("❌ 잘못된 입력입니다. 올바른 번호를 입력하세요.")

        logger.info(f"✅ 선택된 데이터셋: {dataset_name}")

        with h5py.File(self.input_path, "r") as hdf5_file:
            dataset_size = hdf5_file[dataset_name].shape[0]  # 총 데이터 개수 확인

        # 🔥 dataset을 `multiprocessing.Pool`에 전달하지 않고, 인덱스 정보만 전달
        chunk_indices = list(
            tqdm(
                [(self.input_path, dataset_name, i, min(i + self.chunk_size, dataset_size), idx, npyornpz)
                for idx, i in enumerate(range(0, dataset_size, self.chunk_size))],
                desc="🛠 Preparing chunk indices"
            )
        )

        with Pool(self.num_processes) as pool:
            list(tqdm(pool.starmap(self._process_hdf5_chunk, chunk_indices), total=len(chunk_indices), desc="🚀 Converting HDF5 chunks"))

    def _process_hdf5_chunk(self, hdf5_path, dataset_name, start_idx, end_idx, chunk_id, npyornpz):
        """Processes a chunk of an HDF5 file with tqdm progress tracking."""
        with h5py.File(hdf5_path, "r") as f:  # 🔥 여기서 다시 HDF5 파일을 열어야 함!
            dataset = f[dataset_name]
            chunk = dataset[start_idx:end_idx]

        if self.use_gpu:
            chunk = cp.asarray(chunk)  # ✅ NumPy → CuPy 변환
            logger.info(f"🚀 GPU enabled for chunk {chunk_id}")

        chunk_file = os.path.join(self.output_folder, f"chunk_{chunk_id}.{npyornpz}")
        if npyornpz == "npy":
            np.save(chunk_file, chunk)
        else:
            np.savez_compressed(chunk_file, data=chunk)

        logger.info(f"✅ Saved chunk {chunk_id}: {chunk_file}")

    def convert_gadget(self, npyornpz):
        """Converts GADGET binary data to .npy or .npz with tqdm tracking."""
        with open(self.input_path, "rb") as f:
            header_size = struct.unpack("I", f.read(4))[0]  # 헤더 크기 동적 감지
            f.seek(header_size)
            data = np.fromfile(f, dtype=np.float32).reshape(-1, 3)

        if npyornpz == "npy":
            np.save(os.path.join(self.output_folder, data))
        else:
            np.savez_compressed(os.path.join(self.output_folder, data=data))
        logger.info(f"✅ Converted GADGET to {npyornpz.upper()}.")


    def convert_ascii(self, npyornpz):
        """Converts ASCII (CSV or TXT) to .npy or .npz with tqdm tracking."""
        data = np.loadtxt(self.input_path, delimiter=None)  # 자동 구분자 감지
        output_file = os.path.join(self.output_folder, f"converted.{npyornpz}")

        if npyornpz == "npy":
            np.save(output_file, data)
        else:
            np.savez_compressed(output_file, data=data)
        logger.info(f"✅ Converted ASCII to {npyornpz.upper()}.")